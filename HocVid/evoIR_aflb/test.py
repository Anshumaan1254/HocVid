"""
Test/Inference Script — EvoIR Stage 1

Converted from test.ipynb with fixes:
- Bug 2:  Bottleneck uses num_fft_blocks[3]
- Bug 5:  FrequencyGuidedAttentionModule num_heads=4
- Bug 6:  Sigmoid gate multiplies in FFTransformerBlock
- Consistent AdaIR kwarg names
- Tile inference: count.clamp(min=1e-5)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
import numbers
from collections import OrderedDict
from einops import rearrange
from pathlib import Path

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')


# ===== ALL MODEL COMPONENTS =====
def to_3d(x): return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x,h,w): return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, ns):
        super().__init__()
        if isinstance(ns, numbers.Integral): ns=(ns,)
        self.weight=nn.Parameter(torch.ones(torch.Size(ns))); self.normalized_shape=torch.Size(ns)
    def forward(self, x): return x/torch.sqrt(x.var(-1,keepdim=True,unbiased=False)+1e-5)*self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, ns):
        super().__init__()
        if isinstance(ns, numbers.Integral): ns=(ns,)
        ns=torch.Size(ns); self.weight=nn.Parameter(torch.ones(ns)); self.bias=nn.Parameter(torch.zeros(ns))
    def forward(self, x):
        mu=x.mean(-1,keepdim=True); return (x-mu)/torch.sqrt(x.var(-1,keepdim=True,unbiased=False)+1e-5)*self.weight+self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, t='WithBias'):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if t=='BiasFree' else WithBias_LayerNorm(dim)
    def forward(self, x): h,w=x.shape[-2:]; return to_4d(self.body(to_3d(x)),h,w)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads=num_heads; self.temperature=nn.Parameter(torch.ones(num_heads,1,1))
        self.qkv=nn.Conv2d(dim,dim*3,1,bias=bias); self.qkv_dwconv=nn.Conv2d(dim*3,dim*3,3,1,1,groups=dim*3,bias=bias)
        self.project_out=nn.Conv2d(dim,dim,1,bias=bias)
    def forward(self, x):
        b,c,h,w=x.shape; qkv=self.qkv_dwconv(self.qkv(x)); q,k,v=qkv.chunk(3,dim=1)
        q=rearrange(q,'b (h c) H W -> b h c (H W)',h=self.num_heads)
        k=rearrange(k,'b (h c) H W -> b h c (H W)',h=self.num_heads)
        v=rearrange(v,'b (h c) H W -> b h c (H W)',h=self.num_heads)
        q=F.normalize(q,dim=-1); k=F.normalize(k,dim=-1)
        out=((q@k.transpose(-2,-1))*self.temperature).softmax(-1)@v
        return self.project_out(rearrange(out,'b h c (H W) -> b (h c) H W',h=self.num_heads,H=h,W=w))

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden=int(dim*ffn_expansion_factor)
        self.project_in=nn.Conv2d(dim,hidden*2,1,bias=bias)
        self.dwconv=nn.Conv2d(hidden*2,hidden*2,3,1,1,groups=hidden*2,bias=bias)
        self.project_out=nn.Conv2d(hidden,dim,1,bias=bias)
    def forward(self, x):
        x=self.project_in(x); x1,x2=self.dwconv(x).chunk(2,dim=1); return self.project_out(F.gelu(x1)*x2)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LN='WithBias'):
        super().__init__()
        self.norm1=LayerNorm(dim,LN); self.attn=Attention(dim,num_heads,bias)
        self.norm2=LayerNorm(dim,LN); self.ffn=FeedForward(dim,ffn_expansion_factor,bias)
    def forward(self, x): x=x+self.attn(self.norm1(x)); return x+self.ffn(self.norm2(x))

class FD(nn.Module):
    def __init__(self, ic, ks=3, group=8):
        super().__init__()
        self.ks=ks; self.group=group
        self.conv=nn.Conv2d(ic,group*ks**2,1,bias=False); self.bn=nn.BatchNorm2d(group*ks**2)
        self.act=nn.Softmax(dim=-2); nn.init.kaiming_normal_(self.conv.weight,mode='fan_out',nonlinearity='relu')
        self.pad=nn.ReflectionPad2d(ks//2); self.ap=nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        ident=x; lf=self.bn(self.conv(self.ap(x))); n,c,h,w=x.shape
        x=F.unfold(self.pad(x),kernel_size=self.ks).reshape(n,self.group,c//self.group,self.ks**2,h*w)
        n,c1,p,q=lf.shape; lf=self.act(lf.reshape(n,c1//self.ks**2,self.ks**2,p*q).unsqueeze(2))
        low=torch.sum(x*lf,dim=3).reshape(n,c,h,w); return low, ident-low

class FMgM(nn.Module):
    def __init__(self, ic, flag_highF):
        super().__init__()
        self.flag_highF=flag_highF; k=3
        if flag_highF: self.body=nn.Sequential(nn.Conv2d(ic,ic,(1,k),padding=(0,k//2),groups=ic),nn.Conv2d(ic,ic,(k,1),padding=(k//2,0),groups=ic),nn.GELU())
        else: self.body=nn.Sequential(nn.Conv2d(2*ic,2*ic,1),nn.GELU())
    def forward(self, ffm):
        if self.flag_highF: return self.body(ffm)*ffm
        bs,c,H,W=ffm.shape; y=torch.fft.rfft2(ffm.float()); y_f=torch.cat([y.real,y.imag],1)
        y_f=y_f*self.body(y_f); yr,yi=y_f.chunk(2,1); return torch.fft.irfft2(torch.complex(yr,yi),s=(H,W))

class FrequencyGuidedAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        self.num_heads=num_heads; self.temperature=nn.Parameter(torch.ones(num_heads,1,1))
        self.q_proj=nn.Conv2d(dim,dim,1,bias=bias); self.kv_proj=nn.Conv2d(dim,dim*2,1,bias=bias)
        self.q_dwconv=nn.Conv2d(dim,dim,3,padding=1,groups=dim,bias=bias)
        self.kv_dwconv=nn.Conv2d(dim*2,dim*2,3,padding=1,groups=dim*2,bias=bias)
        self.project_out=nn.Conv2d(dim,dim,1,bias=bias)
    def forward(self, x, ff):
        B,C,H,W=x.shape; q=self.q_dwconv(self.q_proj(ff)); kv=self.kv_dwconv(self.kv_proj(x)); k,v=kv.chunk(2,1)
        q=rearrange(q,'b (h c) H W->b h c (H W)',h=self.num_heads); k=rearrange(k,'b (h c) H W->b h c (H W)',h=self.num_heads)
        v=rearrange(v,'b (h c) H W->b h c (H W)',h=self.num_heads)
        q=F.normalize(q,-1); k=F.normalize(k,-1)
        return self.project_out(rearrange((q@k.transpose(-2,-1)*self.temperature).softmax(-1)@v,'b h c (H W)->b (h c) H W',h=self.num_heads,H=H,W=W))

class FrequencyGuidedAttentionModule(nn.Module):
    def __init__(self, ic, oc, decoder_flag=False):
        super().__init__()
        self.decoder_flag = decoder_flag
        # FIX Bug 5: num_heads=4, NOT oc
        self.hfa=FrequencyGuidedAttention(ic,num_heads=4); self.lfa=FrequencyGuidedAttention(ic,num_heads=4)
        if decoder_flag: self.alpha=nn.Parameter(torch.tensor(0.5)); self.proj=nn.Conv2d(ic,oc,1)
        else: self.final_proj=nn.Conv2d(ic*2,oc,1)
    def forward(self, x, hf, lf):
        ho=self.hfa(x,hf); lo=self.lfa(x,lf)
        if self.decoder_flag: a=torch.sigmoid(self.alpha); return self.proj(a*ho+(1-a)*lo)
        return self.final_proj(torch.cat([ho,lo],1))

class FMMPreWork(nn.Module):
    def __init__(self, decoder_flag=False, inchannels=48):
        super().__init__()
        self.fd=FD(inchannels); self.freguide=FrequencyGuidedAttentionModule(inchannels,inchannels,decoder_flag)
        self.fh=FMgM(inchannels,True); self.fl=FMgM(inchannels,False)
    def forward(self, x):
        low,high=self.fd(x); return self.freguide(x,self.fh(high),self.fl(low))+x

class FSAS(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.to_hidden=nn.Conv2d(dim,dim*6,1,bias=bias); self.to_hidden_dw=nn.Conv2d(dim*6,dim*6,3,1,1,groups=dim*6,bias=bias)
        self.project_out=nn.Conv2d(dim*2,dim,1,bias=bias); self.norm=LayerNorm(dim*2); self.ps=2
    def forward(self, x):
        q,k,v=self.to_hidden_dw(self.to_hidden(x)).chunk(3,1)
        qp=rearrange(q,'b c (h p1) (w p2)->b c h w p1 p2',p1=self.ps,p2=self.ps)
        kp=rearrange(k,'b c (h p1) (w p2)->b c h w p1 p2',p1=self.ps,p2=self.ps)
        out=torch.fft.irfft2(torch.fft.rfft2(qp.float())*torch.fft.rfft2(kp.float()),s=(self.ps,self.ps))
        return self.project_out(v*self.norm(rearrange(out,'b c h w p1 p2->b c (h p1) (w p2)',p1=self.ps,p2=self.ps)))

class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        h=int(dim*ffn_expansion_factor); self.ps=2; self.project_in=nn.Conv2d(dim,h,1,bias=bias)
        self.dwconv=nn.Conv2d(h,h*2,3,1,1,groups=h,bias=bias)
        self.fft=nn.Parameter(torch.ones((h,1,1,self.ps,self.ps//2+1))); self.project_out=nn.Conv2d(h,dim,1,bias=bias)
    def forward(self, x):
        x=self.project_in(x); xp=rearrange(x,'b c (h p1) (w p2)->b c h w p1 p2',p1=self.ps,p2=self.ps)
        xp=torch.fft.irfft2(torch.fft.rfft2(xp.float())*self.fft,s=(self.ps,self.ps))
        x=rearrange(xp,'b c h w p1 p2->b c (h p1) (w p2)',p1=self.ps,p2=self.ps)
        x1,x2=self.dwconv(x).chunk(2,1); return self.project_out(F.gelu(x1)*x2)

class FFTransformerBlock(nn.Module):
    def __init__(self, dim, decoder_flag=True, ffn_expansion_factor=2.66, bias=False, att=False):
        super().__init__()
        self.att=att
        if att: self.norm1=LayerNorm(dim); self.attn=FSAS(dim,bias)
        self.norm2=LayerNorm(dim); self.ffn=DFFN(dim,ffn_expansion_factor,bias)
        self.prompt_conv=nn.Conv2d(dim,dim,1); self.prompt_block=FMMPreWork(decoder_flag=decoder_flag,inchannels=dim)
    def forward(self, x):
        # FIX Bug 6: sigmoid gate multiplies, not adds
        p1=self.prompt_block(x); x=torch.sigmoid(self.prompt_conv(p1))*p1
        if self.att: x=x+self.attn(self.norm1(x))
        return x+self.ffn(self.norm2(x))

class Downsample(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(n,n//2,3,1,1,bias=False),nn.PixelUnshuffle(2))
    def forward(self, x): return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(n,n*2,3,1,1,bias=False),nn.PixelShuffle(2))
    def forward(self, x): return self.body(x)

class AdaIR(nn.Module):
    """FIX: Uses consistent attr names matching train.py. Bug 2 fix applied."""
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=[4,6,6,8], num_fft_blocks=[4,2,2,1],
                 num_refinement_blocks=4, heads=[1,2,4,8],
                 ffn_expansion_factor=2.66, bias=False, LN='WithBias'):
        super().__init__()
        self.patch_embed=nn.Conv2d(inp_channels,dim,3,1,1,bias=bias)
        self.enc1=nn.Sequential(*[TransformerBlock(dim,heads[0],ffn_expansion_factor,bias,LN) for _ in range(num_blocks[0])])
        self.enc1f=nn.Sequential(*[FFTransformerBlock(dim) for _ in range(num_fft_blocks[0])])
        self.down1=Downsample(dim)
        self.enc2=nn.Sequential(*[TransformerBlock(dim*2,heads[1],ffn_expansion_factor,bias,LN) for _ in range(num_blocks[1])])
        self.enc2f=nn.Sequential(*[FFTransformerBlock(dim*2) for _ in range(num_fft_blocks[1])])
        self.down2=Downsample(dim*2)
        self.enc3=nn.Sequential(*[TransformerBlock(dim*4,heads[2],ffn_expansion_factor,bias,LN) for _ in range(num_blocks[2])])
        self.enc3f=nn.Sequential(*[FFTransformerBlock(dim*4) for _ in range(num_fft_blocks[2])])
        self.down3=Downsample(dim*4)
        self.latent=nn.Sequential(*[TransformerBlock(dim*8,heads[3],ffn_expansion_factor,bias,LN) for _ in range(num_blocks[3])])
        # FIX Bug 2: bottleneck uses num_fft_blocks[3] (was [1])
        self.latentf=nn.Sequential(*[FFTransformerBlock(dim*8) for _ in range(num_fft_blocks[3])])
        self.up3=Upsample(dim*8); self.red3=nn.Conv2d(dim*8,dim*4,1,bias=bias)
        self.dec3=nn.Sequential(*[TransformerBlock(dim*4,heads[2],ffn_expansion_factor,bias,LN) for _ in range(num_blocks[2])])
        self.dec3f=nn.Sequential(*[FFTransformerBlock(dim*4,decoder_flag=True) for _ in range(num_fft_blocks[2])])
        self.up2=Upsample(dim*4); self.red2=nn.Conv2d(dim*4,dim*2,1,bias=bias)
        self.dec2=nn.Sequential(*[TransformerBlock(dim*2,heads[1],ffn_expansion_factor,bias,LN) for _ in range(num_blocks[1])])
        self.dec2f=nn.Sequential(*[FFTransformerBlock(dim*2,decoder_flag=True) for _ in range(num_fft_blocks[1])])
        self.up1=Upsample(dim*2); self.red1=nn.Conv2d(dim*2,dim,1,bias=bias)
        self.dec1=nn.Sequential(*[TransformerBlock(dim,heads[0],ffn_expansion_factor,bias,LN) for _ in range(num_blocks[0])])
        self.dec1f=nn.Sequential(*[FFTransformerBlock(dim,decoder_flag=True) for _ in range(num_fft_blocks[0])])
        self.refine=nn.Sequential(*[TransformerBlock(dim,heads[0],ffn_expansion_factor,bias,LN) for _ in range(num_refinement_blocks)])
        self.output=nn.Conv2d(dim,out_channels,3,1,1,bias=bias)
    def forward(self, img):
        e1=self.patch_embed(img); e1=self.enc1(self.enc1f(e1))+e1
        e2=self.down1(e1); e2=self.enc2(self.enc2f(e2))+e2
        e3=self.down2(e2); e3=self.enc3(self.enc3f(e3))+e3
        e4=self.down3(e3); lat=self.latent(self.latentf(e4))+e4
        d3=self.red3(torch.cat([self.up3(lat),e3],1)); d3=self.dec3(self.dec3f(d3))+d3
        d2=self.red2(torch.cat([self.up2(d3),e2],1)); d2=self.dec2(self.dec2f(d2))+d2
        d1=self.red1(torch.cat([self.up1(d2),e1],1)); d1=self.dec1(self.dec1f(d1))+d1
        return self.output(self.refine(d1))+img

print('All model modules loaded')


# ===== FRAME CACHE =====
class FrameCache:
    def __init__(self, cache_size=32, threshold=0.95, thumb_size=64):
        self.cache_size=cache_size; self.threshold=threshold; self.thumb_size=thumb_size
        self.cache=OrderedDict(); self._k=0; self.hits=0; self.misses=0
    def _thumb(self, f):
        if f.dim()==4: f=f[0]
        t=F.interpolate(f.unsqueeze(0),(self.thumb_size,self.thumb_size),mode='bilinear',align_corners=False)
        return F.normalize(t.reshape(-1),dim=0)
    def query(self, f):
        th=self._thumb(f); bs,bk=-1.0,None
        for k,(ct,_) in self.cache.items():
            s=torch.dot(th,ct.to(th.device)).item()
            if s>bs: bs=s; bk=k
        if bs>=self.threshold and bk is not None:
            self.hits+=1; self.cache.move_to_end(bk); return True,self.cache[bk][1],bs
        self.misses+=1; return False,None,bs
    def store(self, f, r):
        self.cache[self._k]=(self._thumb(f).cpu(),r.cpu()); self._k+=1
        while len(self.cache)>self.cache_size: self.cache.popitem(last=False)


# ===== METRICS =====
def compute_psnr(pred, target, data_range=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0: return float('inf')
    return (10.0 * torch.log10(data_range**2 / mse)).item()

def compute_ssim(pred, target, window_size=11):
    C1 = 0.01**2; C2 = 0.03**2
    mu_p = F.avg_pool2d(pred, window_size, 1, window_size//2)
    mu_t = F.avg_pool2d(target, window_size, 1, window_size//2)
    mu_pp = mu_p * mu_p; mu_tt = mu_t * mu_t; mu_pt = mu_p * mu_t
    sigma_pp = F.avg_pool2d(pred*pred, window_size, 1, window_size//2) - mu_pp
    sigma_tt = F.avg_pool2d(target*target, window_size, 1, window_size//2) - mu_tt
    sigma_pt = F.avg_pool2d(pred*target, window_size, 1, window_size//2) - mu_pt
    ssim = ((2*mu_pt+C1)*(2*sigma_pt+C2)) / ((mu_pp+mu_tt+C1)*(sigma_pp+sigma_tt+C2))
    return ssim.mean().item()


# ===== TILE INFERENCE =====
def tile_inference(model, inp, tile_size=256, overlap=32):
    """Tile-based inference for large resolution frames.
    FIX: count.clamp(min=1e-5) to prevent division by zero."""
    B, C, H, W = inp.shape
    stride = tile_size - overlap
    out = torch.zeros_like(inp)
    count = torch.zeros_like(inp)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + tile_size, H); x_end = min(x + tile_size, W)
            y_start = max(y_end - tile_size, 0); x_start = max(x_end - tile_size, 0)
            tile = inp[:, :, y_start:y_end, x_start:x_end]
            tile_out = model(tile)
            out[:, :, y_start:y_end, x_start:x_end] += tile_out
            count[:, :, y_start:y_end, x_start:x_end] += 1
    # FIX: clamp count to prevent division by zero
    return out / count.clamp(min=1e-5)


# ===== INFERENCE PIPELINE =====
@torch.no_grad()
def inference_video(model, frames, frame_cache=None, device='cuda',
                    use_amp=True, tile_size=None):
    model.eval()
    results = []
    timings = []
    if isinstance(frames, torch.Tensor) and frames.dim() == 4:
        frames = [frames[i] for i in range(frames.shape[0])]

    for idx, frame in enumerate(frames):
        start = time.time()
        inp = frame.unsqueeze(0).to(device)

        if frame_cache is not None:
            hit, cached, sim = frame_cache.query(inp)
            if hit:
                results.append(cached.squeeze(0))
                timings.append(time.time() - start)
                print(f'  Frame {idx+1}/{len(frames)}: CACHE HIT (sim={sim:.4f}, {timings[-1]*1000:.1f}ms)')
                continue

        with torch.cuda.amp.autocast(enabled=use_amp):
            if tile_size and (frame.shape[-2] > tile_size or frame.shape[-1] > tile_size):
                restored = tile_inference(model, inp, tile_size)
            else:
                restored = model(inp)

        restored = restored.clamp(0, 1)

        if frame_cache is not None:
            frame_cache.store(inp, restored.detach())

        results.append(restored.squeeze(0).cpu())
        timings.append(time.time() - start)
        print(f'  Frame {idx+1}/{len(frames)}: processed ({timings[-1]*1000:.1f}ms)')

    avg_time = np.mean(timings) * 1000
    fps = 1000.0 / avg_time if avg_time > 0 else 0
    print(f'\nAvg: {avg_time:.1f}ms/frame, {fps:.1f} FPS')
    if frame_cache:
        total = frame_cache.hits + frame_cache.misses
        if total > 0:
            print(f'Cache: {frame_cache.hits} hits / {frame_cache.misses} misses ({frame_cache.hits/total*100:.0f}% hit rate)')

    return results, {'avg_time_ms': avg_time, 'fps': fps, 'timings': timings}


# ===== MAIN =====
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # FIX: Use consistent kwarg names matching AdaIR constructor
    model = AdaIR(
        dim=48, num_blocks=[1,1,1,1], num_fft_blocks=[1,1,1,1],
        num_refinement_blocks=1
    ).to(device)

    ckpt_path = './checkpoints/final_model.pth'
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f'Loaded checkpoint: {ckpt_path}')
    else:
        print('No checkpoint found, using random weights (demo mode)')

    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Create synthetic test frames
    print('\nGenerating synthetic video frames...')
    num_frames = 10
    base_frame = torch.rand(3, 128, 128)
    frames = []
    for i in range(num_frames):
        if i % 5 == 0:
            base_frame = torch.rand(3, 128, 128)
        noise = torch.randn_like(base_frame) * 0.05
        frames.append((base_frame + noise).clamp(0, 1))

    print(f'Created {num_frames} frames of shape {frames[0].shape}')

    # Run inference with frame cache
    print('\n--- Inference with Frame Cache ---')
    cache = FrameCache(cache_size=16, threshold=0.90)
    results, metrics = inference_video(model, frames, frame_cache=cache, device=device)

    # Print metrics
    for i in range(min(3, len(results))):
        psnr = compute_psnr(results[i].unsqueeze(0), frames[i].unsqueeze(0))
        print(f'Frame {i+1}: PSNR={psnr:.2f}dB')

    print('\n[OK] Inference complete!')
