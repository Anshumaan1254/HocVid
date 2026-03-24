"""
Training Script — EvoIR Stage 1

Converted from train.ipynb with fixes:
- Bug 2:  Bottleneck uses num_fft_blocks[3]
- Bug 5:  FrequencyGuidedAttentionModule num_heads=4
- Bug 6:  Sigmoid gate multiplies in FFTransformerBlock
- Clamp: Output clamped to [0,1] before MS-SSIM loss
- MS-SSIM: kernel_size=11, betas=(0.0448, 0.2856, 0.3001)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import time
import random
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import copy
import numbers
from collections import OrderedDict
from einops import rearrange

try:
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
    HAS_TORCHMETRICS = True
except ImportError:
    HAS_TORCHMETRICS = False

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')


# ===================== UTILITY MODULES =====================
def to_3d(x): return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w): return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, ns):
        super().__init__()
        if isinstance(ns, numbers.Integral): ns = (ns,)
        ns = torch.Size(ns); self.weight = nn.Parameter(torch.ones(ns)); self.normalized_shape = ns
    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, ns):
        super().__init__()
        if isinstance(ns, numbers.Integral): ns = (ns,)
        ns = torch.Size(ns); self.weight = nn.Parameter(torch.ones(ns)); self.bias = nn.Parameter(torch.zeros(ns)); self.normalized_shape = ns
    def forward(self, x):
        mu = x.mean(-1, keepdim=True); sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, t='WithBias'):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if t == 'BiasFree' else WithBias_LayerNorm(dim)
    def forward(self, x):
        h, w = x.shape[-2:]; return to_4d(self.body(to_3d(x)), h, w)


# ===================== ATTENTION & FFN =====================
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim*3, 1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, 3, 1, 1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)
    def forward(self, x):
        b,c,h,w = x.shape; qkv = self.qkv_dwconv(self.qkv(x)); q,k,v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2,-1)) * self.temperature
        out = attn.softmax(dim=-1) @ v
        return self.project_out(rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w))

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden*2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden*2, hidden*2, 3, 1, 1, groups=hidden*2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)
    def forward(self, x):
        x = self.project_in(x); x1,x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1)*x2)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LN_type='WithBias'):
        super().__init__()
        self.norm1 = LayerNorm(dim, LN_type); self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LN_type); self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
    def forward(self, x):
        x = x + self.attn(self.norm1(x)); x = x + self.ffn(self.norm2(x)); return x


# ===================== FREQUENCY MODULES =====================
class FD(nn.Module):
    def __init__(self, inchannels, kernel_size=3, group=8):
        super().__init__()
        self.kernel_size = kernel_size; self.group = group
        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, 1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size**2); self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.pad = nn.ReflectionPad2d(kernel_size//2); self.ap = nn.AdaptiveAvgPool2d((1,1))
    def forward(self, x):
        identity = x; lf = self.bn(self.conv(self.ap(x)))
        n,c,h,w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n,self.group,c//self.group,self.kernel_size**2,h*w)
        n,c1,p,q = lf.shape
        lf = self.act(lf.reshape(n,c1//self.kernel_size**2,self.kernel_size**2,p*q).unsqueeze(2))
        low = torch.sum(x*lf, dim=3).reshape(n,c,h,w); return low, identity-low

class FMgM(nn.Module):
    def __init__(self, in_channels, flag_highF):
        super().__init__()
        self.flag_highF = flag_highF; dim = in_channels; k=3
        if flag_highF:
            self.body = nn.Sequential(nn.Conv2d(dim,dim,(1,k),padding=(0,k//2),groups=dim),
                                       nn.Conv2d(dim,dim,(k,1),padding=(k//2,0),groups=dim), nn.GELU())
        else:
            self.body = nn.Sequential(nn.Conv2d(2*dim,2*dim,1), nn.GELU())
    def forward(self, ffm):
        if self.flag_highF: return self.body(ffm)*ffm
        bs,c,H,W = ffm.shape
        y = torch.fft.rfft2(ffm.float()); y_f = torch.cat([y.real,y.imag],dim=1)
        y_f = y_f * self.body(y_f); yr,yi = y_f.chunk(2,dim=1)
        return torch.fft.irfft2(torch.complex(yr,yi), s=(H,W))

class FrequencyGuidedAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super().__init__()
        self.num_heads = num_heads; self.temperature = nn.Parameter(torch.ones(num_heads,1,1))
        self.q_proj = nn.Conv2d(dim,dim,1,bias=bias); self.kv_proj = nn.Conv2d(dim,dim*2,1,bias=bias)
        self.q_dwconv = nn.Conv2d(dim,dim,3,padding=1,groups=dim,bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2,dim*2,3,padding=1,groups=dim*2,bias=bias)
        self.project_out = nn.Conv2d(dim,dim,1,bias=bias)
    def forward(self, x, freq_feat):
        B,C,H,W = x.shape
        q = self.q_dwconv(self.q_proj(freq_feat)); kv = self.kv_dwconv(self.kv_proj(x))
        k,v = kv.chunk(2,dim=1)
        q = rearrange(q,'b (h c) H W -> b h c (H W)',h=self.num_heads)
        k = rearrange(k,'b (h c) H W -> b h c (H W)',h=self.num_heads)
        v = rearrange(v,'b (h c) H W -> b h c (H W)',h=self.num_heads)
        q = F.normalize(q,dim=-1); k = F.normalize(k,dim=-1)
        out = (q@k.transpose(-2,-1)*self.temperature).softmax(-1) @ v
        return self.project_out(rearrange(out,'b h c (H W) -> b (h c) H W',h=self.num_heads,H=H,W=W))

class FrequencyGuidedAttentionModule(nn.Module):
    def __init__(self, in_c, out_c, decoder_flag=False):
        super().__init__()
        self.decoder_flag = decoder_flag
        # FIX Bug 5: num_heads=4, NOT out_c
        self.hfa = FrequencyGuidedAttention(in_c, num_heads=4)
        self.lfa = FrequencyGuidedAttention(in_c, num_heads=4)
        if decoder_flag: self.alpha=nn.Parameter(torch.tensor(0.5)); self.proj=nn.Conv2d(in_c,out_c,1)
        else: self.final_proj = nn.Conv2d(in_c*2,out_c,1)
    def forward(self, x, hf, lf):
        ho=self.hfa(x,hf); lo=self.lfa(x,lf)
        if self.decoder_flag:
            a=torch.sigmoid(self.alpha); return self.proj(a*ho+(1-a)*lo)
        return self.final_proj(torch.cat([ho,lo],1))

class FMMPreWork(nn.Module):
    def __init__(self, decoder_flag=False, inchannels=48):
        super().__init__()
        self.fd = FD(inchannels)
        self.freguide = FrequencyGuidedAttentionModule(inchannels,inchannels,decoder_flag)
        self.fh = FMgM(inchannels, True); self.fl = FMgM(inchannels, False)
    def forward(self, x):
        low,high = self.fd(x)
        return self.freguide(x, self.fh(high), self.fl(low)) + x


# ===================== FFT TRANSFORMER =====================
class FSAS(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.to_hidden = nn.Conv2d(dim,dim*6,1,bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim*6,dim*6,3,1,1,groups=dim*6,bias=bias)
        self.project_out = nn.Conv2d(dim*2,dim,1,bias=bias)
        self.norm = LayerNorm(dim*2); self.ps = 2
    def forward(self, x):
        q,k,v = self.to_hidden_dw(self.to_hidden(x)).chunk(3,dim=1)
        qp = rearrange(q,'b c (h p1) (w p2) -> b c h w p1 p2',p1=self.ps,p2=self.ps)
        kp = rearrange(k,'b c (h p1) (w p2) -> b c h w p1 p2',p1=self.ps,p2=self.ps)
        out = torch.fft.irfft2(torch.fft.rfft2(qp.float())*torch.fft.rfft2(kp.float()),s=(self.ps,self.ps))
        out = rearrange(out,'b c h w p1 p2 -> b c (h p1) (w p2)',p1=self.ps,p2=self.ps)
        return self.project_out(v * self.norm(out))

class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim*ffn_expansion_factor); self.ps = 2
        self.project_in = nn.Conv2d(dim,hidden,1,bias=bias)
        self.dwconv = nn.Conv2d(hidden,hidden*2,3,1,1,groups=hidden,bias=bias)
        self.fft = nn.Parameter(torch.ones((hidden,1,1,self.ps,self.ps//2+1)))
        self.project_out = nn.Conv2d(hidden,dim,1,bias=bias)
    def forward(self, x):
        x = self.project_in(x)
        xp = rearrange(x,'b c (h p1) (w p2) -> b c h w p1 p2',p1=self.ps,p2=self.ps)
        xp = torch.fft.irfft2(torch.fft.rfft2(xp.float())*self.fft, s=(self.ps,self.ps))
        x = rearrange(xp,'b c h w p1 p2 -> b c (h p1) (w p2)',p1=self.ps,p2=self.ps)
        x1,x2 = self.dwconv(x).chunk(2,dim=1); return self.project_out(F.gelu(x1)*x2)

class FFTransformerBlock(nn.Module):
    def __init__(self, dim, decoder_flag=True, ffn_expansion_factor=2.66, bias=False, att=False):
        super().__init__()
        self.att = att
        if att: self.norm1=LayerNorm(dim); self.attn=FSAS(dim,bias)
        self.norm2=LayerNorm(dim); self.ffn=DFFN(dim,ffn_expansion_factor,bias)
        self.prompt_conv=nn.Conv2d(dim,dim,1)
        self.prompt_block=FMMPreWork(decoder_flag=decoder_flag,inchannels=dim)
    def forward(self, x):
        p1=self.prompt_block(x)
        # FIX Bug 6: sigmoid gate multiplies, not adds
        x=torch.sigmoid(self.prompt_conv(p1))*p1
        if self.att: x=x+self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


# ===================== ENCODER-DECODER =====================
class Downsample(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n,n//2,3,1,1,bias=False), nn.PixelUnshuffle(2))
    def forward(self, x): return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(n,n*2,3,1,1,bias=False), nn.PixelShuffle(2))
    def forward(self, x): return self.body(x)

class AdaIR(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=[4,6,6,8], num_fft_blocks=[4,2,2,1],
                 num_refinement_blocks=4, heads=[1,2,4,8],
                 ffn_expansion_factor=2.66, bias=False, LN='WithBias'):
        super().__init__()
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, 1, 1, bias=bias)
        # Encoder
        self.enc1 = nn.Sequential(*[TransformerBlock(dim,heads[0],ffn_expansion_factor,bias,LN) for _ in range(num_blocks[0])])
        self.enc1f = nn.Sequential(*[FFTransformerBlock(dim) for _ in range(num_fft_blocks[0])])
        self.down1 = Downsample(dim)
        self.enc2 = nn.Sequential(*[TransformerBlock(dim*2,heads[1],ffn_expansion_factor,bias,LN) for _ in range(num_blocks[1])])
        self.enc2f = nn.Sequential(*[FFTransformerBlock(dim*2) for _ in range(num_fft_blocks[1])])
        self.down2 = Downsample(dim*2)
        self.enc3 = nn.Sequential(*[TransformerBlock(dim*4,heads[2],ffn_expansion_factor,bias,LN) for _ in range(num_blocks[2])])
        self.enc3f = nn.Sequential(*[FFTransformerBlock(dim*4) for _ in range(num_fft_blocks[2])])
        self.down3 = Downsample(dim*4)
        # Bottleneck
        self.latent = nn.Sequential(*[TransformerBlock(dim*8,heads[3],ffn_expansion_factor,bias,LN) for _ in range(num_blocks[3])])
        # FIX Bug 2: use num_fft_blocks[3] (was [1])
        self.latentf = nn.Sequential(*[FFTransformerBlock(dim*8) for _ in range(num_fft_blocks[3])])
        # Decoder
        self.up3 = Upsample(dim*8); self.red3 = nn.Conv2d(dim*8,dim*4,1,bias=bias)
        self.dec3 = nn.Sequential(*[TransformerBlock(dim*4,heads[2],ffn_expansion_factor,bias,LN) for _ in range(num_blocks[2])])
        self.dec3f = nn.Sequential(*[FFTransformerBlock(dim*4,decoder_flag=True) for _ in range(num_fft_blocks[2])])
        self.up2 = Upsample(dim*4); self.red2 = nn.Conv2d(dim*4,dim*2,1,bias=bias)
        self.dec2 = nn.Sequential(*[TransformerBlock(dim*2,heads[1],ffn_expansion_factor,bias,LN) for _ in range(num_blocks[1])])
        self.dec2f = nn.Sequential(*[FFTransformerBlock(dim*2,decoder_flag=True) for _ in range(num_fft_blocks[1])])
        self.up1 = Upsample(dim*2); self.red1 = nn.Conv2d(dim*2,dim,1,bias=bias)
        self.dec1 = nn.Sequential(*[TransformerBlock(dim,heads[0],ffn_expansion_factor,bias,LN) for _ in range(num_blocks[0])])
        self.dec1f = nn.Sequential(*[FFTransformerBlock(dim,decoder_flag=True) for _ in range(num_fft_blocks[0])])
        # Refinement
        self.refine = nn.Sequential(*[TransformerBlock(dim,heads[0],ffn_expansion_factor,bias,LN) for _ in range(num_refinement_blocks)])
        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias)

    def forward(self, img):
        e1 = self.patch_embed(img)
        e1 = self.enc1(self.enc1f(e1)) + e1
        e2 = self.down1(e1); e2 = self.enc2(self.enc2f(e2)) + e2
        e3 = self.down2(e2); e3 = self.enc3(self.enc3f(e3)) + e3
        e4 = self.down3(e3); lat = self.latent(self.latentf(e4)) + e4
        d3 = self.red3(torch.cat([self.up3(lat), e3], 1))
        d3 = self.dec3(self.dec3f(d3)) + d3
        d2 = self.red2(torch.cat([self.up2(d3), e2], 1))
        d2 = self.dec2(self.dec2f(d2)) + d2
        d1 = self.red1(torch.cat([self.up1(d2), e1], 1))
        d1 = self.dec1(self.dec1f(d1)) + d1
        return self.output(self.refine(d1)) + img


# ===================== EOS MODULE =====================
class MS_SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0):
        super().__init__()
        if HAS_TORCHMETRICS:
            # FIX: kernel_size=11, standard betas
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=data_range, kernel_size=11,
                betas=(0.0448, 0.2856, 0.3001))
        else:
            self.ms_ssim = None
    def forward(self, pred, target):
        if self.ms_ssim is not None:
            self.ms_ssim = self.ms_ssim.to(pred.device)
            return 1 - self.ms_ssim(pred, target)
        return F.mse_loss(pred, target)

def init_population(size=5):
    pop = []
    for _ in range(size):
        l1 = random.uniform(0.1, 0.9); pop.append([l1, 1.0-l1])
    return pop

def evaluate_individual(ind, restored_list, clean_list, l1_loss, ms_ssim_loss):
    total = 0.0
    with torch.no_grad():
        for r, c in zip(restored_list, clean_list):
            if r.dim()==3: r=r.unsqueeze(0); c=c.unsqueeze(0)
            total += (ind[0]*l1_loss(r,c) + ind[1]*ms_ssim_loss(r,c)).item()
    return -total

def crossover(p1, p2):
    a = random.random()
    return [a*p1[0]+(1-a)*p2[0], a*p1[1]+(1-a)*p2[1]], [a*p2[0]+(1-a)*p1[0], a*p2[1]+(1-a)*p1[1]]

def mutate(ind, rate=0.1):
    if random.random() < rate:
        d = random.uniform(-0.1, 0.1)
        ind[0] = min(max(ind[0]+d, 0.1), 0.9); ind[1] = 1.0-ind[0]
    return ind

def evolve_weights(restored_list, clean_list, l1_loss, ms_ssim_loss, gens=3, pop_size=5):
    pop = init_population(pop_size); best=None; best_s=-float('inf')
    for _ in range(gens):
        scores = sorted([(evaluate_individual(i,restored_list,clean_list,l1_loss,ms_ssim_loss),i) for i in pop], reverse=True)
        if scores[0][0]>best_s: best_s=scores[0][0]; best=scores[0][1][:]
        nxt = [scores[0][1][:]]
        while len(nxt)<pop_size:
            p1,p2 = random.sample(scores[:3],2)
            c1,c2 = crossover(p1[1],p2[1]); nxt+=[mutate(c1[:]),mutate(c2[:])]
        pop = nxt[:pop_size]
    return best

class EMATeacher:
    def __init__(self, model, momentum=0.999):
        self.momentum = momentum; self.teacher = copy.deepcopy(model)
        for p in self.teacher.parameters(): p.requires_grad = False
        self.teacher.eval()
    @torch.no_grad()
    def update(self, model):
        for tp, sp in zip(self.teacher.parameters(), model.parameters()):
            tp.data.mul_(self.momentum).add_(sp.data, alpha=1-self.momentum)
    @torch.no_grad()
    def evaluate(self, x):
        self.teacher.eval(); return self.teacher(x)


# ===================== FRAME CACHE =====================
class FrameCache:
    def __init__(self, cache_size=32, sim_threshold=0.95, thumb_size=64):
        self.cache_size = cache_size; self.threshold = sim_threshold; self.thumb_size = thumb_size
        self.cache = OrderedDict(); self._key = 0
        self.hits = 0; self.misses = 0
    def _thumb(self, frame):
        if frame.dim()==4: frame=frame[0]
        t = F.interpolate(frame.unsqueeze(0),(self.thumb_size,self.thumb_size),mode='bilinear',align_corners=False)
        return F.normalize(t.reshape(-1), dim=0)
    def query(self, frame):
        thumb = self._thumb(frame)
        best_sim, best_key = -1.0, None
        for k,(ct,_) in self.cache.items():
            sim = torch.dot(thumb, ct.to(thumb.device)).item()
            if sim > best_sim: best_sim=sim; best_key=k
        if best_sim >= self.threshold and best_key is not None:
            self.hits += 1; self.cache.move_to_end(best_key)
            return True, self.cache[best_key][1], best_sim
        self.misses += 1; return False, None, best_sim
    def store(self, frame, result):
        self.cache[self._key] = (self._thumb(frame).cpu(), result.cpu()); self._key += 1
        while len(self.cache) > self.cache_size: self.cache.popitem(last=False)


# ===================== DATASET =====================
class VideoSRDataset(Dataset):
    def __init__(self, degraded_dir=None, clean_dir=None, patch_size=128, num_samples=1000):
        self.patch_size = patch_size
        self.use_synthetic = degraded_dir is None or not os.path.exists(str(degraded_dir))
        self.num_samples = num_samples
        if not self.use_synthetic:
            self.degraded_dir = Path(degraded_dir)
            self.clean_dir = Path(clean_dir)
            exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
            self.files = sorted([f for f in self.degraded_dir.iterdir() if f.suffix.lower() in exts])
            print(f'Found {len(self.files)} frame pairs')
        else:
            print(f'Using synthetic dataset ({num_samples} samples, patch={patch_size})')

    def __len__(self):
        return len(self.files) if not self.use_synthetic else self.num_samples

    def __getitem__(self, idx):
        if self.use_synthetic:
            clean = torch.rand(3, self.patch_size, self.patch_size)
            noise = torch.randn_like(clean) * 0.1
            degraded = (clean + noise).clamp(0, 1)
            return degraded, clean
        else:
            from PIL import Image
            import torchvision.transforms.functional as TF
            deg_path = self.files[idx]
            clean_path = self.clean_dir / deg_path.name
            deg_img = TF.to_tensor(Image.open(deg_path).convert('RGB'))
            clean_img = TF.to_tensor(Image.open(clean_path).convert('RGB'))
            _, h, w = deg_img.shape
            if h > self.patch_size and w > self.patch_size:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                deg_img = deg_img[:, top:top+self.patch_size, left:left+self.patch_size]
                clean_img = clean_img[:, top:top+self.patch_size, left:left+self.patch_size]
            if random.random() > 0.5:
                deg_img = deg_img.flip(-1); clean_img = clean_img.flip(-1)
            return deg_img, clean_img


# ===================== CONFIGURATION =====================
class TrainConfig:
    dim = 48
    num_blocks = [4, 6, 6, 8]
    num_fft_blocks = [4, 2, 2, 1]
    num_refinement_blocks = 4
    heads = [1, 2, 4, 8]
    batch_size = 2
    patch_size = 128
    epochs = 200
    lr = 2e-4
    weight_decay = 1e-4
    eos_interval = 500
    eos_pop_size = 5
    eos_generations = 3
    ema_momentum = 0.999
    initial_lambda = [0.8, 0.2]
    use_amp = True
    compile_model = False
    degraded_dir = None
    clean_dir = None
    num_workers = 4
    num_synthetic_samples = 500
    save_dir = './checkpoints'
    save_every = 10
    log_every = 50
    lr_milestones = [75, 150]
    lr_gamma = 0.5


# ===================== TRAINING LOOP =====================
def train():
    cfg = TrainConfig()

    # For quick demo
    cfg.epochs = 3
    cfg.num_synthetic_samples = 100
    cfg.eos_interval = 20
    cfg.log_every = 10
    cfg.save_every = 1
    cfg.num_workers = 0
    cfg.num_blocks = [1, 1, 1, 1]
    cfg.num_fft_blocks = [1, 1, 1, 1]
    cfg.num_refinement_blocks = 1
    cfg.patch_size = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = AdaIR(
        dim=cfg.dim, num_blocks=cfg.num_blocks, num_fft_blocks=cfg.num_fft_blocks,
        num_refinement_blocks=cfg.num_refinement_blocks, heads=cfg.heads
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,} ({total_params/1e6:.2f}M)')

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_milestones, gamma=cfg.lr_gamma)
    scaler = GradScaler(enabled=cfg.use_amp)

    l1_loss = nn.L1Loss()
    ms_ssim_loss = MS_SSIMLoss(data_range=1.0)

    ema_teacher = EMATeacher(model, momentum=cfg.ema_momentum)
    lambda_l1, lambda_ssim = cfg.initial_lambda

    frame_cache = FrameCache(cache_size=32, sim_threshold=0.95)

    dataset = VideoSRDataset(
        degraded_dir=cfg.degraded_dir, clean_dir=cfg.clean_dir,
        patch_size=cfg.patch_size, num_samples=cfg.num_synthetic_samples)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    os.makedirs(cfg.save_dir, exist_ok=True)

    history = {'loss': [], 'lambda_l1': [], 'lambda_ssim': [], 'cache_hits': [], 'cache_misses': []}
    global_step = 0

    print(f'\n{"="*60}')
    print(f'Starting training: {cfg.epochs} epochs, {len(dataloader)} batches/epoch')
    print(f'Initial weights: lam_L1={lambda_l1:.3f}, lam_SSIM={lambda_ssim:.3f}')
    print(f'{"="*60}\n')

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        restored_cache_for_eos = []
        clean_cache_for_eos = []

        for batch_idx, (degraded, clean) in enumerate(dataloader):
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.use_amp):
                restored = model(degraded)
                # FIX: Clamp output before MS-SSIM loss
                restored_clamped = restored.clamp(0, 1)
                loss = lambda_l1 * l1_loss(restored_clamped, clean) + lambda_ssim * ms_ssim_loss(restored_clamped, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            global_step += 1

            ema_teacher.update(model)

            restored_cache_for_eos.append(restored_clamped.detach().cpu())
            clean_cache_for_eos.append(clean.detach().cpu())
            if len(restored_cache_for_eos) > 32:
                restored_cache_for_eos = restored_cache_for_eos[-32:]
                clean_cache_for_eos = clean_cache_for_eos[-32:]

            frame_cache.store(degraded, restored.detach())

            if global_step % cfg.eos_interval == 0 and len(restored_cache_for_eos) >= 2:
                print(f'  [EOS] Evolving loss weights at step {global_step}...')
                new_weights = evolve_weights(
                    restored_cache_for_eos, clean_cache_for_eos,
                    l1_loss, ms_ssim_loss,
                    gens=cfg.eos_generations, pop_size=cfg.eos_pop_size)
                lambda_l1, lambda_ssim = new_weights
                print(f'  [EOS] New weights: lam_L1={lambda_l1:.4f}, lam_SSIM={lambda_ssim:.4f}')
                restored_cache_for_eos.clear()
                clean_cache_for_eos.clear()

            if global_step % cfg.log_every == 0:
                print(f'  Epoch {epoch+1}/{cfg.epochs} | Step {global_step} | '
                      f'Loss: {loss.item():.4f} | '
                      f'lam_L1={lambda_l1:.3f} lam_SSIM={lambda_ssim:.3f} | '
                      f'Cache: {frame_cache.hits}H/{frame_cache.misses}M')

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - epoch_start

        history['loss'].append(avg_loss)
        history['lambda_l1'].append(lambda_l1)
        history['lambda_ssim'].append(lambda_ssim)
        history['cache_hits'].append(frame_cache.hits)
        history['cache_misses'].append(frame_cache.misses)

        print(f'Epoch {epoch+1}/{cfg.epochs} | Avg Loss: {avg_loss:.4f} | '
              f'LR: {scheduler.get_last_lr()[0]:.6f} | Time: {elapsed:.1f}s')

        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.save_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_teacher_state_dict': ema_teacher.teacher.state_dict(),
                'lambda_l1': lambda_l1, 'lambda_ssim': lambda_ssim, 'loss': avg_loss,
            }, ckpt_path)
            print(f'  Checkpoint saved: {ckpt_path}')

    final_path = os.path.join(cfg.save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f'\nTraining complete! Final model: {final_path}')
    return model, history


if __name__ == '__main__':
    model, history = train()
