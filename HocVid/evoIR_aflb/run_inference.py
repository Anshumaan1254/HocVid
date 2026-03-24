"""
Quick inference script: Run EvoIR AFLB model on a single image.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from collections import OrderedDict
from PIL import Image
import torchvision.transforms.functional as TF
import os
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ===================== MODEL DEFINITION (compact) =====================
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
    def __init__(self, dim, nh=4, bias=False):
        super().__init__()
        self.nh=nh; self.temperature=nn.Parameter(torch.ones(nh,1,1))
        self.q_proj=nn.Conv2d(dim,dim,1,bias=bias); self.kv_proj=nn.Conv2d(dim,dim*2,1,bias=bias)
        self.q_dw=nn.Conv2d(dim,dim,3,padding=1,groups=dim,bias=bias)
        self.kv_dw=nn.Conv2d(dim*2,dim*2,3,padding=1,groups=dim*2,bias=bias)
        self.out=nn.Conv2d(dim,dim,1,bias=bias)
    def forward(self, x, ff):
        B,C,H,W=x.shape; q=self.q_dw(self.q_proj(ff)); kv=self.kv_dw(self.kv_proj(x)); k,v=kv.chunk(2,1)
        q=rearrange(q,'b (h c) H W->b h c (H W)',h=self.nh); k=rearrange(k,'b (h c) H W->b h c (H W)',h=self.nh)
        v=rearrange(v,'b (h c) H W->b h c (H W)',h=self.nh)
        q=F.normalize(q,-1); k=F.normalize(k,-1)
        return self.out(rearrange((q@k.transpose(-2,-1)*self.temperature).softmax(-1)@v,'b h c (H W)->b (h c) H W',h=self.nh,H=H,W=W))

class FrequencyGuidedAttentionModule(nn.Module):
    def __init__(self, ic, oc, df=False):
        super().__init__()
        # FIX Bug 5: use nh=4, NOT oc as num_heads
        self.df=df; self.hfa=FrequencyGuidedAttention(ic,nh=4); self.lfa=FrequencyGuidedAttention(ic,nh=4)
        if df: self.alpha=nn.Parameter(torch.tensor(0.5)); self.proj=nn.Conv2d(ic,oc,1)
        else: self.fp=nn.Conv2d(ic*2,oc,1)
    def forward(self, x, hf, lf):
        ho=self.hfa(x,hf); lo=self.lfa(x,lf)
        if self.df: a=torch.sigmoid(self.alpha); return self.proj(a*ho+(1-a)*lo)
        return self.fp(torch.cat([ho,lo],1))

class FMMPreWork(nn.Module):
    def __init__(self, df=False, ic=48):
        super().__init__()
        self.fd=FD(ic); self.fg=FrequencyGuidedAttentionModule(ic,ic,df)
        self.fh=FMgM(ic,True); self.fl=FMgM(ic,False)
    def forward(self, x):
        low,high=self.fd(x); return self.fg(x,self.fh(high),self.fl(low))+x

class FSAS(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.th=nn.Conv2d(dim,dim*6,1,bias=bias); self.thd=nn.Conv2d(dim*6,dim*6,3,1,1,groups=dim*6,bias=bias)
        self.po=nn.Conv2d(dim*2,dim,1,bias=bias); self.norm=LayerNorm(dim*2); self.ps=2
    def forward(self, x):
        q,k,v=self.thd(self.th(x)).chunk(3,1)
        qp=rearrange(q,'b c (h p1) (w p2)->b c h w p1 p2',p1=self.ps,p2=self.ps)
        kp=rearrange(k,'b c (h p1) (w p2)->b c h w p1 p2',p1=self.ps,p2=self.ps)
        out=torch.fft.irfft2(torch.fft.rfft2(qp.float())*torch.fft.rfft2(kp.float()),s=(self.ps,self.ps))
        return self.po(v*self.norm(rearrange(out,'b c h w p1 p2->b c (h p1) (w p2)',p1=self.ps,p2=self.ps)))

class DFFN(nn.Module):
    def __init__(self, dim, ffe=2.66, bias=False):
        super().__init__()
        h=int(dim*ffe); self.ps=2; self.pi=nn.Conv2d(dim,h,1,bias=bias)
        self.dw=nn.Conv2d(h,h*2,3,1,1,groups=h,bias=bias)
        self.fft=nn.Parameter(torch.ones((h,1,1,self.ps,self.ps//2+1))); self.po=nn.Conv2d(h,dim,1,bias=bias)
    def forward(self, x):
        x=self.pi(x); xp=rearrange(x,'b c (h p1) (w p2)->b c h w p1 p2',p1=self.ps,p2=self.ps)
        xp=torch.fft.irfft2(torch.fft.rfft2(xp.float())*self.fft,s=(self.ps,self.ps))
        x=rearrange(xp,'b c h w p1 p2->b c (h p1) (w p2)',p1=self.ps,p2=self.ps)
        x1,x2=self.dw(x).chunk(2,1); return self.po(F.gelu(x1)*x2)

class FFTransformerBlock(nn.Module):
    def __init__(self, dim, df=True, ffe=2.66, bias=False, att=False):
        super().__init__()
        self.att=att
        if att: self.norm1=LayerNorm(dim); self.attn=FSAS(dim,bias)
        self.norm2=LayerNorm(dim); self.ffn=DFFN(dim,ffe,bias)
        self.pc=nn.Conv2d(dim,dim,1); self.pb=FMMPreWork(df=df,ic=dim)
    def forward(self, x):
        # FIX Bug 6: sigmoid gate multiplies, not adds
        p1=self.pb(x); x=torch.sigmoid(self.pc(p1))*p1
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
    def __init__(self, inp=3, out=3, dim=48, nb=[4,6,6,8], nf=[4,2,2,1], nrb=4, heads=[1,2,4,8], ffe=2.66, bias=False, LN='WithBias'):
        super().__init__()
        self.pe=nn.Conv2d(inp,dim,3,1,1,bias=bias)
        self.e1=nn.Sequential(*[TransformerBlock(dim,heads[0],ffe,bias,LN) for _ in range(nb[0])])
        self.e1f=nn.Sequential(*[FFTransformerBlock(dim) for _ in range(nf[0])])
        self.d1=Downsample(dim)
        self.e2=nn.Sequential(*[TransformerBlock(dim*2,heads[1],ffe,bias,LN) for _ in range(nb[1])])
        self.e2f=nn.Sequential(*[FFTransformerBlock(dim*2) for _ in range(nf[1])])
        self.d2=Downsample(dim*2)
        self.e3=nn.Sequential(*[TransformerBlock(dim*4,heads[2],ffe,bias,LN) for _ in range(nb[2])])
        self.e3f=nn.Sequential(*[FFTransformerBlock(dim*4) for _ in range(nf[2])])
        self.d3=Downsample(dim*4)
        self.lat=nn.Sequential(*[TransformerBlock(dim*8,heads[3],ffe,bias,LN) for _ in range(nb[3])])
        # FIX Bug 2: bottleneck uses nf[3] (was nf[1])
        self.latf=nn.Sequential(*[FFTransformerBlock(dim*8) for _ in range(nf[3])])
        self.u3=Upsample(dim*8); self.r3=nn.Conv2d(dim*8,dim*4,1,bias=bias)
        self.dc3=nn.Sequential(*[TransformerBlock(dim*4,heads[2],ffe,bias,LN) for _ in range(nb[2])])
        self.dc3f=nn.Sequential(*[FFTransformerBlock(dim*4,df=True) for _ in range(nf[2])])
        self.u2=Upsample(dim*4); self.r2=nn.Conv2d(dim*4,dim*2,1,bias=bias)
        self.dc2=nn.Sequential(*[TransformerBlock(dim*2,heads[1],ffe,bias,LN) for _ in range(nb[1])])
        self.dc2f=nn.Sequential(*[FFTransformerBlock(dim*2,df=True) for _ in range(nf[1])])
        self.u1=Upsample(dim*2); self.r1=nn.Conv2d(dim*2,dim,1,bias=bias)
        self.dc1=nn.Sequential(*[TransformerBlock(dim,heads[0],ffe,bias,LN) for _ in range(nb[0])])
        self.dc1f=nn.Sequential(*[FFTransformerBlock(dim,df=True) for _ in range(nf[0])])
        self.refine=nn.Sequential(*[TransformerBlock(dim,heads[0],ffe,bias,LN) for _ in range(nrb)])
        self.output=nn.Conv2d(dim,out,3,1,1,bias=bias)
    def forward(self, img):
        e1=self.pe(img); e1=self.e1(self.e1f(e1))+e1
        e2=self.d1(e1); e2=self.e2(self.e2f(e2))+e2
        e3=self.d2(e2); e3=self.e3(self.e3f(e3))+e3
        e4=self.d3(e3); lat=self.lat(self.latf(e4))+e4
        d3=self.r3(torch.cat([self.u3(lat),e3],1)); d3=self.dc3(self.dc3f(d3))+d3
        d2=self.r2(torch.cat([self.u2(d3),e2],1)); d2=self.dc2(self.dc2f(d2))+d2
        d1=self.r1(torch.cat([self.u1(d2),e1],1)); d1=self.dc1(self.dc1f(d1))+d1
        return self.output(self.refine(d1))+img

# ===================== INFERENCE =====================
def pad_to_multiple(img, multiple=16):
    """Pad image so H and W are multiples of `multiple`."""
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img, h, w

def run_inference(input_path, output_path):
    """Run AFLB model on a single image."""
    
    # Load image
    img = Image.open(input_path).convert('RGB')
    orig_w, orig_h = img.size
    print(f"Input image: {input_path}")
    print(f"Original size: {orig_w}x{orig_h}")
    
    # Convert to tensor
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)  # (1, 3, H, W)
    
    # Pad to multiple of 16 (required by PixelUnshuffle)
    img_padded, orig_h, orig_w = pad_to_multiple(img_tensor, multiple=16)
    print(f"Padded size: {img_padded.shape[-2]}x{img_padded.shape[-1]}")
    
    # Create model (smaller config for fast inference)
    print("\nCreating AFLB model...")
    model = AdaIR(
        dim=48, 
        nb=[2, 2, 2, 2],      # Reduced blocks for speed
        nf=[2, 1, 1, 1],      # Reduced FFT blocks
        nrb=2,
        heads=[1, 2, 4, 8]
    ).to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Run inference
    print("\nRunning AFLB inference...")
    import time
    start = time.time()
    
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                restored = model(img_padded)
        else:
            restored = model(img_padded)
    
    elapsed = time.time() - start
    print(f"Inference time: {elapsed*1000:.1f}ms")
    
    # Remove padding and clamp
    restored = restored[:, :, :orig_h, :orig_w]
    restored = restored.clamp(0, 1)
    
    # Save output
    output_img = TF.to_pil_image(restored.squeeze(0).cpu())
    output_img.save(output_path, quality=95)
    print(f"\nOutput saved: {output_path}")
    print(f"Output size: {output_img.size[0]}x{output_img.size[1]}")
    
    # Also save side-by-side comparison
    comparison_path = output_path.replace('.png', '_comparison.png')
    
    # Resize input to match output for comparison
    input_resized = img.resize(output_img.size, Image.LANCZOS)
    
    # Create side-by-side
    comp_width = input_resized.width * 2 + 20
    comp_height = input_resized.height + 40
    comparison = Image.new('RGB', (comp_width, comp_height), (30, 30, 30))
    comparison.paste(input_resized, (0, 40))
    comparison.paste(output_img, (input_resized.width + 20, 40))
    
    # Add labels using simple drawing
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        draw.text((input_resized.width//2 - 20, 5), "INPUT", fill=(255, 100, 100), font=font)
        draw.text((input_resized.width + 20 + input_resized.width//2 - 30, 5), "AFLB OUTPUT", fill=(100, 255, 100), font=font)
    except:
        pass
    
    comparison.save(comparison_path, quality=95)
    print(f"Comparison saved: {comparison_path}")
    
    return output_img, comparison

# ===================== RUN =====================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "input_horse.png")
    output_path = os.path.join(script_dir, "output_restored.png")
    
    # Check if input exists
    if not os.path.exists(input_path):
        print(f"Input not found at {input_path}")
        print("Please ensure the horse image is saved there.")
        sys.exit(1)
    
    output_img, comparison = run_inference(input_path, output_path)
    print("\n[OK] Done!")

