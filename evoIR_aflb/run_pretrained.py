"""
Download pretrained EvoIR weights and run inference with them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import os
import time
from einops import rearrange
from collections import OrderedDict
from PIL import Image
import torchvision.transforms.functional as TF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ===================== STEP 1: DOWNLOAD FROM HUGGINGFACE =====================
print("\n" + "="*60)
print("STEP 1: Downloading pretrained weights from HuggingFace")
print("="*60)

from huggingface_hub import hf_hub_download

pretrained_dir = os.path.join(os.path.dirname(__file__), 'pretrained')
os.makedirs(pretrained_dir, exist_ok=True)

ckpt_path = os.path.join(pretrained_dir, 'model_landsat.pth')
if not os.path.exists(ckpt_path):
    print("Downloading model_landsat.pth from leonmakise/EvoIR...")
    ckpt_path = hf_hub_download(
        repo_id='leonmakise/EvoIR',
        filename='model_landsat.pth',
        local_dir=pretrained_dir
    )
    print(f"Downloaded to: {ckpt_path}")
else:
    print(f"Already cached: {ckpt_path}")

# ===================== STEP 2: INSPECT CHECKPOINT =====================
print("\n" + "="*60)
print("STEP 2: Inspecting checkpoint structure")
print("="*60)

checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

if isinstance(checkpoint, dict):
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'epoch' in checkpoint:
            print(f"Trained for {checkpoint['epoch']+1} epochs")
        if 'task_weight' in checkpoint:
            print(f"EOS weights: {checkpoint['task_weight']}")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Clean DDP module prefix
cleaned_sd = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '')
    cleaned_sd[new_key] = v

print(f"\nTotal parameters in checkpoint: {len(cleaned_sd)}")
# Show first 20 keys to understand naming
print("\nFirst 20 parameter keys:")
for i, k in enumerate(list(cleaned_sd.keys())[:20]):
    print(f"  {k}: {cleaned_sd[k].shape}")

# ===================== STEP 3: BUILD MODEL FROM CHECKPOINT KEYS =====================
print("\n" + "="*60)
print("STEP 3: Building model architecture to match checkpoint")
print("="*60)

# === Full model definition (matches EvoIR source) ===
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
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1=LayerNorm(dim,LayerNorm_type); self.attn=Attention(dim,num_heads,bias)
        self.norm2=LayerNorm(dim,LayerNorm_type); self.ffn=FeedForward(dim,ffn_expansion_factor,bias)
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
    def forward(self, x, freq_feat):
        B,C,H,W=x.shape; q=self.q_dwconv(self.q_proj(freq_feat)); kv=self.kv_dwconv(self.kv_proj(x)); k,v=kv.chunk(2,1)
        q=rearrange(q,'b (h c) H W->b h c (H W)',h=self.num_heads); k=rearrange(k,'b (h c) H W->b h c (H W)',h=self.num_heads)
        v=rearrange(v,'b (h c) H W->b h c (H W)',h=self.num_heads)
        q=F.normalize(q,-1); k=F.normalize(k,-1)
        return self.project_out(rearrange((q@k.transpose(-2,-1)*self.temperature).softmax(-1)@v,'b h c (H W)->b (h c) H W',h=self.num_heads,H=H,W=W))

class FrequencyGuidedAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, decoder_flag=False):
        super().__init__()
        self.decoder_flag=decoder_flag
        self.high_freq_attention=FrequencyGuidedAttention(in_channels,out_channels)
        self.low_freq_attention=FrequencyGuidedAttention(in_channels,out_channels)
        if decoder_flag:
            self.alpha=nn.Parameter(torch.tensor(0.5)); self.proj=nn.Conv2d(in_channels,out_channels,1)
        else:
            self.final_proj=nn.Conv2d(in_channels*2,out_channels,1)
    def forward(self, x, hf, lf):
        ho=self.high_freq_attention(x,hf); lo=self.low_freq_attention(x,lf)
        if self.decoder_flag:
            a=torch.sigmoid(self.alpha); return self.proj(a*ho+(1-a)*lo)
        return self.final_proj(torch.cat([ho,lo],1))

class FMMPreWork(nn.Module):
    def __init__(self, decoder_flag=False, inchannels=48):
        super().__init__()
        self.fd=FD(inchannels); self.decoder_flag=decoder_flag
        self.freguide=FrequencyGuidedAttentionModule(inchannels,inchannels,decoder_flag)
        self.FSPG_high=FMgM(inchannels,True); self.FSPG_low=FMgM(inchannels,False)
    def forward(self, x):
        low,high=self.fd(x); return self.freguide(x,self.FSPG_high(high),self.FSPG_low(low))+x

class FSAS(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.to_hidden=nn.Conv2d(dim,dim*6,1,bias=bias); self.to_hidden_dw=nn.Conv2d(dim*6,dim*6,3,1,1,groups=dim*6,bias=bias)
        self.project_out=nn.Conv2d(dim*2,dim,1,bias=bias); self.norm=LayerNorm(dim*2); self.patch_size=2
    def forward(self, x):
        q,k,v=self.to_hidden_dw(self.to_hidden(x)).chunk(3,1)
        qp=rearrange(q,'b c (h p1) (w p2)->b c h w p1 p2',p1=self.patch_size,p2=self.patch_size)
        kp=rearrange(k,'b c (h p1) (w p2)->b c h w p1 p2',p1=self.patch_size,p2=self.patch_size)
        out=torch.fft.irfft2(torch.fft.rfft2(qp.float())*torch.fft.rfft2(kp.float()),s=(self.patch_size,self.patch_size))
        return self.project_out(v*self.norm(rearrange(out,'b c h w p1 p2->b c (h p1) (w p2)',p1=self.patch_size,p2=self.patch_size)))

class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden=int(dim*ffn_expansion_factor); self.patch_size=2
        self.project_in=nn.Conv2d(dim,hidden,1,bias=bias)
        self.dwconv=nn.Conv2d(hidden,hidden*2,3,1,1,groups=hidden,bias=bias)
        self.fft=nn.Parameter(torch.ones((hidden,1,1,self.patch_size,self.patch_size//2+1)))
        self.project_out=nn.Conv2d(hidden,dim,1,bias=bias)
    def forward(self, x):
        x=self.project_in(x)
        xp=rearrange(x,'b c (h p1) (w p2)->b c h w p1 p2',p1=self.patch_size,p2=self.patch_size)
        xp=torch.fft.irfft2(torch.fft.rfft2(xp.float())*self.fft,s=(self.patch_size,self.patch_size))
        x=rearrange(xp,'b c h w p1 p2->b c (h p1) (w p2)',p1=self.patch_size,p2=self.patch_size)
        x1,x2=self.dwconv(x).chunk(2,1); return self.project_out(F.gelu(x1)*x2)

class FFTransformerBlock(nn.Module):
    def __init__(self, dim, decoder_flag=True, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', att=False):
        super().__init__()
        self.att=att
        if att: self.norm1=LayerNorm(dim,LayerNorm_type); self.attn=FSAS(dim,bias)
        self.norm2=LayerNorm(dim,LayerNorm_type); self.ffn=DFFN(dim,ffn_expansion_factor,bias)
        self.prompt_conv=nn.Conv2d(dim,dim,1)
        self.Wp=nn.Parameter(torch.randn(dim,dim))
        self.prompt_block=FMMPreWork(decoder_flag=decoder_flag,inchannels=dim)
    def forward(self, x):
        prompt1=self.prompt_block(x); prompt2=self.prompt_conv(prompt1)
        prompt=torch.sigmoid(prompt2); x=prompt+prompt1
        if self.att: x=x+self.attn(self.norm1(x))
        return x+self.ffn(self.norm2(x))

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(n_feat,n_feat//2,3,1,1,bias=False),nn.PixelUnshuffle(2))
    def forward(self, x): return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(n_feat,n_feat*2,3,1,1,bias=False),nn.PixelShuffle(2))
    def forward(self, x): return self.body(x)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj=nn.Conv2d(in_c,embed_dim,3,1,1,bias=bias)
    def forward(self, x): return self.proj(x)

# Use EXACT same naming as official EvoIR to match checkpoint keys
class AdaIR(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48,
                 num_blocks=[4,6,6,8], num_fft_blocks=[4,2,2,1],
                 num_refinement_blocks=4, heads=[1,2,4,8],
                 ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias'):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim,heads[0],ffn_expansion_factor,bias,LayerNorm_type) for _ in range(num_blocks[0])])
        self.encoder_level1_fft = nn.Sequential(*[FFTransformerBlock(dim) for _ in range(num_fft_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim*2,heads[1],ffn_expansion_factor,bias,LayerNorm_type) for _ in range(num_blocks[1])])
        self.encoder_level2_fft = nn.Sequential(*[FFTransformerBlock(dim*2) for _ in range(num_fft_blocks[1])])
        self.down2_3 = Downsample(dim*2)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim*4,heads[2],ffn_expansion_factor,bias,LayerNorm_type) for _ in range(num_blocks[2])])
        self.encoder_level3_fft = nn.Sequential(*[FFTransformerBlock(dim*4) for _ in range(num_fft_blocks[2])])
        self.down3_4 = Downsample(dim*4)
        self.latent = nn.Sequential(*[TransformerBlock(dim*8,heads[3],ffn_expansion_factor,bias,LayerNorm_type) for _ in range(num_blocks[3])])
        self.latent_fft = nn.Sequential(*[FFTransformerBlock(dim*8) for _ in range(num_fft_blocks[3])])
        self.up4_3 = Upsample(dim*8); self.reduce_chan_level3 = nn.Conv2d(dim*8,dim*4,1,bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim*4,heads[2],ffn_expansion_factor,bias,LayerNorm_type) for _ in range(num_blocks[2])])
        self.decoder_level3_fft = nn.Sequential(*[FFTransformerBlock(dim*4,decoder_flag=True) for _ in range(num_fft_blocks[2])])
        self.up3_2 = Upsample(dim*4); self.reduce_chan_level2 = nn.Conv2d(dim*4,dim*2,1,bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim*2,heads[1],ffn_expansion_factor,bias,LayerNorm_type) for _ in range(num_blocks[1])])
        self.decoder_level2_fft = nn.Sequential(*[FFTransformerBlock(dim*2,decoder_flag=True) for _ in range(num_fft_blocks[1])])
        self.up2_1 = Upsample(dim*2); self.reduce_chan_level1 = nn.Conv2d(dim*2,dim,1,bias=bias)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim,heads[0],ffn_expansion_factor,bias,LayerNorm_type) for _ in range(num_blocks[0])])
        self.decoder_level1_fft = nn.Sequential(*[FFTransformerBlock(dim,decoder_flag=True) for _ in range(num_fft_blocks[0])])
        self.refinement = nn.Sequential(*[TransformerBlock(dim,heads[0],ffn_expansion_factor,bias,LayerNorm_type) for _ in range(num_refinement_blocks)])
        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias)
        self.decoder=True

    def forward(self, inp_img, noise_emb=None):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1_fft=self.encoder_level1_fft(inp_enc_level1)
        out_enc_level1 = self.encoder_level1(out_enc_level1_fft)+inp_enc_level1
        inp_enc_level2=self.down1_2(out_enc_level1)
        out_enc_level2_fft=self.encoder_level2_fft(inp_enc_level2)
        out_enc_level2=self.encoder_level2(out_enc_level2_fft)+inp_enc_level2
        inp_enc_level3=self.down2_3(out_enc_level2)
        out_enc_level3_fft=self.encoder_level3_fft(inp_enc_level3)
        out_enc_level3=self.encoder_level3(out_enc_level3_fft)+inp_enc_level3
        inp_enc_level4=self.down3_4(out_enc_level3)
        out_enc_level4_fft=self.latent_fft(inp_enc_level4)
        latent=self.latent(out_enc_level4_fft)+inp_enc_level4
        if self.decoder:
            inp_dec_level3=self.up4_3(latent)
            inp_dec_level3=torch.cat([inp_dec_level3,out_enc_level3],1)
            inp_dec_level3=self.reduce_chan_level3(inp_dec_level3)
            inp_dec_level3_fft=self.decoder_level3_fft(inp_dec_level3)
            out_dec_level3=self.decoder_level3(inp_dec_level3_fft)+inp_dec_level3
            inp_dec_level2=self.up3_2(out_dec_level3)
            inp_dec_level2=torch.cat([inp_dec_level2,out_enc_level2],1)
            inp_dec_level2=self.reduce_chan_level2(inp_dec_level2)
            inp_dec_level2_fft=self.decoder_level2_fft(inp_dec_level2)
            out_dec_level2=self.decoder_level2(inp_dec_level2_fft)+inp_dec_level2
            inp_dec_level1=self.up2_1(out_dec_level2)
            inp_dec_level1=torch.cat([inp_dec_level1,out_enc_level1],1)
            inp_dec_level1=self.reduce_chan_level1(inp_dec_level1)
            inp_dec_level1_fft=self.decoder_level1_fft(inp_dec_level1)
            out_dec_level1=self.decoder_level1(inp_dec_level1_fft)+inp_dec_level1
        out_dec_level1=self.refinement(out_dec_level1)
        out_dec_level1=self.output(out_dec_level1)+inp_img
        return out_dec_level1

# ===================== STEP 4: LOAD WEIGHTS =====================
print("\n" + "="*60)
print("STEP 3: Loading pretrained weights into model")
print("="*60)

model = AdaIR().to(device)
total = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total:,} ({total/1e6:.2f}M)")

# Try loading with exact keys
result = model.load_state_dict(cleaned_sd, strict=False)
matched = len(model.state_dict()) - len(result.missing_keys)
print(f"\n  Matched keys:    {matched}")
print(f"  Missing keys:    {len(result.missing_keys)}")
print(f"  Unexpected keys: {len(result.unexpected_keys)}")

if result.missing_keys:
    print(f"\n  Missing (first 5): {result.missing_keys[:5]}")
if result.unexpected_keys:
    print(f"\n  Unexpected (first 5): {result.unexpected_keys[:5]}")

if matched > 0:
    print(f"\n✅ Successfully loaded {matched} pretrained parameter tensors!")
else:
    print("\n⚠️ No keys matched. Running with partial/random weights.")

# ===================== STEP 5: INFERENCE =====================
print("\n" + "="*60)
print("STEP 4: Running inference on horse image")
print("="*60)

def pad_to_multiple(img, multiple=16):
    _, _, h, w = img.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
    return img, h, w

input_path = os.path.join(os.path.dirname(__file__), 'test2.png')
output_path = os.path.join(os.path.dirname(__file__), 'test2output.png')

img = Image.open(input_path).convert('RGB')
orig_w, orig_h = img.size
print(f"Input: {input_path} ({orig_w}x{orig_h})")

# Resize to manageable size for CPU inference if needed
max_dim = 256
if max(orig_w, orig_h) > max_dim and device.type == 'cpu':
    ratio = max_dim / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
    # Make divisible by 16
    new_w = (new_w // 16) * 16
    new_h = (new_h // 16) * 16
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    print(f"Resized to {new_w}x{new_h} for CPU inference")
else:
    img_resized = img

img_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)
img_padded, h_orig, w_orig = pad_to_multiple(img_tensor, 16)
print(f"Processing size: {img_padded.shape[-2]}x{img_padded.shape[-1]}")

model.eval()
start = time.time()
with torch.no_grad():
    # Swap RGB to BGR (many pretrained models expect BGR)
    img_bgr = img_padded[:, [2, 1, 0], :, :]
    restored = model(img_bgr)
    # Swap back from BGR to RGB
    restored = restored[:, [2, 1, 0], :, :]
elapsed = time.time() - start
print(f"Inference time: {elapsed:.2f}s")

restored = restored[:, :, :h_orig, :w_orig].clamp(0, 1)

# Save output at original resolution
out_pil = TF.to_pil_image(restored.squeeze(0).cpu())
if out_pil.size != (orig_w, orig_h):
    out_pil = out_pil.resize((orig_w, orig_h), Image.LANCZOS)
out_pil.save(output_path, quality=95)
print(f"Output saved: {output_path}")

# Side-by-side comparison
comp_path = os.path.join(os.path.dirname(__file__), 'test2output_comparison.png')
comp_w = orig_w * 2 + 20; comp_h = orig_h + 50
comp = Image.new('RGB', (comp_w, comp_h), (30, 30, 30))
comp.paste(img, (0, 50)); comp.paste(out_pil, (orig_w + 20, 50))
try:
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comp)
    try: font = ImageFont.truetype("arial.ttf", 16)
    except: font = ImageFont.load_default()
    draw.text((orig_w//2 - 30, 10), "INPUT (Degraded)", fill=(255,120,120), font=font)
    draw.text((orig_w + 20 + orig_w//2 - 50, 10), "AFLB OUTPUT (Pretrained)", fill=(120,255,120), font=font)
except: pass
comp.save(comp_path, quality=95)
print(f"Comparison saved: {comp_path}")

print(f"\n{'='*60}")
print("✅ DONE! Pretrained EvoIR AFLB inference complete!")
print(f"{'='*60}")
