# diff_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- helpers ---
def conv3(in_c, out_c):
    return nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode='circular', bias=True)

def conv1(in_c, out_c):
    return nn.Conv2d(in_c, out_c, kernel_size=1, bias=True)

# --- Timestep / Sigma embedding -> AdaGN (FiLM do normy) ---
class SigmaEmbed(nn.Module):
    def __init__(self, hidden=128, fourier=16):
        super().__init__()
        self.fourier = fourier
        self.mlp = nn.Sequential(
            nn.Linear(2*fourier, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU()
        )
    def forward(self, sigma):     # sigma: [B] (float)
        b = torch.arange(self.fourier, device=sigma.device, dtype=sigma.dtype)
        ang = sigma[:, None] * (2.0 ** b)[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)   # [B, 2*fourier]
        return self.mlp(emb)                                       # [B, hidden]

class AdaGN(nn.Module):
    def __init__(self, c, cond_dim):
        super().__init__()
        self.gn = nn.GroupNorm(1, c)
        self.to_scale = nn.Linear(cond_dim, c)
        self.to_shift = nn.Linear(cond_dim, c)
    def forward(self, x, h):
        y = self.gn(x)
        s = self.to_scale(h).unsqueeze(-1).unsqueeze(-1)
        b = self.to_shift(h).unsqueeze(-1).unsqueeze(-1)
        return y * (1 + s) + b

# --- lekkie attention kanałowe + residual scaling ---
class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(conv1(c, max(1, c//r)), nn.SiLU(), conv1(max(1, c//r), c), nn.Sigmoid())
    def forward(self, x): return x * self.fc(self.avg(x))

class ResBlockCond(nn.Module):
    def __init__(self, in_c, out_c, cond_dim, scale=0.2, se=True):
        super().__init__()
        self.c1 = conv3(in_c, out_c); self.n1 = AdaGN(out_c, cond_dim)
        self.c2 = conv3(out_c, out_c); self.n2 = AdaGN(out_c, cond_dim)
        self.act = nn.SiLU()
        self.skip = conv1(in_c, out_c) if in_c!=out_c else nn.Identity()
        self.se = SE(out_c) if se else nn.Identity()
        self.scale = scale
    def forward(self, x, h):
        y = self.c1(x); y = self.n1(y, h); y = self.act(y)
        y = self.c2(y); y = self.n2(y, h)
        y = self.se(y)
        s = x if isinstance(self.skip, nn.Identity) else self.skip(x)
        return self.act(s + self.scale * y)

class BottleneckMHSA(nn.Module):
    def __init__(self, c, heads=4, scale=0.2):
        super().__init__()
        assert c % heads == 0
        self.h = heads; self.d = c // heads
        self.qkv = conv1(c, c*3); self.proj = conv1(c, c)
        self.norm = nn.GroupNorm(1, c)
        self.scale = scale
    def forward(self, x):
        B,C,H,W = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=1)
        q = q.view(B,self.h,self.d,H*W).transpose(2,3)   # [B,h,HW,d]
        k = k.view(B,self.h,self.d,H*W)                  # [B,h,d,HW]
        v = v.view(B,self.h,self.d,H*W).transpose(2,3)   # [B,h,HW,d]
        a = torch.softmax((q @ k) / (self.d ** 0.5), dim=-1)  # [B,h,HW,HW]
        y = a @ v                                         # [B,h,HW,d]
        y = y.transpose(2,3).contiguous().view(B,C,H,W)
        y = self.proj(y)
        return self.norm(x + self.scale * y)

# --- UNet dyfuzyjny: wejście = [source, x_t, (opcjonalnie x0_hat_sc)] ---
class DiffUNet(nn.Module):
    def __init__(self, in_ch=4, base=48, out_ch=4, sc_channels=4, use_sc=True, heads=4):
        """
        in_ch: kanały źródła (vanilla), out_ch: predykowany epsilon (RGBA)
        sc_channels: kanały self-conditioning (x0_hat); gdy use_sc=False, ignorowane
        """
        super().__init__()
        self.use_sc = use_sc

        cin = in_ch + 4 + (sc_channels if use_sc else 0)  # source + x_t + (x0_hat)
        c1, c2, c3 = base, base*2, base*4

        cond_dim = 128
        self.sigma_emb = SigmaEmbed(hidden=cond_dim)

        self.b1 = ResBlockCond(cin, c1, cond_dim, scale=0.2, se=True)
        self.d1 = nn.Conv2d(c1, c2, 4, 2, 1)

        self.b2 = ResBlockCond(c2, c2, cond_dim, scale=0.2, se=True)
        self.d2 = nn.Conv2d(c2, c3, 4, 2, 1)

        self.b3a = ResBlockCond(c3, c3, cond_dim, scale=0.2, se=False)
        self.att = BottleneckMHSA(c3, heads=heads, scale=0.2)
        self.b3b = ResBlockCond(c3, c3, cond_dim, scale=0.2, se=False)

        self.up1 = conv1(c3, c2)
        self.b4  = ResBlockCond(c2+c2, c2, cond_dim, scale=0.2, se=True)

        self.up2 = conv1(c2, c1)
        self.b5  = ResBlockCond(c1+c1, c1, cond_dim, scale=0.2, se=True)

        self.out = conv1(c1, out_ch)   # przewidujemy epsilon (RGBA)

    def forward(self, source, x_t, sigma, x0_sc=None):
        """
        source: [B,4,H,W]  (vanilla)
        x_t   : [B,4,H,W]  (noisy sample)
        sigma : [B]        (poziom szumu)
        x0_sc : [B,4,H,W]  (opcjonalny self-conditioning: poprzednie x0_hat)
        """
        h = self.sigma_emb(sigma)  # [B, cond_dim]

        if self.use_sc:
            if x0_sc is None:
                x0_sc = torch.zeros_like(x_t)
            inp = torch.cat([source, x_t, x0_sc], dim=1)
        else:
            inp = torch.cat([source, x_t], dim=1)

        x1 = self.b1(inp, h)
        x2 = self.b2(self.d1(x1), h)
        x3 = self.b3a(self.d2(x2), h)
        x3 = self.att(x3)
        x3 = self.b3b(x3, h)

        y  = F.interpolate(x3, scale_factor=2, mode='nearest'); y = self.up1(y)
        y  = self.b4(torch.cat([y, x2], dim=1), h)

        y  = F.interpolate(y, scale_factor=2, mode='nearest'); y = self.up2(y)
        y  = self.b5(torch.cat([y, x1], dim=1), h)

        eps = self.out(y)
        return eps



# === Diffusion helpers ===
def default_sigmas(steps=3):
    # krótkie, do 32×32 w zupełności wystarcza
    if steps == 3:  return torch.tensor([0.50, 0.25, 0.00], dtype=torch.float32)
    if steps == 4:  return torch.tensor([0.70, 0.35, 0.15, 0.00], dtype=torch.float32)
    raise ValueError("steps must be 3 or 4")

def to_n11(x):     # [0,1] -> [-1,1]
    return x * 2.0 - 1.0

def from_n11(x):  # [-1,1] -> [0,1]
    return (x + 1.0) * 0.5

@torch.no_grad()
def ddim_infer(model, source01, steps=3, cfg_scale=1.5, use_sc=True):
    device = source01.device
    sigmas = default_sigmas(steps).to(device)
    B = source01.size(0)

    source = to_n11(source01)
    # KLUCZOWA ZMIANA: startuj z szumu o poziomie sigmas[0]
    x = sigmas[0] * torch.randn_like(source)
    x0_sc = torch.zeros_like(source) if use_sc else None

    for i in range(len(sigmas) - 1):
        s = torch.full((B,), sigmas[i], device=device)

        eps_c = model(source, x, s, x0_sc)
        if cfg_scale > 0.0:
            eps_u = model(torch.zeros_like(source), x, s, x0_sc)
            eps = eps_c + cfg_scale * (eps_c - eps_u)
        else:
            eps = eps_c

        x0_hat = x - sigmas[i] * eps
        if use_sc: x0_sc = x0_hat.detach()
        x = x0_hat + sigmas[i+1] * eps

    return from_n11(x).clamp(0, 1)
