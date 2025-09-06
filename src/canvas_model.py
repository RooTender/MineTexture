# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3(in_c, out_c):
    # circular padding dla tilingu
    return nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode='circular')

class Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.c1 = conv3(in_c, out_c)
        self.c2 = conv3(out_c, out_c)
        self.n1 = nn.GroupNorm(1, out_c)
        self.n2 = nn.GroupNorm(1, out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # manual circular pad (PyTorch Conv2d nie ma padding_mode='circular' z autogradem na wszystkich wersjach)
        x = self.c1(x); x = self.n1(x); x = self.act(x)
        x = self.c2(x); x = self.n2(x); x = self.act(x)
        return x

class TinyUNet(nn.Module):
    def __init__(self, in_ch=4, base=32, out_ch=4):
        super().__init__()
        self.b1 = Block(in_ch, base)
        self.d1 = nn.Conv2d(base, base*2, 4, 2, 1)
        self.b2 = Block(base*2, base*2)
        self.d2 = nn.Conv2d(base*2, base*4, 4, 2, 1)
        self.b3 = Block(base*4, base*4)

        self.up1 = nn.Conv2d(base*4, base*2, 1)
        self.b4  = Block(base*4, base*2)
        self.up2 = nn.Conv2d(base*2, base, 1)
        self.b5  = Block(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(self.d1(x1))
        x3 = self.b3(self.d2(x2))

        y  = F.interpolate(x3, scale_factor=2, mode='nearest')
        y  = self.up1(y)
        y  = self.b4(torch.cat([y, x2], dim=1))

        y  = F.interpolate(y, scale_factor=2, mode='nearest')
        y  = self.up2(y)
        y  = self.b5(torch.cat([y, x1], dim=1))

        return x + self.out(y)





# model_plus.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3(in_c, out_c):
    return nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, padding_mode='circular', bias=True)

def conv1(in_c, out_c):
    return nn.Conv2d(in_c, out_c, kernel_size=1, bias=True)

# --- Attention: SE + (opcjonalnie) prosty spatial (CBAM) ---
class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            conv1(c, max(1, c // r)),
            nn.SiLU(),
            conv1(max(1, c // r), c),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

class SpatialAttn(nn.Module):
    # CBAM-like: avg+max po kanałach -> conv7x7 (circular)
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, padding_mode='circular', bias=True)
        self.act = nn.Sigmoid()
    def forward(self, x):
        m = torch.mean(x, dim=1, keepdim=True)
        M, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([m, M], dim=1)
        a = self.act(self.conv(s))
        return x * a

# --- Residual block z opcjonalnym SE/CBAM i residual-scaling ---
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, use_se=True, use_spatial=True, scale=0.2):
        super().__init__()
        self.c1 = conv3(in_c, out_c)
        self.n1 = nn.GroupNorm(1, out_c)
        self.c2 = conv3(out_c, out_c)
        self.n2 = nn.GroupNorm(1, out_c)
        self.act = nn.SiLU()
        self.skip = conv1(in_c, out_c) if in_c != out_c else nn.Identity()
        self.se = SE(out_c) if use_se else nn.Identity()
        self.sa = SpatialAttn() if use_spatial else nn.Identity()
        self.scale = scale

    def forward(self, x):
        y = self.c1(x); y = self.n1(y); y = self.act(y)
        y = self.c2(y); y = self.n2(y)
        y = self.se(y)
        y = self.sa(y)
        res = self.skip(x)

        # (x->1x1) + scaled residual – stabilniej przy większym "base"
        return self.act(res + self.scale * y)

# --- MHSA tylko w bottlenecku: HW=8x8 przy 32x32 ---
class BottleneckMHSA(nn.Module):
    def __init__(self, c, heads=4, scale=0.2):
        super().__init__()
        assert c % heads == 0
        self.h = heads; self.d = c // heads
        self.qkv = conv1(c, c * 3)
        self.proj = conv1(c, c)
        self.scale = scale
        self.norm = nn.GroupNorm(1, c)
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=1)        # [B, C, H, W]
        # [B, h, d, HW]
        q = q.view(B, self.h, self.d, H*W)
        k = k.view(B, self.h, self.d, H*W)
        v = v.view(B, self.h, self.d, H*W)
        attn = torch.softmax((q.transpose(2,3) @ k) / (self.d ** 0.5), dim=-1)   # [B,h,HW,HW]
        y = attn @ v.transpose(2,3)                                              # [B,h,HW,d]
        y = y.transpose(2,3).contiguous().view(B, C, H, W)
        y = self.proj(y)
        # residual z lekkim skalowaniem – nie rozjeżdża treningu
        return self.norm(x + self.scale * y)
    
class GatedSkip(nn.Module):
    def __init__(self, c_low, c_high):
        super().__init__()
        # prosta bramka: 1x1 nad złączonym tensorem -> sigmoid
        self.gate = nn.Sequential(
            nn.Conv2d(c_low + c_high, c_high, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, low, high):   # low=skip z enkodera, high=cecha z dekodera
        g = self.gate(torch.cat([low, high], dim=1))
        return high * g + low * (1 - g)

class TinyUNetPlus(nn.Module):
    def __init__(self, in_ch=4, base=32, out_ch=4,
                 se=True, spatial=True, heads=4):
        super().__init__()
        c1, c2, c3 = base, base*2, base*4

        # self.g1 = GatedSkip(c2, c2)
        # self.g2 = GatedSkip(c1, c1)

        self.b1 = ResBlock(in_ch, c1, use_se=se, use_spatial=spatial, scale=0.2)
        self.d1 = nn.Conv2d(c1, c2, 4, 2, 1)      # zostawiamy nearest+conv w up
        self.b2 = ResBlock(c2, c2, use_se=se, use_spatial=spatial, scale=0.2)
        self.d2 = nn.Conv2d(c2, c3, 4, 2, 1)
        # Bottleneck: kilka ResBlocków + MHSA
        self.b3a = ResBlock(c3, c3, use_se=se, use_spatial=False, scale=0.2)
        self.att = BottleneckMHSA(c3, heads=heads, scale=0.2)
        self.b3b = ResBlock(c3, c3, use_se=se, use_spatial=False, scale=0.2)

        self.up1 = conv1(c3, c2)
        self.b4  = ResBlock(c2 + c2, c2, use_se=se, use_spatial=spatial, scale=0.2)
        self.up2 = conv1(c2, c1)
        self.b5  = ResBlock(c1 + c1, c1, use_se=se, use_spatial=spatial, scale=0.2)
        self.out = conv1(c1, out_ch)

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(self.d1(x1))
        x3 = self.b3a(self.d2(x2))
        x3 = self.att(x3)
        x3 = self.b3b(x3)

        y  = F.interpolate(x3, scale_factor=2, mode='nearest')
        y  = self.up1(y)
        # y  = self.g1(x2, y)
        y  = self.b4(torch.cat([y, x2], dim=1))

        y  = F.interpolate(y, scale_factor=2, mode='nearest')
        y  = self.up2(y)
        # y  = self.g2(x1, y)
        y  = self.b5(torch.cat([y, x1], dim=1))

        return x + self.out(y)
