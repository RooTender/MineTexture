import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3(in_c, out_c, padding_mode='zeros', dilation=1):
    pad = dilation
    return nn.Conv2d(in_c, out_c, kernel_size=3, padding=pad, dilation=dilation,
                     padding_mode=padding_mode if padding_mode!='zeros' else 'zeros', bias=True)

def conv1(in_c, out_c):
    return nn.Conv2d(in_c, out_c, kernel_size=1, bias=True)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, scale=0.2, padding_mode='zeros', dilation=1):
        super().__init__()
        self.c1 = conv3(in_c, out_c, padding_mode, dilation=1)
        self.n1 = nn.GroupNorm(1, out_c)
        self.c2 = conv3(out_c, out_c, padding_mode, dilation=dilation)  # <- tu wstrzykujemy dilation
        self.n2 = nn.GroupNorm(1, out_c)
        self.act = nn.SiLU()
        self.skip = conv1(in_c, out_c) if in_c != out_c else nn.Identity()
        self.scale = scale

    def forward(self, x):
        y = self.c1(x); y = self.n1(y); y = self.act(y)
        y = self.c2(y); y = self.n2(y)
        return self.act(self.skip(x) + self.scale * y)

class TinyUNetLite(nn.Module):
    def __init__(self, in_ch=4, base=32, out_ch=4, padding_mode='zeros', bottleneck_dilation=1):
        super().__init__()
        c1, c2, c3 = base, base*2, base*4

        # Encoder
        self.b1 = ResBlock(in_ch, c1, padding_mode=padding_mode, dilation=1)
        self.d1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)  # /2
        self.b2 = ResBlock(c2, c2, padding_mode=padding_mode, dilation=1)
        self.d2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)  # /4
        self.b3 = ResBlock(c3, c3, padding_mode=padding_mode, dilation=bottleneck_dilation)  # RF+

        # Decoder
        self.up1 = conv1(c3, c2)
        self.b4  = ResBlock(c2 + c2, c2, padding_mode=padding_mode, dilation=1)
        self.up2 = conv1(c2, c1)
        self.b5  = ResBlock(c1 + c1, c1, padding_mode=padding_mode, dilation=1)
        self.out = conv1(c1, out_ch)

    @staticmethod
    def _pick_depth_from_hw(H, W):
        m = min(H, W)
        if m <= 16:  return 0          # malutkie (8→16, 16→16): najpłycej
        if m <= 64:  return 1          # 32–64: pół-UNet
        return 2                        # ≥128: pełny UNet

    def forward(self, x, depth=None):
        if depth is None:
            depth = self._pick_depth_from_hw(x.shape[-2], x.shape[-1])

        # wspólne wejście
        x1 = self.b1(x)                         # H, W

        if depth >= 1:
            x2 = self.b2(self.d1(x1))           # H/2, W/2
        else:
            x2 = None

        if depth >= 2:
            x3 = self.b3(self.d2(x2))           # H/4, W/4
        else:
            x3 = None

        # dekoder zależny od głębokości
        if depth == 2:
            y = F.interpolate(x3, scale_factor=2, mode='nearest')  # H/2
            y = self.up1(y)
            y = self.b4(torch.cat([y, x2], dim=1))
            y = F.interpolate(y, scale_factor=2, mode='nearest')   # H
            y = self.up2(y)
            y = self.b5(torch.cat([y, x1], dim=1))

        elif depth == 1:
            # pracujemy tylko na poziomach H/2 ↔ H
            y = self.b4(torch.cat([x2, x2], dim=1))                # bez up1
            y = F.interpolate(y, scale_factor=2, mode='nearest')   # H
            y = self.up2(y)
            y = self.b5(torch.cat([y, x1], dim=1))

        else:  # depth == 0
            # najpłytsza ścieżka: lokalny refine na H
            y = self.b5(torch.cat([x1, x1], dim=1))

        return x + self.out(y)  # residual = ostrość



def l1_masked(pred, target, mask):
    l = (pred - target).abs() * mask
    return l.sum() / (mask.sum() * pred.size(1) + 1e-8)
