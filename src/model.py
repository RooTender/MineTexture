import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3(in_c, out_c):
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
