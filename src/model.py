import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, padding_mode='zeros')

def conv1(in_c, out_c):
    return nn.Conv2d(in_c, out_c, kernel_size=1)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = conv3(ch, ch, stride=1)
        self.c2 = conv3(ch, ch, stride=1)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        y = self.act(self.c1(x))
        y = self.c2(y)
        return self.act(x + y)

class TinyUNet(nn.Module):
    """
    TinyUNet + ResBlocki + dodatkowe zejście do 4x4 (bez poszerzania kanałów na 4x4).
    Skala: 32 -> 16 -> 8 -> 4 -> 8 -> 16 -> 32
    """
    def __init__(self, in_ch=4, base=32, out_ch=4):
        super().__init__()

        self.act = nn.SiLU(inplace=True)

        # --- ENCODER ---
        self.e1 = conv3(in_ch, base, stride=1)      # 32x32,  base
        self.rb1 = ResBlock(base)

        self.e2 = conv3(base, base*2, stride=2)     # 16x16,  2*base
        self.rb2 = ResBlock(base*2)

        self.e3 = conv3(base*2, base*4, stride=2)   # 8x8,    4*base
        self.rb3 = ResBlock(base*4)

        # Nowe, dodatkowe zejście do 4x4 – zostawiamy tę samą szerokość (4*base)
        self.e4 = conv3(base*4, base*4, stride=2)   # 4x4,    4*base
        self.rb4 = ResBlock(base*4)

        # --- BOTTLENECK (4x4) ---
        self.b  = conv3(base*4, base*4, stride=1)   # 4x4

        # --- DECODER ---
        # 4 -> 8
        self.u0 = nn.ConvTranspose2d(base*4, base*4, kernel_size=4, stride=2, padding=1)
        self.rb0 = ResBlock(base*4)

        # 8 -> 16
        self.u1 = nn.ConvTranspose2d(base*4, base*2, kernel_size=4, stride=2, padding=1)
        self.rb5 = ResBlock(base*2)

        # 16 -> 32
        self.u2 = nn.ConvTranspose2d(base*2, base,   kernel_size=4, stride=2, padding=1)
        self.rb6 = ResBlock(base)

        # --- PŁYTKI SKIP 32x32 ---
        self.skip_mix = conv1(base * 2, base)

        # --- WYJŚCIE ---
        self.out = conv1(base, out_ch)

        # Inicjalizacja
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # ENCODER
        y1 = self.act(self.e1(x))   # 32x32
        y1 = self.rb1(y1)

        y = self.act(self.e2(y1))   # 16x16
        y = self.rb2(y)

        y = self.act(self.e3(y))   # 8x8
        y = self.rb3(y)

        y = self.act(self.e4(y))   # 4x4
        y = self.rb4(y)

        # BOTTLENECK
        y = self.act(self.b(y))    # 4x4

        # DECODER
        y = self.act(self.u0(y))   # 8x8
        y = self.rb0(y)

        y = self.act(self.u1(y))   # 16x16
        y = self.rb5(y)

        y = self.act(self.u2(y))   # 32x32
        y = self.rb6(y)

        y  = torch.cat([y, y1], dim=1)   # (base + base) -> 2*base
        y  = self.act(self.skip_mix(y))  # z powrotem do 'base'

        return x + self.out(y)
