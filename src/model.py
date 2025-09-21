import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)

def conv1(in_c, out_c):
    return nn.Conv2d(in_c, out_c, kernel_size=1)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(ch, affine=True)
        self.c1 = conv3(ch, ch, stride=1)
        self.norm2 = nn.InstanceNorm2d(ch, affine=True)
        self.c2 = conv3(ch, ch, stride=1)
        self.act = nn.SiLU(inplace=True)

        # skalowanie residualu
        self.alpha = nn.Parameter(torch.ones(ch))

        # zero-init drugiej konw
        nn.init.zeros_(self.c2.weight)
        if self.c2.bias is not None:
            nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        y = self.c1(self.act(self.norm1(x)))
        y = self.c2(self.act(self.norm2(y)))
        return self.act(x + self.alpha.view(1,-1,1,1) * y)

class TinyUNet(nn.Module):
    """
    32 -> 16 -> 8 -> 4 -> 8 -> 16 -> 32
    ResBlock na każdym poziomie + JEDEN skip 32x32 (z e1) po ostatnim upsamplu.
    """
    def __init__(self, in_ch=4, base=32, out_ch=4):
        super().__init__()
        self.act = nn.SiLU(inplace=True)

        # --- ENCODER ---
        self.e1 = conv3(in_ch, base, stride=1)      # 32x32
        self.rb1 = ResBlock(base)

        self.e2 = conv3(base, base*2, stride=2)     # 16x16
        self.rb2 = ResBlock(base * 2)

        self.e3 = conv3(base*2, base*4, stride=2)   # 8x8
        self.rb3 = ResBlock(base*4)

        self.e4 = conv3(base*4, base*4, stride=2)   # 4x4 (nie poszerzamy kanałów)
        self.rb4 = ResBlock(base*4)

        # --- BOTTLENECK 4x4 ---
        self.b = nn.Sequential(
            conv3(base*4, base*4, stride=1),
            ResBlock(base*4),
        )

        # --- DECODER ---
        self.u0 = nn.ConvTranspose2d(base*4, base*4, kernel_size=4, stride=2, padding=1)  # 4->8
        self.rb_u0 = ResBlock(base*4)

        self.u1 = nn.ConvTranspose2d(base*4, base*2, kernel_size=4, stride=2, padding=1)  # 8->16
        self.rb_u1 = ResBlock(base * 2)

        self.u2 = nn.ConvTranspose2d(base*2, base,   kernel_size=4, stride=2, padding=1)  # 16->32
        self.rb_u2 = ResBlock(base)

        # --- SKIPY ---
        self.skip_scale_8 = nn.Parameter(torch.tensor(1.0))
        self.skip_mix_8 = conv1(base * 8, base * 4)

        self.skip_scale_16 = nn.Parameter(torch.tensor(1.0))
        self.skip_mix_16 = conv1(base * 4, base * 2)

        self.skip_scale_32 = nn.Parameter(torch.tensor(1.0))
        self.skip_mix_32 = conv1(base * 2, base)

        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.out = conv1(base, out_ch)

        # Inicjalizacja
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.zeros_(self.out.weight)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x):
        # Encoder
        y = self.act(self.e1(x))   # 32x32, base
        y = self.rb1(y)
        y_32 = y

        y  = self.act(self.e2(y_32))  # 16x16, 2*base
        y  = self.rb2(y)
        y_16 = y

        y  = self.act(self.e3(y))   # 8x8, 4*base
        y  = self.rb3(y)
        y_8 = y

        y  = self.act(self.e4(y))   # 4x4, 4*base
        y  = self.rb4(y)

        # Bottleneck
        y  = self.act(self.b(y))    # 4x4

        # Decoder
        y  = self.act(self.u0(y))   # 8x8
        y  = self.rb_u0(y)

        y  = torch.cat([y, self.skip_scale_8 * y_8], dim=1)
        y  = self.act(self.skip_mix_8(y))

        y  = self.act(self.u1(y))   # 16x16
        y  = self.rb_u1(y)

        y  = torch.cat([y, self.skip_scale_16 * y_16], dim=1)
        y  = self.act(self.skip_mix_16(y))

        y  = self.act(self.u2(y))   # 32x32
        y  = self.rb_u2(y)

        # Skip 32x32
        y  = torch.cat([y, self.skip_scale_32 * y_32], dim=1)
        y  = self.act(self.skip_mix_32(y))

        y = x + self.alpha * self.out(y)
        return F.hardtanh(y, 0.0, 1.0)
