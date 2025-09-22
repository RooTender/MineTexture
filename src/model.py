import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3(in_c, out_c, stride=1, dilation=1):
    pad = dilation
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=pad, dilation=dilation)

def conv1(in_c, out_c):
    return nn.Conv2d(in_c, out_c, kernel_size=1)

class ResBlock(nn.Module):
    def __init__(self, ch, dilation=1):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(ch, affine=True)
        self.c1 = conv3(ch, ch, stride=1, dilation=dilation)
        self.act = nn.SiLU(inplace=True)

        self.alpha = nn.Parameter(torch.ones(ch))

    def forward(self, x):
        y = self.c1(self.act(self.norm1(x)))
        return self.act(x + self.alpha.view(1, -1, 1, 1) * y)
    
class SelfAttention2d(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.q = conv1(in_ch, in_ch // 8)
        self.k = conv1(in_ch, in_ch // 8)
        self.v = conv1(in_ch, in_ch)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.q(x).view(B, -1, H * W)           # B, Cq, N
        k = self.k(x).view(B, -1, H * W)           # B, Ck, N
        v = self.v(x).view(B, -1, H * W)           # B, Cv, N

        attn = torch.bmm(q.permute(0, 2, 1), k)    # B, N, N
        attn = F.softmax(attn / (q.shape[1] ** 0.5), dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))  # B, Cv, N
        out = out.view(B, C, H, W)

        return x + self.gamma * out


class TinyUNet(nn.Module):
    """
    64 -> 32 -> 16 -> 8 -> 16 -> 32 -> 64
    ResBlock na każdym poziomie + JEDEN skip 32x32 (z e1) po ostatnim upsamplu.
    """
    def __init__(self, in_ch=4, base=64, out_ch=4):
        super().__init__()
        self.act = nn.SiLU(inplace=True)

        # --- ENCODER ---
        self.e1 = conv3(in_ch, base, stride=1)      # 32x32
        self.rb1 = ResBlock(base)

        self.e2 = conv3(base, base*2, stride=2)     # 16x16
        self.rb2 = ResBlock(base * 2)
        # self.attn32 = SelfAttention2d(base*2)

        self.e3 = conv3(base*2, base*4, stride=2)   # 8x8
        self.rb3 = ResBlock(base*4)


        # --- BOTTLENECK 4x4 ---
        self.b = nn.Sequential(
            ResBlock(base*4, dilation=1),
            SelfAttention2d(base*4),
        )

        # --- DECODER ---
        self.u1 = nn.ConvTranspose2d(base*4, base*2, kernel_size=4, stride=2, padding=1)  # 8->16
        self.rb_u1 = ResBlock(base * 2)

        self.u2 = nn.ConvTranspose2d(base*2, base,   kernel_size=4, stride=2, padding=1)  # 16->32
        self.rb_u2 = ResBlock(base)

        # --- SKIPY ---
        self.skip8_proj  = conv1(base*4, base*4)   # proj do tego samego wymiaru
        self.skip8_alpha = nn.Parameter(torch.tensor(1.0))
        self.skip8_norm  = nn.InstanceNorm2d(base*4, affine=True)

        self.skip16_proj  = conv1(base*2, base*2)
        self.skip16_alpha = nn.Parameter(torch.tensor(1.0))
        self.skip16_norm  = nn.InstanceNorm2d(base*2, affine=True)

        self.skip32_proj  = conv1(base, base)
        self.skip32_alpha = nn.Parameter(torch.tensor(1.0))
        self.skip32_norm  = nn.InstanceNorm2d(base, affine=True)

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
        # y  = self.attn32(y)
        y_16 = y

        y  = self.act(self.e3(y))   # 8x8, 4*base
        y  = self.rb3(y)

        # Bottleneck
        y  = self.act(self.b(y))    # 4x4

        # Decoder
        # 16x16
        y = self.act(self.u1(y))
        y = self.rb_u1(y)
        skip = self.act(self.skip16_norm(y_16))
        skip = self.skip16_proj(skip)
        y = y + self.skip16_alpha * skip

        # 32x32
        y = self.act(self.u2(y))
        y = self.rb_u2(y)
        skip = self.act(self.skip32_norm(y_32))
        skip = self.skip32_proj(skip)
        y = y + self.skip32_alpha * skip

        y = x + self.alpha * self.out(y)
        return F.hardtanh(y, 0.0, 1.0)
