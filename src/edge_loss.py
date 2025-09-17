import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = "mean",
                 thin_oriented: bool = False,  # NOWE: kierunkowe "cienienie"
                 gamma: float = 1.0):          # NOWE: wzmocnienie wierzchołków
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.thin_oriented = thin_oriented
        self.gamma = gamma

        # Central difference [-1, 0, 1]
        kx_cd = torch.tensor([[-1., 0., 1.]], dtype=torch.float32).view(1,1,1,3)
        ky_cd = torch.tensor([[-1.], [0.], [1.]], dtype=torch.float32).view(1,1,3,1)
        self.register_buffer("kx_cdiff", kx_cd)
        self.register_buffer("ky_cdiff", ky_cd)

    def _pad_reflect_xy(self, x, px: int, py: int):
        return F.pad(x, (px, px, py, py), mode="reflect")

    def _cdiff_gxgy_mag(self, x: torch.Tensor):
        """Zwraca (gx, gy, mag) z kierunkowym padem; rozmiar = wejście."""
        C = x.size(1)
        xw = self._pad_reflect_xy(x, px=1, py=0)
        gx = F.conv2d(xw, self.kx_cdiff.to(x.dtype).expand(C,1,1,3), groups=C)
        xh = self._pad_reflect_xy(x, px=0, py=1)
        gy = F.conv2d(xh, self.ky_cdiff.to(x.dtype).expand(C,1,3,1), groups=C)
        mag = torch.sqrt(gx*gx + gy*gy + self.eps)
        return gx, gy, mag

    def _thin_oriented(self, mag: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor, kappa: float = 8.0):
        """
        Kierunkowa soft-supresja: porównaj mag z próbkami wzdłuż ±θ (bilinear).
        Znika dithering na skosach; krawędź robi się 1-px i ciągła.
        """
        B, C, H, W = mag.shape
        # jednostkowy wektor kierunku (ux, uy)
        norm = torch.sqrt(gx*gx + gy*gy + self.eps)
        ux = gx / (norm + self.eps)
        uy = gy / (norm + self.eps)

        # siatka bazowa w koord. znormalizowanych [-1,1] (align_corners=True)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=mag.device, dtype=mag.dtype),
            torch.linspace(-1, 1, W, device=mag.device, dtype=mag.dtype),
            indexing='ij'
        )
        base = torch.stack([xx, yy], dim=-1)              # [H,W,2]
        base = base.view(1, 1, H, W, 2).expand(B, C, -1, -1, -1)

        # offset 1 piksela w koord. znormalizowanych
        off_x = 2.0 / (W - 1)
        off_y = 2.0 / (H - 1)
        dx = ux * off_x
        dy = uy * off_y

        grid_pos = base.clone()
        grid_neg = base.clone()
        grid_pos[..., 0] += dx
        grid_pos[..., 1] += dy
        grid_neg[..., 0] -= dx
        grid_neg[..., 1] -= dy

        m_in = mag.view(B * C, 1, H, W)
        gp = F.grid_sample(m_in, grid_pos.view(B * C, H, W, 2),
                           mode='bilinear', padding_mode='border',
                           align_corners=True).view(B, C, H, W)
        gn = F.grid_sample(m_in, grid_neg.view(B * C, H, W, 2),
                           mode='bilinear', padding_mode='border',
                           align_corners=True).view(B, C, H, W)

        neigh = torch.maximum(gp, gn)
        mask = torch.sigmoid(kappa * (mag - neigh))
        return mag * mask

    def _edge_map(self, x: torch.Tensor) -> torch.Tensor:
        # cdiff + (opcjonalnie) oriented thinning + gamma-sharpening
        gx, gy, mag = self._cdiff_gxgy_mag(x)
        if self.thin_oriented:
            mag = self._thin_oriented(mag, gx, gy)
        if self.gamma != 1.0:
            # zachowaj znak? tu mamy magnitudę, więc tylko potęga
            mag = torch.pow(torch.clamp(mag, min=0.0) + self.eps, self.gamma)
        return mag

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pr, tr = pred[:, :4], target[:, :4]
        gp = self._edge_map(pr)
        gt = self._edge_map(tr)
        diff = (gp - gt).abs()
        if self.reduction == "mean":
            return diff.mean()
        elif self.reduction == "sum":
            return diff.sum()
        else:
            return diff
