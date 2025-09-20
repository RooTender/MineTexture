import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

        # central difference [-1,0,1]
        kx_cd = torch.tensor([[-1., 0., 1.]], dtype=torch.float32).view(1,1,1,3)
        ky_cd = torch.tensor([[-1.], [0.], [1.]], dtype=torch.float32).view(1,1,3,1)
        self.register_buffer("kx_cdiff", kx_cd)
        self.register_buffer("ky_cdiff", ky_cd)

    def _pad_reflect_xy(self, x, px: int, py: int):
        return F.pad(x, (px, px, py, py))

    @staticmethod
    def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
        # x in [0,1]
        a = 0.055
        low  = x <= 0.04045
        high = ~low
        y = torch.empty_like(x)
        y[low]  = x[low] / 12.92
        y[high] = ((x[high] + a) / (1 + a)).pow(2.4)
        return y

    @staticmethod
    def luma709_linear(x_lin: torch.Tensor) -> torch.Tensor:
        w = torch.tensor([0.2126, 0.7152, 0.0722],
                         dtype=x_lin.dtype, device=x_lin.device).view(1,3,1,1)
        return (x_lin * w).sum(dim=1, keepdim=True)

    def edge_map(self, x: torch.Tensor):
        C = x.size(1)
        xw = self._pad_reflect_xy(x, px=1, py=0)
        xh = self._pad_reflect_xy(x, px=0, py=1)
        gx = F.conv2d(xw, self.kx_cdiff.to(x.dtype).expand(C,1,1,3), groups=C)
        gy = F.conv2d(xh, self.ky_cdiff.to(x.dtype).expand(C,1,3,1), groups=C)
        return torch.sqrt(gx*gx + gy*gy + self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pr_rgb, tr_rgb = pred[:, :3], target[:, :3]
        pr_a,   tr_a   = pred[:, 3:4], target[:, 3:4]

        # 1) linearize sRGB
        pr_rgb_lin = self.srgb_to_linear(pr_rgb.clamp(0,1))
        tr_rgb_lin = self.srgb_to_linear(tr_rgb.clamp(0,1))

        # 2) luminancja
        pr_y = self.luma709_linear(pr_rgb_lin)
        tr_y = self.luma709_linear(tr_rgb_lin)

        # 3) edge maps
        gp_y = self.edge_map(pr_y)
        gt_y = self.edge_map(tr_y)
        loss_y = (gp_y - gt_y).abs().mean()

        gp_a = self.edge_map(pr_a)
        gt_a = self.edge_map(tr_a)
        loss_a = (gp_a - gt_a).abs().mean()

        return loss_y + loss_a



def _string_mask_from_target(target, edge_module,
                            thr_luma=0.6, thr_edge=0.4):
    """
    Zwraca maskę [B,1,H,W] ~ {0,1} na 'cięciwę' (jasna + krawędź).
    Prosta, bez CPU/numphy, działa w AMP.
    """
    # luminancja (linear) + opcjonalnie premultiply alphą, jeśli chcesz
    tr_rgb_lin = edge_module.srgb_to_linear(target[:, :3].clamp(0,1).float())
    tr_y = edge_module.luma709_linear(tr_rgb_lin)  # [B,1,H,W]
    # krawędź (magnituda)
    edge_mag = edge_module.edge_map(tr_y).float()

    # normalizacje „na szybko”
    y_max = tr_y.flatten(1).amax(dim=1, keepdim=True).view(-1,1,1,1).clamp_min(1e-8)
    e_max = edge_mag.flatten(1).amax(dim=1, keepdim=True).view(-1,1,1,1).clamp_min(1e-8)
    y_n = (tr_y / y_max)          # 0..1
    e_n = (edge_mag / e_max)      # 0..1

    # „jasne” AND „krawędź”
    m = (y_n > thr_luma).float() * (e_n > thr_edge).float()  # [B,1,H,W]

    # delikatna dylacja (złap 1 piksel halo)
    m = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
    return m


def weighted_l1(pred, target, edge_module, edge_w: float = 3.0,
                base_w: float = 1.0):
    tr_rgb_lin = edge_module.srgb_to_linear(target[:, :3].clamp(0,1).float())
    tr_y       = edge_module.luma709_linear(tr_rgb_lin).float()
    edge_mag   = edge_module.edge_map(tr_y).float()

    denom = edge_mag.flatten(1).amax(dim=1, keepdim=True).view(-1,1,1,1).clamp_min(1e-8)
    edge_norm = (edge_mag / denom).clamp_(0, 1)

    w = base_w + edge_w * edge_norm
    extra_mask = _string_mask_from_target(target, edge_module)

    if extra_mask is not None:
        em = extra_mask.detach().float()
        if em.dim() == 3: em = em.unsqueeze(1)
        w = w + em

    return (w.to(pred.dtype) * (pred - target).abs()).mean()
