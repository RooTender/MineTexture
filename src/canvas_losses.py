import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def l1_rgba_weighted(pred, target, canvas_mask, w_alpha: float = 0.25):
    # pred/target: [B,4,H,W], canvas_mask: [B,1,H,W] (1 tam gdzie jest tekstura celu)
    pr, pa = pred[:, :3], pred[:, 3:4]
    tr, ta = target[:, :3], target[:, 3:4]

    # RGB liczymy tam, gdzie cel jest widoczny (ważenie alfą celu)
    m_rgb = (canvas_mask * ta).repeat(1, 3, 1, 1)  # [B,3,H,W]
    m_a   = canvas_mask                             # [B,1,H,W]

    num_rgb = m_rgb.sum(dim=(1,2,3)).clamp_min(1e-6)
    num_a   = m_a.sum(dim=(1,2,3)).clamp_min(1e-6)

    l1_rgb = ((pr - tr).abs() * m_rgb).sum(dim=(1,2,3)).div(num_rgb).mean()
    l1_a   = ((pa - ta).abs() * m_a).sum(dim=(1,2,3)).div(num_a).mean()

    return l1_rgb + w_alpha * l1_a, l1_rgb, l1_a

@torch.jit.script
def sobel_filter(x: torch.Tensor) -> torch.Tensor:
    gx = torch.tensor([[1.,0.,-1.],
                       [2.,0.,-2.],
                       [1.,0.,-1.]], device=x.device).view(1,1,3,3)
    gy = torch.tensor([[1.,2.,1.],
                       [0.,0.,0.],
                       [-1.,-2.,-1.]], device=x.device).view(1,1,3,3)
    C = x.size(1)
    gradx = F.conv2d(x, gx.expand(C,1,3,3), padding=1, groups=C)
    grady = F.conv2d(x, gy.expand(C,1,3,3), padding=1, groups=C)
    return torch.sqrt(gradx * gradx + grady * grady + 1e-6)

@torch.jit.script
def edge_mask_from_alpha(a: torch.Tensor, thresh: float = 0.05, dilate: int = 1) -> torch.Tensor:
    # a: [B,1,H,W]
    g = sobel_filter(a.repeat(1,3,1,1))[:, :1]     # [B,1,H,W]
    m = (g > thresh).to(a.dtype)

    if dilate > 0:
        k = torch.ones((1,1,3,3), device=a.device, dtype=a.dtype)
        i: int = 0
        while i < dilate:
            m = (F.conv2d(m, k, padding=1) > 0).to(a.dtype)
            i += 1
    return m

@torch.jit.script
def edge_loss(pred: torch.Tensor, target: torch.Tensor, canvas_mask: torch.Tensor) -> torch.Tensor:
    pr, tr = pred[:, :3], target[:, :3]
    gp = sobel_filter(pr)
    gt = sobel_filter(tr)
    edge_m = edge_mask_from_alpha(target[:, 3:4], 0.02, 1)
    w = torch.clamp(canvas_mask + 2.0 * edge_m, 0.0, 3.0)
    num = w.sum().clamp_min(1e-6)
    return ((gp - gt).abs() * w.repeat(1,3,1,1)).sum() / num
