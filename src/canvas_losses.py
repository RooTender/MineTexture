
import torch
import torch.nn.functional as F

def masked_l1(pred, target, mask):
    num = mask.sum(dim=(1,2,3)).clamp_min(1e-6)        # [B]
    diff = (pred - target).abs() * mask
    return (diff.sum(dim=(1,2,3)) / num).mean()

def masked_ssim(pred, target, mask, C1=0.01**2, C2=0.03**2, win=3):
    assert win % 2 == 1
    pad = win // 2

    x = pred[:, :3]    # [B,3,H,W]
    y = target[:, :3]  # [B,3,H,W]

    x = F.pad(x, (pad,pad,pad,pad), mode='circular')
    y = F.pad(y, (pad,pad,pad,pad), mode='circular')

    B, C, H, W = x.shape
    k = torch.ones((C,1,win,win), device=x.device, dtype=x.dtype) / (win*win)

    def mean(u):  return F.conv2d(u, k, groups=C)
    mu_x, mu_y = mean(x), mean(y)
    sigma_x  = mean(x*x) - mu_x*mu_x
    sigma_y  = mean(y*y) - mu_y*mu_y
    sigma_xy = mean(x*y) - mu_x*mu_y

    ssim = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x*mu_x + mu_y*mu_y + C1)*(sigma_x + sigma_y + C2))
    # ssim ma rozmiar [B,3,H,W] — dopasuj maskę
    m = mask[..., :H, :W].repeat(1, 3, 1, 1)
    num = m.sum(dim=(1,2,3)).clamp_min(1e-6)
    return ((1.0 - ssim) * m).sum(dim=(1,2,3)).div(num).mean()


def seam_loss(pred, mask):
    x = pred[:, :3]
    diff_x = (x - torch.roll(x, shifts=1, dims=-1)).abs()
    diff_y = (x - torch.roll(x, shifts=1, dims=-2)).abs()
    m_x = mask * torch.roll(mask, shifts=1, dims=-1)
    m_y = mask * torch.roll(mask, shifts=1, dims=-2)
    num = (m_x.sum(dim=(1,2,3)) + m_y.sum(dim=(1,2,3))).clamp_min(1e-6)
    val = (diff_x * m_x).sum(dim=(1,2,3)) + (diff_y * m_y).sum(dim=(1,2,3))
    return (val / num).mean()
