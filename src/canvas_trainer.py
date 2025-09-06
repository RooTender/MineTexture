from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class TexturePairs(Dataset):
    def __init__(self, root_vanilla: str, root_styled: str):
        self.v_root = Path(root_vanilla)
        self.s_root = Path(root_styled)

        # Zindeksuj styled: rel_ścieżka (względem bucketa) -> pełna ścieżka
        self.styled_map = {}
        for s_bucket in sorted(self.s_root.glob("x*")):
            if not s_bucket.is_dir():
                continue
            for p_s in s_bucket.rglob("*"):
                if p_s.is_file():
                    rel = p_s.relative_to(s_bucket)
                    self.styled_map[str(rel).replace("\\", "/")] = p_s

        # Zbierz pary (vanilla, styled) po tej samej rel_ścieżce
        self.pairs: List[Tuple[Path, Path]] = []
        for v_bucket in sorted(self.v_root.glob("x*")):
            if not v_bucket.is_dir():
                continue
            for p_v in v_bucket.rglob("*"):
                if not p_v.is_file():
                    continue
                rel = str(p_v.relative_to(v_bucket)).replace("\\", "/")
                p_s = self.styled_map.get(rel, None)
                if p_s is not None:
                    self.pairs.append((p_v, p_s))

        self.pairs.sort(key=lambda t: (str(t[0]).lower(), str(t[1]).lower()))

    @staticmethod
    def _load_rgba(path: Path) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGBA")

    @staticmethod
    def _to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.uint8)           # H W 4
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        p_v, p_s = self.pairs[idx]
        img_v = self._load_rgba(p_v)
        img_s = self._load_rgba(p_s)

        # Zawsze dopasuj vanilla do rozmiaru styled (piksel-art safe).
        if img_v.size != img_s.size:
            img_v = img_v.resize(img_s.size, resample=Image.Resampling.NEAREST)

        return {
            "vanilla": self._to_tensor(img_v),
            "styled":  self._to_tensor(img_s),
        }

train_pairs = TexturePairs(
    root_vanilla="data/augmented/train/vanilla",
    root_styled="data/augmented/train/styled",
)

valid_pairs = TexturePairs(
    root_vanilla="data/augmented/valid/vanilla",
    root_styled="data/augmented/valid/styled",
)

from canvas_model import TinyUNetPlus
from canvas_losses import l1_rgba_weighted, edge_loss, isolated_alpha_loss
from torchvision.utils import make_grid

import torch, os
from torch.utils.data import DataLoader, random_split

import wandb

LR = 5e-4
WEIGHT_DECAY = 1e-4
BASE = 32
EPOCHS = 250
BATCH_SIZE = 256

device = "cuda"



n_val = max(1, int(len(ds) * 0.2))
n_train = len(ds) - n_val
train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

train_dl = DataLoader(train_ds, 
                      batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True, 
                      persistent_workers=True, pin_memory_device="cuda", prefetch_factor=4)

val_dl   = DataLoader(val_ds,   
                      batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, 
                      persistent_workers=True, pin_memory_device="cuda", prefetch_factor=4)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

use_bf16 = torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

model = TinyUNetPlus(in_ch=4, base=BASE, out_ch=4).to(device, memory_format=torch.channels_last)

compile_mode = "reduce-overhead"
model = torch.compile(model, mode=compile_mode, fullgraph=False, dynamic=False)

opt = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.99), eps=1e-8, fused=True
)

def to_device_channels_last(x):
    return x.to(device, non_blocking=True).to(memory_format=torch.channels_last)

import torch.nn.functional as F

def calc_losses(pred_img, vanilla, target, mask):
    base_total, l1_rgb, l1_a = l1_rgba_weighted(pred_img, target, mask)

    def change_focus_weight(v, s, gamma, eps=1e-4):
        diff_rgb = torch.abs(v[:,:3] - s[:,:3]).mean(dim=1, keepdim=True)
        diff_a   = torch.abs(v[:, 3:4] - s[:, 3:4])
        diff     = (diff_rgb + diff_a) * 0.5

        w = (diff + eps) ** gamma
        w = w / (w.mean(dim=[1,2,3], keepdim=True) + 1e-6)
        return w
    
    def smooth_w(w, k=3):
        return F.avg_pool2d(w, kernel_size=k, stride=1, padding=k//2)

    w_focus = change_focus_weight(vanilla, target, gamma=1.0)
    w_focus = smooth_w(w_focus, k=3)
    l1_focal = (w_focus * mask * torch.abs(pred_img - target)).mean()

    island_loss = isolated_alpha_loss(pred_img, mask)

    e_loss = edge_loss(pred_img, target, mask)
    loss = base_total + 0.5 * l1_focal + 0.15 * e_loss + 0.10 * island_loss

    return {
        "loss": loss,
        "edge": e_loss,
        "focal": l1_focal,
        "island": island_loss,
        "l1_rgb": l1_rgb,
        "l1_a": l1_a,
    }

def train_step(batch):
    model.train()
    v = to_device_channels_last(batch["vanilla"])
    s = to_device_channels_last(batch["styled"])
    ms = to_device_channels_last(batch["mask_s"])

    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
        pred = model(v)
        losses = calc_losses(pred, v, s, ms)

    losses["loss"].backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    
    return {k: float(v.item() if torch.is_tensor(v) else v) for k, v in losses.items()}

@torch.no_grad()
def val_step(batch, preview = False):
    model.eval()
    v  = to_device_channels_last(batch["vanilla"])
    s  = to_device_channels_last(batch["styled"])
    ms = to_device_channels_last(batch["mask_s"])

    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
        pred = (model(v)).clamp(0, 1)
        losses = calc_losses(pred, v, s, ms)

    sample = None
    if preview:
        k = 4
        grid = torch.cat([v[:k].float(), s[:k].float(), pred[:k].float()], dim=0)
        grid = make_grid(grid, nrow=k)
        
        sample = wandb.Image(grid, caption="(vanilla | styled | pred)")

    return {k: float(v.item() if torch.is_tensor(v) else v) for k, v in losses.items()}, sample

outdir = "runs/quickcheck"

wandb.init(
    project="minecraft-textures",  # nazwa projektu
    config={
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight-decay": WEIGHT_DECAY,
        "base": BASE,
        "epochs": EPOCHS,
        "model": "TinyUNet",
    }
)

def _tofloat(x): return float(x.item()) if torch.is_tensor(x) else float(x)

def reduce_add(acc, new, weight):
    for k, v in new.items():
        acc[k] = acc.get(k, 0.0) + _tofloat(v) * weight
    return acc

def reduce_mean(sums, denom):
    return {k: v / max(1, denom) for k, v in sums.items()}

best_val = float("inf")

for epoch in range(EPOCHS):
    metrics_summary = {}

    metrics_acc = {}
    batches_sum = 0

    for batch in tqdm(train_dl, desc=f"Training ({epoch+1}/{EPOCHS})"):
        metrics = train_step(batch)

        batch_size = batch["vanilla"].size(0)
        metrics_acc = reduce_add(metrics_acc, metrics, batch_size)

        batches_sum += batch_size

    metrics_summary.update({**{
        f"train/{k}": v for k, v in reduce_mean(metrics_acc, batches_sum).items()
    }})

    metrics_acc = {}
    batches_sum = 0

    first_step = True
    sample_img = None

    for batch in tqdm(val_dl, desc=f"Validating ({epoch+1}/{EPOCHS})"):
        metrics, sample = val_step(batch, first_step)
        first_step = False

        if sample_img is None:
            sample_img = sample

        batch_size = batch["vanilla"].size(0)
        metrics_acc = reduce_add(metrics_acc, metrics, batch_size)

        batches_sum += batch_size

    metrics_summary.update({**{
        f"valid/{k}": v for k, v in reduce_mean(metrics_acc, batches_sum).items()
    }})

    metrics_summary.update({
        "lr": LR # opt.param_groups[0]["lr"]
    })

    metrics_summary.update({"valid/sample": sample_img})

    if metrics_summary["valid/loss"] < best_val:
        best_val = metrics_summary["valid/loss"]
        print(f"Saving model on epoch {epoch + 1} with loss: {best_val}")
        os.makedirs("checkpoints", exist_ok=True)

        def unwrap_state_dict(m: torch.nn.Module):
            return m._orig_mod.state_dict() if hasattr(m, "_orig_mod") else m.state_dict()

        ckpt_path = f"checkpoints/best_epoch_{epoch}.pt"
        torch.save({
            "model": unwrap_state_dict(model),
            "opt": opt.state_dict(),
            "epoch": epoch
            }, ckpt_path
        )

    wandb.log(metrics_summary)
