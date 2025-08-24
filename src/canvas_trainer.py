from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

CANVAS_SIZE = 16

class TexturePairDataset(Dataset):
    """
    Paruje pliki vanilla <-> styled po identycznej ścieżce relatywnej.
    Jeśli styled/<rel_path> nie istnieje -> pomijamy.
    """
    def __init__(
        self,
        root_vanilla: str,
        root_styled: str,
        styled_scale: int = 1
    ):
        self.root_v = Path(root_vanilla)
        self.root_s = Path(root_styled)

        if CANVAS_SIZE * styled_scale > CANVAS_SIZE:
            self.canvas = CANVAS_SIZE * styled_scale
        else:
            self.canvas = CANVAS_SIZE

        self.pairs: List[Tuple[Path, Path]] = []

        files = list(self.root_v.rglob("*"))
        for p_v in tqdm(files, desc="Loading textures..."):
            if not p_v.is_file():
                continue

            with Image.open(p_v) as img:
                width, height = img.size

                if width > self.canvas or height > self.canvas:
                    continue

            rel = p_v.relative_to(self.root_v)
            p_s = self.root_s / rel

            if p_s.is_file():
                with Image.open(p_s) as img:
                    width, height = img.size

                    if width > self.canvas or height > self.canvas:
                        continue

                self.pairs.append((p_v, p_s))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        p_v, p_s = self.pairs[idx]

        def to_rgba(path: str) -> torch.Tensor:
            with Image.open(path) as img:
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                arr = np.array(img, dtype=np.uint8)       # [H,W,4]
            t = torch.from_numpy(arr).permute(2,0,1)  # [4,H,W]

            return t.to(torch.float32) / 255.0

        tv = to_rgba(p_v)
        ts = to_rgba(p_s)

        def to_canvas_and_mask(
                x: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
            _, h, w = x.shape

            x_pad = torch.zeros((4, self.canvas, self.canvas), dtype=x.dtype)
            x_pad[:, :h, :w] = x

            mask = torch.zeros((1, self.canvas, self.canvas), dtype=x.dtype)
            mask[:, :h, :w] = 1.0

            return x_pad, mask

        tv_pad, mv = to_canvas_and_mask(tv)
        ts_pad, ms = to_canvas_and_mask(ts)

        return {
            "vanilla": tv_pad,
            "mask_v": mv,
            "styled": ts_pad,
            "mask_s": ms
        }

from canvas_model import TinyUNet
from canvas_losses import masked_l1, masked_ssim, seam_loss
from torchvision.utils import make_grid

import torch, os
from torch.utils.data import DataLoader, random_split

import wandb

LR = 3e-4
WEIGHT_DECAY = 1e-4
BASE = 32
EPOCHS = 50
BATCH_SIZE = 64

device = "cuda"

ds = TexturePairDataset(
    root_vanilla="data/augmented/vanilla",
    root_styled="data/augmented/styled",
    styled_scale=2,
)

n_val = max(1, int(len(ds) * 0.2))
n_train = len(ds) - n_val
train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

train_dl = DataLoader(train_ds, 
                      batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True, 
                      persistent_workers=True, pin_memory_device="cuda", prefetch_factor=2)

val_dl   = DataLoader(val_ds,   
                      batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, 
                      persistent_workers=True, pin_memory_device="cuda", prefetch_factor=2)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

model = TinyUNet(in_ch=4, base=BASE, out_ch=4).to(device, memory_format=torch.channels_last)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def to_device_channels_last(x):
    return x.to(device, non_blocking=True).to(memory_format=torch.channels_last)

def calc_losses(pred, target, mask):
    l1   = masked_l1(pred, target, mask)
    ssim = masked_ssim(pred, target, mask)
    seam = seam_loss(pred, mask)

    loss = 0.6*l1 + 0.35*ssim + 0.05*seam

    return {
        "loss": loss,
        "l1":   l1,
        "ssim": ssim,
        "seam": seam,
    }

def train_step(batch):
    model.train()
    v = to_device_channels_last(batch["vanilla"])
    s = to_device_channels_last(batch["styled"])
    ms = to_device_channels_last(batch["mask_s"])

    pred = model(v)
    losses = calc_losses(pred, s, ms)

    opt.zero_grad(set_to_none=True)
    losses["loss"].backward()
    opt.step() 
    
    return { 
        "loss": float(losses['loss'].item()), 
        "l1": float(losses['l1'].item()), 
        "ssim": float(losses['ssim'].item()), 
        "seam": float(losses['seam'].item()), 
    }

@torch.no_grad()
def val_step(batch, preview = False):
    model.eval()
    v  = to_device_channels_last(batch["vanilla"])
    s  = to_device_channels_last(batch["styled"])
    ms = to_device_channels_last(batch["mask_s"])

    pred = model(v).clamp(0, 1)
    losses = calc_losses(pred, s, ms)

    sample = None

    if preview:
        k = 4
        grid = torch.cat([v[:k].float(), s[:k].float(), pred[:k].float()], dim=0)
        grid = make_grid(grid, nrow=k)
        
        sample = wandb.Image(grid, caption="(vanilla | styled | pred)")

    return {
        "loss": float(losses['loss'].item()),
        "l1":   float(losses['l1'].item()),
        "ssim": float(losses['ssim'].item()),
        "seam": float(losses['seam'].item()),
    }, sample

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

    metrics_summary.update({"valid/sample": sample_img})

    if metrics_summary["valid/loss"] < best_val:
        best_val = metrics_summary["valid/loss"]
        print(f"Saving model on epoch {epoch + 1} with loss: {best_val}")
        os.makedirs("checkpoints", exist_ok=True)

        ckpt_path = f"checkpoints/best_epoch_{epoch}.pt"
        torch.save({"model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": epoch}, ckpt_path)

    wandb.log(metrics_summary)
    # prosty decay LR co epokę (opcjonalnie)
    # for g in opt.param_groups:
    #     g["lr"] *= 0.98

    # validate(model, batch, outdir, step)