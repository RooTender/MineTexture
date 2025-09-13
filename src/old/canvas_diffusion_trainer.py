from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random

BASE_MAX_DIM = 16

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
        self.styled_scale = styled_scale

        self.pairs: List[Tuple[Path, Path]] = []

        files = list(self.root_v.rglob("*"))
        for p_v in tqdm(files, desc="Loading textures..."):
            if not p_v.is_file():
                continue

            with Image.open(p_v) as img:
                width, height = img.size

                if width > BASE_MAX_DIM or height > BASE_MAX_DIM:
                    continue

            rel = p_v.relative_to(self.root_v)
            p_s = self.root_s / rel

            if p_s.is_file():
                with Image.open(p_s) as img:
                    width, height = img.size

                    if width > BASE_MAX_DIM * styled_scale or height > BASE_MAX_DIM * styled_scale:
                        continue

                self.pairs.append((p_v, p_s))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        p_v, p_s = self.pairs[idx]

        def to_rgba_and_scale(path: str, scale: int = 1) -> torch.Tensor:
            with Image.open(path) as img:
                if img.mode != "RGBA":
                    img = img.convert("RGBA")

                if scale != 1:
                    w, h = img.size
                    img = img.resize((w * scale, h * scale), resample=Image.Resampling.NEAREST)

                arr = np.array(img, dtype=np.uint8)
            t = torch.from_numpy(arr).permute(2,0,1)

            return t.to(torch.float32) / 255.0

        tv = to_rgba_and_scale(p_v, self.styled_scale)
        ts = to_rgba_and_scale(p_s)

        def to_canvas_and_mask(x: torch.Tensor, canvas: int, ox: Optional[int]=None, oy: Optional[int]=None):
            _, h, w = x.shape
            y = torch.zeros((4, canvas, canvas), dtype=x.dtype)
            m = torch.zeros((1, canvas, canvas), dtype=x.dtype)

            max_y = canvas - h
            max_x = canvas - w
            if oy is None: oy = random.randint(0, max_y) if max_y > 0 else 0
            if ox is None: ox = random.randint(0, max_x) if max_x > 0 else 0

            y[:, oy:oy+h, ox:ox+w] = x
            m[:, oy:oy+h, ox:ox+w] = 1.0
            return y, m, ox, oy

        tv_pad, mv, ox, oy = to_canvas_and_mask(tv, BASE_MAX_DIM * self.styled_scale)
        ts_pad, ms, _, _   = to_canvas_and_mask(ts, BASE_MAX_DIM * self.styled_scale, ox=ox, oy=oy)

        return {
            "vanilla": tv_pad,
            "mask_v": mv,
            "styled": ts_pad,
            "mask_s": ms
        }

from canvas_diffusion import DiffUNet, ddim_infer, default_sigmas, from_n11, to_n11
from canvas_losses import l1_rgba_weighted, edge_loss
from torchvision.utils import make_grid

import torch, os
from torch.utils.data import DataLoader, random_split

import wandb

LR = 3e-4
WEIGHT_DECAY = 2e-4
BASE = 32
EPOCHS = 50
BATCH_SIZE = 64

DIFF_STEPS      = 3      # 3 albo 4
P_SELFCOND      = 0.5    # prawdopodobieństwo self-conditioning
P_CFG_DROPOUT   = 0.10   # ile razy uczymy "uncond"
AUX_LOSS_WEIGHT = 0.25   # waga Twoich L1/edge dla małych sigma

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

model = DiffUNet(in_ch=4, base=BASE, out_ch=4).to(device, memory_format=torch.channels_last)

opt = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.99), eps=1e-8, fused=True
)

from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    opt,
    max_lr = LR * 2,                # spróbuj 2–3x bazowego
    epochs = EPOCHS,
    steps_per_epoch = len(train_dl),
    pct_start = 0.1,                # 20% czasu na wzrost
    anneal_strategy = 'cos',
    div_factor = 10,                # start_lr = max_lr / 10
    final_div_factor = 20           # final_lr = max_lr / 20
)

def to_device_channels_last(x):
    return x.to(device, non_blocking=True).to(memory_format=torch.channels_last)

def calc_losses(pred_img, target, mask):
    base_total, l1_rgb, l1_a = l1_rgba_weighted(pred_img, target, mask)
    le_main = edge_loss(pred_img, target, mask)  # Twój dotychczasowy

    loss = base_total + 0.5*le_main

    return {
        "loss": loss,
        "edge": le_main,
        "l1_rgb": l1_rgb,
        "l1_a": l1_a,
    }

def train_step(batch):
    model.train()

    v01 = to_device_channels_last(batch["vanilla"])
    s01 = to_device_channels_last(batch["styled"])
    ms  = to_device_channels_last(batch["mask_s"])
    v   = to_n11(v01)
    x0  = to_n11(s01)

    B = v.size(0); device = v.device

    def sample_sigma(B, sigma_min=0.01, sigma_max=0.7, device=None):
        u = torch.rand(B, device=device)
        return sigma_min * (sigma_max / sigma_min) ** u

    sigma = sample_sigma(B, 0.01, 0.7, device)
    sb = sigma.view(B,1,1,1)

    eps_true = torch.randn_like(x0)
    x_t = x0 + sb * eps_true

    # --- self-conditioning: 50% teacher-forcing, 50% puste ---
    use_tf = torch.rand(()) < P_SELFCOND  # P_SELFCOND = 0.5
    if use_tf:
        with torch.no_grad():
            eps_tf = model(v, x_t, sigma)          # ε_tf
            x0_sc  = (x_t - sb * eps_tf).detach()  # x0_hat (teacher forcing)
    else:
        x0_sc = torch.zeros_like(x_t)

    # CFG dropout (uncond)
    v_in = torch.zeros_like(v) if (torch.rand(()) < P_CFG_DROPOUT) else v

    eps_hat = model(v_in, x_t, sigma, x0_sc)
    x0_hat = x_t - sb * eps_hat
    x0_hat01 = from_n11(x0_hat).clamp(0,1)

    # --- główny loss: na x0 (L1 + edge) z maską ---
    main = calc_losses(x0_hat01, s01, ms)["loss"]

    # (opcjonalny) mały składnik na ε – pomaga uśredniać szum
    eps_mse = torch.mean((eps_hat - eps_true) ** 2)

    loss = main + 0.1 * eps_mse  # 0.1 zwykle wystarcza

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stabilizator
    opt.step()
    scheduler.step()

    return {"loss": float(loss.item()), "x0_main": float(main.item()), "eps_mse": float(eps_mse.item())}


@torch.no_grad()
def val_step_like_train(batch):
    model.eval()
    v01 = to_device_channels_last(batch["vanilla"])
    s01 = to_device_channels_last(batch["styled"])
    ms  = to_device_channels_last(batch["mask_s"])

    v   = to_n11(v01)
    x0  = to_n11(s01)
    B   = v.size(0); device = v.device

    # losuj sigma z tego samego rozkładu co w train
    def sample_sigma(B, sigma_min=0.01, sigma_max=0.7, device=None):
        u = torch.rand(B, device=device)
        return sigma_min * (sigma_max / sigma_min) ** u
    sigma = sample_sigma(B, 0.01, 0.7, device)
    sb = sigma.view(B,1,1,1)

    eps_true = torch.randn_like(x0)
    x_t = x0 + sb * eps_true

    # bez TF i CFG-drop (czysta ewaluacja)
    eps_hat = model(v, x_t, sigma, torch.zeros_like(x_t))
    x0_hat01 = from_n11(x_t - sb * eps_hat).clamp(0,1)

    losses = calc_losses(x0_hat01, s01, ms)
    return {"loss": float(losses["loss"].item())}



@torch.no_grad()
def val_inference(batch, preview = False):
    model.eval()
    v01 = to_device_channels_last(batch["vanilla"])
    s01 = to_device_channels_last(batch["styled"])
    ms  = to_device_channels_last(batch["mask_s"])

    # generacja w 3 krokach (szybko) + delikatny guidance
    pred01 = ddim_infer(model, v01, steps=DIFF_STEPS, cfg_scale=1.5, use_sc=True)

    losses = calc_losses(pred01, s01, ms)

    sample = None
    if preview:
        k = 4
        grid = torch.cat([v01[:k], s01[:k], pred01[:k]], dim=0)
        grid = make_grid(grid, nrow=k)
        sample = wandb.Image(grid, caption="(vanilla | styled | pred)")

    return { "loss": float(losses['loss'].item()) }, sample


@torch.inference_mode()
def val_step(val_dl, epoch):
    gen_acc, x0_acc = {}, {}
    n = 0
    sample_img = None
    first = True

    for batch in tqdm(val_dl, desc=f"Validating ({epoch+1}/{EPOCHS})"):
        # 1) Generation (DDIM)
        gen_metrics, sample = val_inference(batch, preview=first)
        if sample_img is None: sample_img = sample

        # 2) Train-like (single step, jak w train)
        x0_metrics = val_step_like_train(batch)

        bs = batch["vanilla"].size(0)
        def add(acc, d): 
            for k,v in d.items(): acc[k] = acc.get(k,0.0) + float(v)*bs

        add(gen_acc, {"loss": gen_metrics["loss"]})
        add(x0_acc,  {"loss": x0_metrics["loss"]})
        n += bs
        first = False

    gen_mean = {k: v/max(1,n) for k,v in gen_acc.items()}
    x0_mean  = {k: v/max(1,n) for k,v in x0_acc.items()}
    return gen_mean, x0_mean, sample_img


outdir = "runs/quickcheck"

wandb.init(
    project="minecraft-textures",  # nazwa projektu
    config={
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight-decay": WEIGHT_DECAY,
        "base": BASE,
        "epochs": EPOCHS,
        "model": "DiffUNet-miniDDIM",
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
    model.train()
    train_acc, n = {}, 0

    for batch in tqdm(train_dl, desc=f"Training ({epoch+1}/{EPOCHS})"):
        m = train_step(batch)
        bs = batch["vanilla"].size(0); n += bs
        for k,v in m.items(): train_acc[k] = train_acc.get(k,0.0) + float(v)*bs
    train_mean = {k: v/max(1,n) for k,v in train_acc.items()}

    model.eval()
    gen_mean, x0_mean, sample_img = val_step(val_dl, epoch)

    metrics_summary = {}
    metrics_summary.update({f"train/{k}": v for k,v in train_mean.items()})
    metrics_summary.update({f"valid_gen/{k}": v for k,v in gen_mean.items()})
    metrics_summary.update({f"valid_x0/{k}":  v for k,v in x0_mean.items()})
    metrics_summary["lr"] = scheduler.get_last_lr()[0]
    metrics_summary["valid/sample"] = sample_img

    wandb.log(metrics_summary)

    if metrics_summary["valid_x0/loss"] < best_val:
        best_val = metrics_summary["valid_x0/loss"]

        print(f"Saving model on epoch {epoch + 1} with loss: {best_val}")
        os.makedirs("checkpoints", exist_ok=True)

        ckpt_path = f"checkpoints/best_epoch_{epoch}.pt"
        torch.save({"model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "epoch": epoch}, ckpt_path)

    wandb.log(metrics_summary)
