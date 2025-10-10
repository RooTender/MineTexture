from pathlib import Path
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from loader import TexturePairDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from model import TinyUNet
from losses import EdgeLoss, weighted_l1
from tqdm import tqdm
import wandb

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.conv.fp32_precision = "tf32" # type: ignore
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.set_float32_matmul_precision("high")

WATCH_NAMES = ['crossbow_pulling_0.pt', 'kelp_18.pt', 'raw_copper_block.pt', 'comparator.pt']

EPOCHS        = 100
BATCH_SIZE    = 256
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-4

DEVICE = 'cuda'


train_ds = TexturePairDataset(
    path_a=Path("data/augmented_pt/train/vanilla"),
    path_b=Path("data/augmented_pt/train/styled"),
    scale=2,
)
valid_ds = TexturePairDataset(
    path_a=Path("data/augmented_pt/valid/vanilla"),
    path_b=Path("data/augmented_pt/valid/styled"),
    scale=2,
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,drop_last=True,
                          num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False,drop_last=True,
                          num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)


model = TinyUNet().to(DEVICE).to(memory_format=torch.channels_last)
# model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                  betas=(0.9, 0.999), fused=True)


l1_criterion = nn.L1Loss()
edge_criterion = EdgeLoss().to(DEVICE)


def compute_losses(pred, target):
    l1            = l1_criterion(pred, target)
    edge          = edge_criterion(pred, target)
    weighted_loss = weighted_l1(pred, target, edge_criterion)

    total = l1 + edge + weighted_loss
    return total, l1, edge, weighted_loss


wandb.init(
    project="MineTexture",
    config={
        "model": "TinyUNet",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    },
)


for epoch in range(1, EPOCHS + 1):
    train_tot = torch.zeros((), device=DEVICE)
    train_l1  = torch.zeros((), device=DEVICE)
    train_edge= torch.zeros((), device=DEVICE)
    train_w1  = torch.zeros((), device=DEVICE)

    valid_tot = torch.zeros((), device=DEVICE)
    valid_l1  = torch.zeros((), device=DEVICE)
    valid_edge= torch.zeros((), device=DEVICE)
    valid_w1  = torch.zeros((), device=DEVICE)
    watched = {}

    model.train()

    for batch in tqdm(train_loader, desc=f"Training ({epoch}/{EPOCHS})"):
        _, A, B = batch
        A = A.pin_memory().to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        B = B.pin_memory().to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        # with torch.autocast(device_type=DEVICE, dtype=torch.float16):
        A = A.float().mul_(1/255.0)
        B = B.float().mul_(1/255.0)

        pred = model(A)
        total, l1, edge, w1 = compute_losses(pred, B)

        total.backward()
        optimizer.step()

        bs = A.size(0)
        train_tot  += total.detach() * bs
        train_l1   += l1.detach()    * bs
        train_edge += edge.detach()  * bs
        train_w1   += w1.detach()    * bs
    
    model.eval()

    with torch.inference_mode():
        for batch in tqdm(valid_loader, desc=f"Validating ({epoch}/{EPOCHS})"):
            names, A, B = batch
            A = A.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
            B = B.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)

            # with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            A = A.float().mul_(1/255.0)
            B = B.float().mul_(1/255.0)

            pred = model(A).clamp(0, 1)
            total, l1, edge, w1 = compute_losses(pred, B)

            bs = A.size(0)
            valid_tot  += total.detach() * bs
            valid_l1   += l1.detach()    * bs
            valid_edge += edge.detach()  * bs
            valid_w1   += w1.detach()    * bs

            for i, name in enumerate(names):
                if name in WATCH_NAMES and name not in watched:
                    watched[name] = {
                        "input":  A[i].detach().float(),
                        "target": B[i].detach().float(),
                        "pred":   pred[i].detach().float(),
                    }

    A_cat = torch.cat([watched[n]["input"].unsqueeze(0)  for n in WATCH_NAMES], dim=0)
    B_cat = torch.cat([watched[n]["target"].unsqueeze(0) for n in WATCH_NAMES], dim=0)
    P_cat = torch.cat([watched[n]["pred"].unsqueeze(0)   for n in WATCH_NAMES], dim=0)

    samples = make_grid(torch.cat([A_cat, B_cat, P_cat], dim=0), nrow=len(WATCH_NAMES))

    train_sz = len(train_ds)
    valid_sz = len(valid_ds)
    wandb.log({
        "train":        (train_tot / train_sz).item(),
        "train/l1":     (train_l1  / train_sz).item(),
        "train/edge":   (train_edge/ train_sz).item(),
        "train/w1":     (train_w1  / train_sz).item(),
        "valid":        (valid_tot / valid_sz).item(),
        "valid/l1":     (valid_l1  / valid_sz).item(),
        "valid/edge":   (valid_edge/ valid_sz).item(),
        "valid/w1":     (valid_w1  / valid_sz).item(),
        "sample":       wandb.Image(samples, caption="(vanilla | styled | pred)"),
    })
