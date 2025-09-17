from pathlib import Path
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from loader import TexturePairDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from model import TinyUNet
from edge_loss import EdgeLoss
from tqdm import tqdm
import wandb

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

WATCH_NAMES = ['bow.png', 'seagrass_0.png', 'raw_copper_block.png', 'comparator.png']

EPOCHS        = 100
BATCH_SIZE    = 256
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-4

DEVICE = 'cuda'


train_ds = TexturePairDataset(
    path_a=Path("data/augmented/train/vanilla"),
    path_b=Path("data/augmented/train/styled"),
    scale=2,
)
valid_ds = TexturePairDataset(
    path_a=Path("data/augmented/valid/vanilla"),
    path_b=Path("data/augmented/valid/styled"),
    scale=2,
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                          num_workers=8, pin_memory=True, pin_memory_device=DEVICE,
                          persistent_workers=True, prefetch_factor=8)

valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True,
                          num_workers=4, pin_memory=True,  pin_memory_device=DEVICE,
                          persistent_workers=True, prefetch_factor=4)


model = TinyUNet().to(DEVICE).to(memory_format=torch.channels_last)
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                  betas=(0.9, 0.999), fused=True)


l1_criterion = nn.L1Loss()
edge_criterion = EdgeLoss().to(DEVICE)

def compute_losses(pred, target):
    l1   = l1_criterion(pred, target)
    edge = edge_criterion(pred, target)
    total = l1 + 0.1 * edge
    return total, l1, edge


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

    valid_tot = torch.zeros((), device=DEVICE)
    valid_l1  = torch.zeros((), device=DEVICE)
    valid_edge= torch.zeros((), device=DEVICE)
    watched = {}

    model.train()

    for batch in tqdm(train_loader, desc=f"Training ({epoch}/{EPOCHS})"):
        A = batch["input"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        B = batch["target"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            pred = model(A)
            total, l1, edge = compute_losses(pred, B)

        total.backward()
        optimizer.step()

        bs = A.size(0)
        train_tot  += total.detach() * bs
        train_l1   += l1.detach()    * bs
        train_edge += edge.detach()  * bs
    
    model.eval()

    with torch.inference_mode():
        for batch in tqdm(valid_loader, desc=f"Validating ({epoch}/{EPOCHS})"):
            A = batch["input"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
            B = batch["target"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)

            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                pred = model(A).clamp(0, 1)
                total, l1, edge = compute_losses(pred, B)

            bs = A.size(0)
            valid_tot  += total.detach() * bs
            valid_l1   += l1.detach()    * bs
            valid_edge += edge.detach()  * bs

            for i, name in enumerate(batch["basename"]):
                if name in WATCH_NAMES and name not in watched:
                    watched[name] = {
                        "input": A[i].detach().to("cpu", non_blocking=True).float(),
                        "target": B[i].detach().to("cpu", non_blocking=True).float(),
                        "pred":   pred[i].detach().to("cpu", non_blocking=True).float(),
                    }

    samples = []
    for key in ("input", "target", "pred"):
        for name in WATCH_NAMES:
            samples.append(watched[name][key])

    grid = torch.stack(samples, dim=0)
    grid = make_grid(grid, nrow=len(WATCH_NAMES))

    train_sz = len(train_ds)
    valid_sz = len(valid_ds)
    wandb.log({
        "train":        (train_tot / train_sz).item(),
        "train/l1":     (train_l1  / train_sz).item(),
        "train/edge":   (train_edge/ train_sz).item(),
        "valid":        (valid_tot / valid_sz).item(),
        "valid/l1":     (valid_l1  / valid_sz).item(),
        "valid/edge":   (valid_edge/ valid_sz).item(),
        "sample": wandb.Image(grid, caption="(vanilla | styled | pred)")
    })
