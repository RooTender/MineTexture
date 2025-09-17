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

EPOCHS        = 1000
BATCH_SIZE    = 256
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 0 #1e-5

DEVICE = 'cuda'


train_ds = TexturePairDataset(
    path_a=Path("data/split/train/vanilla"),
    path_b=Path("data/split/train/styled"),
    scale=2,
)
tiny_ds = torch.utils.data.Subset(train_ds, list(range(BATCH_SIZE)))

train_loader = DataLoader(tiny_ds, batch_size=BATCH_SIZE,
                          pin_memory=True, pin_memory_device=DEVICE)

model = TinyUNet().to(DEVICE).to(memory_format=torch.channels_last)
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, 
                  betas=(0.9, 0.999), fused=True)


l1_criterion = nn.L1Loss()
edge_criterion = EdgeLoss().to(DEVICE)


wandb.init(
    project="MineTexture",
    name="clip_skip_16_8",
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
    grid = None

    model.train()

    for batch in tqdm(train_loader, desc=f"Training ({epoch}/{EPOCHS})"):
        A = batch["input"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        B = batch["target"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            pred  = model(A)
            l1    = l1_criterion(pred, B)
            edge  = edge_criterion(pred, B)
            total = l1 + edge

        total.backward()
        optimizer.step()

        if grid is None:
            k = 4
            pred = pred.clamp(0, 1)
            grid = torch.cat([A[:k].float(), B[:k].float(), pred[:k].float()], dim=0)
            grid = make_grid(grid, nrow=k)

            with torch.no_grad():
                # bierzemy pierwsze 4 kanały jak w lossie
                eA = edge_criterion._edge_map(A[:k, :4].float()).mean(1, keepdim=True)    # [k,1,H,W]
                eB = edge_criterion._edge_map(B[:k, :4].float()).mean(1, keepdim=True)
                eP = edge_criterion._edge_map(pred[:k, :4].float()).mean(1, keepdim=True)

                # (opcjonalnie) sam diff krawędzi do diagnozy
                eDiff = (eP - eB).abs()

            def prep(x: torch.Tensor) -> torch.Tensor:
                # per-obraz normalizacja do [0,1] + zamiana na 3 kanały dla czytelnego renderu
                x_min = x.amin(dim=(1,2,3), keepdim=True)
                x_max = x.amax(dim=(1,2,3), keepdim=True)
                x = (x - x_min) / (x_max - x_min + 1e-8)
                return x.expand(-1, 3, -1, -1)

            # --- EDGES grid w tym samym układzie co RGB: (A | B | pred) ---
            edges = make_grid(torch.cat([prep(eA), prep(eB), prep(eP)], dim=0), nrow=k)

        bs = A.size(0)
        train_tot  += total.detach() * bs
        train_l1   += l1.detach()    * bs
        train_edge += edge.detach()  * bs

    train_sz = len(tiny_ds)
    wandb.log({
        "train":        (train_tot / train_sz).item(),
        "train/l1":     (train_l1  / train_sz).item(),
        "train/edge":   (train_edge/ train_sz).item(),
        "sample":       wandb.Image(grid, caption="(vanilla | styled | pred)"),
        "edges":        wandb.Image(edges,  caption="(vanilla | styled | pred)"),
    })
