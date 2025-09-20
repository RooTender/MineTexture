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
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

WATCH_NAMES = ['bow.pt', 'seagrass_0.pt', 'raw_copper_block.pt', 'comparator.pt']

EPOCHS        = 250
BATCH_SIZE    = 512
LEARNING_RATE = 2e-3
WEIGHT_DECAY  = 0 # 1e-5

DEVICE = 'cuda'


train_ds = TexturePairDataset(
    path_a=Path("data/split_pt/valid/vanilla"),
    path_b=Path("data/split_pt/valid/styled"),
    scale=2,
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          pin_memory=True, pin_memory_device=DEVICE, drop_last=True)

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
    name="edge_loss_1+2",
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
    train_wl1 = torch.zeros((), device=DEVICE)
    watched = {}

    model.train()

    for batch in tqdm(train_loader, desc=f"Training ({epoch}/{EPOCHS})"):
        names, A, B = batch
        A = A.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        B = B.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            A = A.float().mul_(1/255.0)
            B = B.float().mul_(1/255.0)

            pred = model(A)
            total, l1, edge, wl1 = compute_losses(pred, B)

        total.backward()
        optimizer.step()

        bs = A.size(0)
        train_tot  += total.detach() * bs
        train_l1   += l1.detach()    * bs
        train_edge += edge.detach()  * bs
        train_wl1  += wl1.detach()   * bs

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

    with torch.no_grad():
        def edge_total(x: torch.Tensor):
            rgb = x[:, :3].clamp(0, 1)
            a   = x[:, 3:4].clamp(0, 1)
            # linear + premultiply
            rgb_lin = edge_criterion.srgb_to_linear(rgb)
            y_lin   = edge_criterion.luma709_linear(rgb_lin * a)
            g_y     = edge_criterion.edge_map(y_lin)
            g_a     = edge_criterion.edge_map(a)
            return g_y + g_a   # SUMA jak w forward

        eA = edge_total(A_cat)
        eB = edge_total(B_cat)
        eP = edge_total(P_cat)
        eDiff = (eP - eB).abs()

    def norm_to_rgb(x):
        x_min = x.amin(dim=(2,3), keepdim=True)
        x_max = x.amax(dim=(2,3), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)
        return x.expand(-1, 3, -1, -1)

    edges = make_grid(
        torch.cat([
            norm_to_rgb(eA), norm_to_rgb(eB), norm_to_rgb(eP),  # edge maps
            #norm_to_rgb(eDiff),                                # opcjonalnie różnica
        ], dim=0),
        nrow=len(WATCH_NAMES)
    )

    train_sz = len(train_ds)
    wandb.log({
        "train":          (train_tot / train_sz).item(),
        "train/l1":       (train_l1  / train_sz).item(),
        "train/edge":     (train_edge/ train_sz).item(),
        "train/wl1":      (train_wl1 / train_sz).item(),
        "sample":       wandb.Image(samples, caption="(vanilla | styled | pred)"),
        "edges":        wandb.Image(edges,   caption="(vanilla | styled | pred)"),
    })
