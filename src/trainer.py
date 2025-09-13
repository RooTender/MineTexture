from pathlib import Path
import torch
import torch.nn as nn
from loader import TexturePairDataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from model import TinyUNet
from tqdm import tqdm
import wandb


EPOCHS        = 50
BATCH_SIZE    = 256
LEARNING_RATE = 2e-4
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

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=4, pin_memory=True, persistent_workers=True)

model = TinyUNet(in_ch=4, base=32, out_ch=4).to(DEVICE).to(memory_format=torch.channels_last)
# model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.L1Loss(reduction="mean")

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
    train_loss = 0.0
    valid_loss = 0.0

    model.train()

    for batch in tqdm(train_loader, desc=f"Training ({epoch}/{EPOCHS})"):
        A = batch["input"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        B = batch["target"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)

        optimizer.zero_grad()

        pred = model(A)
        loss = criterion(pred, B)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * A.size(0)
    
    model.eval()

    with torch.no_grad():
        for batch in tqdm(valid_loader, desc=f"Validating ({epoch}/{EPOCHS})"):
            A = batch["input"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
            B = batch["target"].to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)

            pred = model(A)
            loss = criterion(pred, B)

            valid_loss += loss.item() * A.size(0)

    wandb.log({
        "train": train_loss / len(train_ds),
        "valid": valid_loss / len(valid_ds),
    })
