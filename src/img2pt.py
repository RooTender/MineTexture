from pathlib import Path
import torch
from torchvision.io import read_image, ImageReadMode
import torch.nn.functional as F
from tqdm import tqdm

DIR = "augmented"

SRC = Path(f"data/{DIR}")
DST = Path(f"data/{DIR}_pt")

def resize_nearest_u8(img_u8, scale):
    if scale == 1:
        return img_u8
    f = img_u8.float().unsqueeze(0)  # [1,4,H,W]
    _, _, H, W = f.shape
    f = F.interpolate(f, size=(H*scale, W*scale), mode="nearest")
    return f.squeeze(0).round().to(torch.uint8)

for path in tqdm(SRC.rglob("*.png")):
    rel = path.relative_to(SRC)
    out_path = (DST / rel).with_suffix(".pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scale = 2 if "vanilla" in path.parts else 1

    img = read_image(str(path), mode=ImageReadMode.RGB_ALPHA)  # [4,H,W] uint8
    img = resize_nearest_u8(img, scale)
    torch.save(img, out_path)

