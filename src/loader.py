from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from functools import lru_cache
from typing import Tuple, List, Dict
from torchvision.io import read_image, ImageReadMode
import math

TARGET = 32

def _get_overlap_anchors(length: int, win: int) -> List[int]:
    if length <= win:
        return [0]
    n = math.ceil(length / win)
    L = length - win
    k = n - 1
    return [(i * L) // k for i in range(n)]

def _crop_pad_32_at_uint8(t: torch.Tensor, x: int, y: int) -> torch.Tensor:
    """
    t: [4,H,W] uint8. Returns [4,32,32] uint8. Center-pad if H/W < 32.
    """
    C, H, W = t.shape

    # source bounds
    y0, x0 = max(y, 0), max(x, 0)
    y1, x1 = min(y + TARGET, H), min(x + TARGET, W)

    # allocate once via pad rather than manual indexing
    # compute pad to place the (potentially smaller than 32x32) crop centered
    crop_h, crop_w = max(0, y1 - y0), max(0, x1 - x0)
    if crop_h == 0 or crop_w == 0:
        # fully empty → just zeros
        return torch.zeros((C, TARGET, TARGET), dtype=t.dtype)

    patch = t[:, y0:y1, x0:x1]  # [4, crop_h, crop_w]

    pad_top  = (TARGET - crop_h) // 2 if crop_h < TARGET else 0
    pad_bot  = TARGET - crop_h - pad_top if crop_h < TARGET else 0
    pad_left = (TARGET - crop_w) // 2 if crop_w < TARGET else 0
    pad_right= TARGET - crop_w - pad_left if crop_w < TARGET else 0

    # F.pad pads last dims: (left, right, top, bottom)
    return F.pad(patch, (pad_left, pad_right, pad_top, pad_bot))

@lru_cache(maxsize=4096)
def _read_rgba_uint8_resized(path: str, scale: int) -> torch.Tensor:
    """
    Fast decode PNG → [4,H,W] uint8 (0..255), optional resize (nearest).
    Cached per (path, scale) in each worker process.
    """
    img = read_image(path, mode=ImageReadMode.RGB_ALPHA)  # [4,H,W] uint8
    if scale != 1:
        # resize in float to use interpolate, then round back to uint8
        f = img.float().unsqueeze(0)  # [1,4,H,W]
        _, _, H, W = f.shape
        f = F.interpolate(
            f, size=(H * scale, W * scale),
            mode="nearest"
        )
        img = f.squeeze(0).round().to(torch.uint8)
    return img  # [4,H',W'] uint8

class TexturePairDataset(Dataset):
    def __init__(self, path_a: Path, path_b: Path, scale: int):
        self.path_a = Path(path_a)
        self.path_b = Path(path_b)
        self.scale = int(scale)
        self.items: list[tuple[Path, Path, int | None, int | None]] = []

        # Indeksujemy per-patch: na etapie indeksu znamy rozmiary (bez ładowania pikseli)
        for size_dir in sorted(self.path_a.iterdir()):
            w, h = map(int, size_dir.name.lower().split("x"))
            target_dir = self.path_b / f"{w*self.scale}x{h*self.scale}"

            for img_orig in sorted(size_dir.glob("*.png")):
                img_target = target_dir / img_orig.name

                a_uint8 = _read_rgba_uint8_resized(str(img_orig), self.scale)
                alpha = a_uint8[3]
                h, w = alpha.shape
                
                # starty dla A i B w każdej osi
                x_anchors = _get_overlap_anchors(w, TARGET)
                y_anchors = _get_overlap_anchors(h, TARGET)

                for y in y_anchors:
                    for x in x_anchors:
                        if (alpha[y:(y + TARGET), x:(x + TARGET)] != 0).any():
                            self.items.append((str(img_orig), str(img_target), x, y))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        orig, target, x, y = self.items[idx]

        a_u8 = _read_rgba_uint8_resized(orig, self.scale)   # [4,H,W] uint8
        b_u8 = _read_rgba_uint8_resized(target, 1)          # [4,H,W] uint8

        a32_u8 = _crop_pad_32_at_uint8(a_u8, x, y)          # [4,32,32] uint8
        b32_u8 = _crop_pad_32_at_uint8(b_u8, x, y)          # [4,32,32] uint8

        # convert at the end; avoid .clone(), just to(float)/div_
        a32 = a32_u8.to(torch.float32).mul_(1.0 / 255.0)
        b32 = b32_u8.to(torch.float32).mul_(1.0 / 255.0)

        return {
            "basename": Path(orig).name,
            "input": a32,
            "target": b32
        }
