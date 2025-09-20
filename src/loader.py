from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List
import math
from tqdm import tqdm
from functools import lru_cache

TARGET = 32

def _get_overlap_anchors(length: int, win: int) -> List[int]:
    if length <= win:
        return [0]
    n = math.ceil(length / win)
    L = length - win
    k = n - 1
    return [(i * L) // k for i in range(n)]

@torch.no_grad()
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
    return F.pad(patch, (pad_left, pad_right, pad_top, pad_bot)).contiguous()

class TexturePairDataset(Dataset):
    def __init__(self, path_a: Path, path_b: Path, scale: int = 1):
        self.path_a = Path(path_a)
        self.path_b = Path(path_b)
        self.scale  = int(scale)
        self.items: list[tuple[Path, Path, int | None, int | None]] = []

        THRESHOLD_FULL = 0.01
        THRESHOLD_CROP = 0.1

        # Indeksujemy per-patch: na etapie indeksu znamy rozmiary (bez ładowania pikseli)
        for size_dir in tqdm(sorted(self.path_a.iterdir()), desc="Loading dataset"):
            w, h = map(int, size_dir.name.lower().split("x"))
            target_dir = self.path_b / f"{w*self.scale}x{h*self.scale}"

            for img_orig in sorted(size_dir.glob("*.pt")):
                img_target = target_dir / img_orig.name

                a_u8 = torch.load(img_orig, weights_only=True, map_location="cpu")
                b_u8 = torch.load(img_target, weights_only=True, map_location="cpu")
                alpha = a_u8[3]
                h, w = alpha.shape

                with torch.no_grad():
                    diff = (a_u8[:3] - b_u8[:3]).abs().sum().item()
                    diff /= (3 * h * w * 255.0)

                if diff < THRESHOLD_FULL:
                    continue

                # starty dla A i B w każdej osi
                x_anchors = _get_overlap_anchors(w, TARGET)
                y_anchors = _get_overlap_anchors(h, TARGET)

                with torch.no_grad():
                    cached_crops = []
                    for y in y_anchors:
                        for x in x_anchors:
                            if not (alpha[y:(y + TARGET), x:(x + TARGET)] != 0).any():
                                continue

                            crop = _crop_pad_32_at_uint8(a_u8, x, y)

                            redundant_crop = False
                            for cached in cached_crops:
                                diff_num = (cached[:3].to(torch.int16) - crop[:3].to(torch.int16)).abs_().sum().item()
                                diff = diff_num / (3 * h * w * 255.0)

                                if diff < THRESHOLD_CROP:
                                    redundant_crop = True
                                    break

                            if redundant_crop:
                                continue

                            cached_crops.append(crop)
                            self.items.append((str(img_orig), str(img_target), x, y))

    def __len__(self) -> int:
        return len(self.items)
    
    @lru_cache(maxsize=1024)
    def _load_pt_cached(self, path: str) -> torch.Tensor:
        # uint8 [4, H, W]
        return torch.load(path, map_location="cpu", weights_only=True)

    def __getitem__(self, idx: int):
        orig, target, x, y = self.items[idx]

        a_u8 = self._load_pt_cached(orig)
        b_u8 = self._load_pt_cached(target)

        a32_u8 = _crop_pad_32_at_uint8(a_u8, x, y)          # [4,32,32] uint8
        b32_u8 = _crop_pad_32_at_uint8(b_u8, x, y)          # [4,32,32] uint8

        return Path(orig).name, a32_u8, b32_u8

def collate(batch):
    # batch: list[(name, a_u8, b_u8)]
    names = [b[0] for b in batch]
    a = torch.stack([b[1] for b in batch], dim=0)  # [B,4,32,32] uint8
    b = torch.stack([b[2] for b in batch], dim=0)

    return names, a, b
