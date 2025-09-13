from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
import math

TARGET = 32

def pil_rgba_to_tensor(img: Image.Image, scale: float | None = None) -> torch.Tensor:
    img = img.convert("RGBA")

    if scale:
        dest_w = img.width  * scale
        dest_h = img.height * scale
        img = img.resize((dest_w, dest_h), Image.Resampling.NEAREST)

    return pil_to_tensor(img).float().clone() / 255.0

def _get_overlap_anchors(length: int, win: int) -> list[int]:
    """
    Pierwszy=0, ostatni=length-win, liczba okien minimalna (ceil(length/win)),
    odstępy możliwie równe. Dla length<=win zwraca [0].
    """
    if length <= win:
        return [0]

    n = math.ceil(length / win)      # liczba okien
    L = length - win                 # odcinek do pokrycia startami
    k = n - 1                        # liczba przerw między startami

    # Równomierny rozkład całkowity: floor(i * L / k)
    # Monotoniczne z definicji, pierwszy=0, ostatni=L
    starts = [(i * L) // k for i in range(n)]
    return starts

def _crop_pad_32_at(t: torch.Tensor, x: int, y: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Zwraca okno 32x32 z t [4,H,W]. Dla wymiarów >32: crop od (sy/sx).
    Dla wymiarów <=32: brak cropu w tej osi, centrowany pad do 32.
    Zwraca (patch[4,32,32]).
    """
    C, H, W = t.shape
    out = torch.zeros((C, TARGET, TARGET), dtype=t.dtype)  # own, resizable storage

    # Determine source box (y..y+32, x..x+32) intersected with the image
    y0 = max(y, 0)
    x0 = max(x, 0)
    y1 = min(y + TARGET, H)
    x1 = min(x + TARGET, W)

    # Compute where that lands in the 32x32 output (center when smaller than 32)
    if H <= TARGET:
        pad_top  = (TARGET - H) // 2
    else:
        pad_top  = 0
    if W <= TARGET:
        pad_left = (TARGET - W) // 2
    else:
        pad_left = 0

    # Destination coords in out
    dy0 = pad_top  + max(0, -min(0, y)) + (0 if H > TARGET else max(0, -y))
    dx0 = pad_left + max(0, -min(0, x)) + (0 if W > TARGET else max(0, -x))

    h = max(0, y1 - y0)
    w = max(0, x1 - x0)
    if h > 0 and w > 0:
        out[:, dy0:dy0 + h, dx0:dx0 + w] = t[:, y0:y1, x0:x1]

    return out

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

                with Image.open(img_orig) as im_a:
                    im_a = im_a.convert("RGBA")
                    if self.scale != 1:
                        im_a = im_a.resize((w * self.scale, h * self.scale), Image.Resampling.NEAREST)
                    alpha_img = im_a.getchannel("A")
                    alpha = pil_to_tensor(alpha_img).squeeze(0)
                
                # starty dla A i B w każdej osi
                x_anchors = _get_overlap_anchors(w * self.scale, TARGET)
                y_anchors = _get_overlap_anchors(h * self.scale, TARGET)

                kept = []
                for y in y_anchors:
                    for x in x_anchors:
                        if not alpha[y:y+TARGET, x:x+TARGET].any().item():
                            continue

                        kept.append((x, y))

                for x, y in kept:
                    self.items.append((img_orig, img_target, x, y))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        orig, target, x, y = self.items[idx]

        a = pil_rgba_to_tensor(Image.open(orig), self.scale)
        b = pil_rgba_to_tensor(Image.open(target))

        # Dopasowanie do 32x32 + maski (kluczowe przy małych teksturach)
        a32 = _crop_pad_32_at(a, x, y)
        b32 = _crop_pad_32_at(b, x, y)

        return {
            "input": a32,
            "target": b32
        }
