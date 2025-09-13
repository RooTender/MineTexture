# bucket_dataset.py
import os, math, random
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

# ======= KONFIG =======
VANILLA = "data/vanilla/1.21.4/assets/minecraft/textures"
STYLED  = "data/styled/Faithful 32x/assets/minecraft/textures"

# wiadra rozmiaru (docelowe boki obrazków w batchu)
BUCKET_SIZES = [8, 16, 32, 64]

# oversampling (prawdopodobieństwa wyboru wiadra)
BUCKET_PROBS = {8: 0.10, 16: 0.40, 32: 0.30, 64: 0.20}  # dopasuj wg swoich danych

def load_rgba(path: str) -> np.ndarray:
    im = Image.open(path).convert("RGBA")
    return np.array(im)  # H,W,4 uint8

def alpha_trim(img_rgba: np.ndarray) -> np.ndarray:
    a = img_rgba[..., 3]
    rows = np.where(a.max(axis=1) > 0)[0]
    cols = np.where(a.max(axis=0) > 0)[0]
    if rows.size == 0 or cols.size == 0:
        return img_rgba
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]
    return img_rgba[top:bottom+1, left:right+1]

def pad_to_multiple(img_rgba: np.ndarray, base: int = 8, min_size: int = 8) -> np.ndarray:
    H, W = img_rgba.shape[:2]
    Ht = max(min_size, ((H + base - 1)//base)*base)
    Wt = max(min_size, ((W + base - 1)//base)*base)
    pad_t = (Ht - H)
    pad_l = (Wt - W)
    t = pad_t // 2
    b = pad_t - t
    l = pad_l // 2
    r = pad_l - l
    if t==b==l==r==0:
        return img_rgba
    out = np.zeros((H+t+b, W+l+r, 4), dtype=img_rgba.dtype)
    out[t:t+H, l:l+W] = img_rgba
    return out

def nearest_resize(img_rgba: np.ndarray, size: int) -> np.ndarray:
    im = Image.fromarray(img_rgba, mode="RGBA").resize((size, size), Image.Resampling.NEAREST)
    return np.array(im)

def to_tensor_rgba(img_rgba: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.from_numpy(img_rgba).float().permute(2,0,1) / 255.0
    rgb = t[:3]*2-1
    a = t[3]
    return rgb, a

class TexturePairsMulti(Dataset):
    """
    Zwraca pary (vanilla, styled) jako sample.
    __getitem__ akceptuje indeks int LUB (int, bucket_size).
    """
    def __init__(self, vanilla_root: str, styled_root: str, base_multiple: int = 8):
        self.base_multiple = base_multiple
        self.items: List[Tuple[str,str,str,bool]] = []  # (vpath, spath, type, tilable)

        # zbuduj pary po relatywnej ścieżce
        for root,_,files in os.walk(vanilla_root):
            for f in files:
                if not f.lower().endswith(".png"):
                    continue
                v = os.path.join(root, f)
                rel = os.path.relpath(v, vanilla_root)
                s = os.path.join(styled_root, rel)
                if os.path.exists(s):
                    self.items.append((v, s))

        if not self.items:
            raise RuntimeError("Brak par tekstur.")

    def __len__(self):
        return len(self.items)

    def _prepare_pair(self, vpath: str, spath: str, tilable: bool, out_size: int):
        v = load_rgba(vpath)
        s = load_rgba(spath)

        # alpha-trim + pad do wielokrotności 8 (eliminuje 18x18 itp.)
        v = pad_to_multiple(alpha_trim(v), base=self.base_multiple, min_size=8)
        s = pad_to_multiple(alpha_trim(s), base=self.base_multiple, min_size=8)

        # konwersja → tensory
        v_rgb, v_a = to_tensor_rgba(v)
        s_rgb, s_a = to_tensor_rgba(s)

        # skala względem bazowego 16×16 (log2 daje sensowną rozpiętość)
        scale_log2 = math.log2(out_size / 16.0)
        is_tilable = 1.0 if tilable else 0.0

        sample = {
            "van_rgb": v_rgb, "van_a": v_a,
            "sty_rgb": s_rgb, "sty_a": s_a,
            "is_tilable": torch.tensor([is_tilable], dtype=torch.float32),
            "scale_log2": torch.tensor([scale_log2], dtype=torch.float32),
            "out_size": out_size,
            "vpath": vpath
        }
        return sample

    def __getitem__(self, key: Union[int, Tuple[int,int]]):
        if isinstance(key, tuple):
            idx, out_size = key
        else:
            idx, out_size = key, random.choice(BUCKET_SIZES)
        vpath, spath, til = self.items[idx]
        return self._prepare_pair(vpath, spath, til, out_size)
