from pathlib import Path
import os
from PIL import Image
import shutil
import random

src_vanilla = Path("data/vanilla/1.21.4")
src_styled  = Path("data/styled/Faithful 32x")
dst_root    = Path("data/sorted")

SCALE = 2  # expected ratio styled/vanilla

for file in src_vanilla.rglob("*"):
    if not file.is_file():
        continue

    rel = file.relative_to(src_vanilla)
    styled_file = src_styled / rel
    if not styled_file.exists():
        continue  # skip if no pair

    with Image.open(file) as img_v:
        wv, hv = img_v.size
    with Image.open(styled_file) as img_s:
        ws, hs = img_s.size

    # check scale
    if ws != wv * SCALE or hs != hv * SCALE:
        continue  # skip wrong scale

    # --- vanilla ---
    dst_v = dst_root / "vanilla" / f"{wv}x{hv}"
    os.makedirs(dst_v, exist_ok=True)
    shutil.copy2(file, dst_v)

    # --- styled ---
    dst_s = dst_root / "styled" / f"{ws}x{hs}"
    os.makedirs(dst_s, exist_ok=True)
    shutil.copy2(styled_file, dst_s)

sorted_root = Path("data/sorted")
split_root  = Path("data/split")
train_ratio = 0.8

random.seed(42)

vanilla_idx = {}
styled_idx  = {}

v_root = sorted_root / "vanilla"
s_root = sorted_root / "styled"

if v_root.exists():
    for size_dir in v_root.iterdir():
        if size_dir.is_dir():
            for f in size_dir.iterdir():
                if f.is_file():
                    vanilla_idx[f.name] = f

if s_root.exists():
    for size_dir in s_root.iterdir():
        if size_dir.is_dir():
            for f in size_dir.iterdir():
                if f.is_file():
                    styled_idx[f.name] = f

paired_names = list(set(vanilla_idx.keys()) & set(styled_idx.keys()))
random.shuffle(paired_names)

split_idx = int(len(paired_names) * train_ratio)

train_names = paired_names[:split_idx]
valid_names = paired_names[split_idx:]

def copy_pair(name: str, split: str):
    # VANILLA
    v_path = vanilla_idx[name]
    v_size = v_path.parent.name  # np. "32x32"
    dst_v_dir = split_root / split / "vanilla" / v_size
    os.makedirs(dst_v_dir, exist_ok=True)
    shutil.copy2(v_path, dst_v_dir)

    # STYLED
    s_path = styled_idx[name]
    s_size = s_path.parent.name
    dst_s_dir = split_root / split / "styled" / s_size
    os.makedirs(dst_s_dir, exist_ok=True)
    shutil.copy2(s_path, dst_s_dir)

# 3) Kopiuj pary do train/valid
for n in train_names:
    copy_pair(n, "train")
for n in valid_names:
    copy_pair(n, "valid")

SRC_ROOT = Path("data/split")  # skąd bierzemy train/valid po wstępnym splicie
DST_ROOT = Path("data/bucket")      # dokąd tworzymy kubełki x8..x256

BINS = [8, 16, 32, 64, 128, 256, 512]

def pick_bin(min_side: int) -> int:
    for b in BINS:
        if min_side <= b:
            return b
    return BINS[-1]  # >256 -> x256

for split in ["train", "valid"]:
    for kind in ["vanilla", "styled"]:
        src_kind = SRC_ROOT / split / kind
        if not src_kind.exists():
            continue

        # iterujemy po katalogach rozmiarów (np. 16x16, 32x32), bez wchodzenia w podkatalogi
        for size_dir in src_kind.iterdir():
            if not size_dir.is_dir():
                continue

            for f in size_dir.iterdir():
                if not f.is_file():
                    continue

                with Image.open(f) as img:
                    w, h = img.size
                bin_size = pick_bin(min(w, h))

                dst_dir = DST_ROOT / split / kind / f"x{bin_size}"
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(f, dst_dir)

print("Buckets done")
