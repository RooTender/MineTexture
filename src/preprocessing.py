from pathlib import Path
from PIL import Image
import shutil
import random

src_vanilla = Path("data/vanilla/1.20.4")
src_styled  = Path("data/styled/Faithful 32x - 1.20.4")
dst_root    = Path("data/sorted")

SCALE = 2  # expected ratio styled/vanilla

def save_as_rgba_png(src_path: Path, dst_dir: Path) -> Path:
    """
    Open an image, convert to RGBA, and save as PNG (icc_profile stripped).
    Returns the destination path.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / (src_path.stem + ".png")
    with Image.open(src_path) as im:
        im = im.convert("RGBA")
        im.save(dst_path, format="PNG", optimize=True, icc_profile=None)
    return dst_path

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
    save_as_rgba_png(file, dst_v)

    # --- styled ---
    dst_s = dst_root / "styled" / f"{ws}x{hs}"
    save_as_rgba_png(styled_file, dst_s)

sorted_root = Path("data/sorted")
split_root  = Path("data/split")
train_ratio = 0.8

random.seed(42)

v_root = sorted_root / "vanilla"
s_root = sorted_root / "styled"

def parse_size(name: str):
    name = name.lower().replace('×', 'x')
    w_str, h_str = name.split('x', 1)
    return int(w_str), int(h_str)

def mul_size(name: str, k: int) -> str:
    w, h = parse_size(name)
    return f"{w*k}x{h*k}"

# Zbierz indeksy: (nazwa_pliku, rozmiar) -> lista ścieżek
vanilla_idx = {}
styled_idx  = {}

if v_root.exists():
    for v_size_dir in v_root.iterdir():
        if not v_size_dir.is_dir():
            continue
        v_size = v_size_dir.name
        for f in v_size_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ".png":
                vanilla_idx.setdefault((f.name, v_size), []).append(f)

if s_root.exists():
    for s_size_dir in s_root.iterdir():
        if not s_size_dir.is_dir():
            continue
        s_size = s_size_dir.name
        for f in s_size_dir.iterdir():
            if f.is_file() and f.suffix.lower() == ".png":
                styled_idx.setdefault((f.name, s_size), []).append(f)

# Zbuduj pary: dla każdego (name, v_size) szukaj (name, s_size = v_size*SCALE)
pairs = []
for (name, v_size), v_list in vanilla_idx.items():
    s_size = mul_size(v_size, SCALE)
    s_list = styled_idx.get((name, s_size), [])
    if not s_list:
        continue
    # ułóż losowo, by potem deterministycznie “zużywać”
    random.shuffle(v_list)
    random.shuffle(s_list)
    # bierz 1:1 tyle, ile jest wspólnych sztuk
    take = min(len(v_list), len(s_list))
    for i in range(take):
        v_file = v_list[i]
        s_file = s_list[i]
        pairs.append((v_file, s_file, v_size, s_size))

# potasuj pary i podziel
random.shuffle(pairs)
split_idx = int(len(pairs) * train_ratio)
train_pairs = pairs[:split_idx]
valid_pairs = pairs[split_idx:]

def copy_pair(v_file: Path, s_file: Path, v_size: str, s_size: str, split: str):
    # VANILLA
    dst_v_dir = split_root / split / "vanilla" / v_size
    dst_v_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(v_file, dst_v_dir)

    # STYLED
    dst_s_dir = split_root / split / "styled" / s_size
    dst_s_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s_file, dst_s_dir)

for v_file, s_file, v_size, s_size in train_pairs:
    copy_pair(v_file, s_file, v_size, s_size, "train")
for v_file, s_file, v_size, s_size in valid_pairs:
    copy_pair(v_file, s_file, v_size, s_size, "valid")

print(f"Pary znalezione: {len(pairs)} | train: {len(train_pairs)} | valid: {len(valid_pairs)}")
