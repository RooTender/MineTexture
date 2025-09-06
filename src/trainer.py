from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
import numpy as np
import random, math
from collections import defaultdict
from tqdm import tqdm

BATCH_SIZE = 4
BASE = 32

def side_bin(x: int) -> str:
    if x <= 8:    return "x8"
    if x <= 16:   return "x16"
    if x <= 32:   return "x32"
    if x <= 64:   return "x64"
    if x <= 128:  return "x128"
    if x <= 256:  return "x256"
    return "x512"

def aspect_bucket(h: int, w: int) -> str:
    s = min(h, w); l = max(h, w)
    # orientacja: czy krótszy bok jest wysokością (h<=w)?
    ori = "hshort" if h <= w else "wshort"
    return f"{ori}|{side_bin(s)}|{side_bin(l)}"

class TexturePairs(Dataset):
    def __init__(self, root_vanilla: str, root_styled: str):
        self.v_root = Path(root_vanilla)
        self.s_root = Path(root_styled)

        # mapa ścieżek względem kubełka (zachowujesz swoją logikę)
        self.styled_map = {}
        for s_bucket in sorted(self.s_root.glob("x*")):
            if not s_bucket.is_dir(): continue
            for p_s in s_bucket.rglob("*"):
                if p_s.is_file():
                    rel = p_s.relative_to(s_bucket)
                    self.styled_map[str(rel).replace("\\", "/")] = p_s

        self.pairs: List[Tuple[Path, Path]] = []
        self.sizes: List[Tuple[int,int]] = []   # (H,W) stylu
        self.bins:  List[str] = []              # kosz po stylu

        for v_bucket in sorted(self.v_root.glob("x*")):
            if not v_bucket.is_dir(): continue
            for p_v in v_bucket.rglob("*"):
                if not p_v.is_file(): continue
                rel = str(p_v.relative_to(v_bucket)).replace("\\", "/")
                p_s = self.styled_map.get(rel, None)
                if p_s is None: continue

                # odczytaj rozmiar stylu (1x open bez dekodowania kanałów)
                with Image.open(p_s) as im_s:
                    w_s, h_s = im_s.size
                self.pairs.append((p_v, p_s))
                self.sizes.append((h_s, w_s))
                self.bins.append(aspect_bucket(h_s, w_s))

        # stabilna kolejność
        self.pairs, self.sizes, self.bins = zip(*sorted(
            zip(self.pairs, self.sizes, self.bins),
            key=lambda t: (str(t[0][0]).lower(), str(t[0][1]).lower())
        ))
        self.pairs = list(self.pairs); self.sizes = list(self.sizes); self.bins = list(self.bins)

    @staticmethod
    def _load_rgba(path: Path) -> Image.Image:
        with Image.open(path) as img:
            return img.convert("RGBA")

    @staticmethod
    def _to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1)  # uint8

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        p_v, p_s = self.pairs[idx]
        img_v = self._load_rgba(p_v)
        img_s = self._load_rgba(p_s)

        # dopasuj vanilla do stylu (NEAREST, zero AA)
        if img_v.size != img_s.size:
            img_v = img_v.resize(img_s.size, resample=Image.Resampling.NEAREST)

        sample = {
            "vanilla": self._to_tensor(img_v),   # [4,H,W]
            "styled":  self._to_tensor(img_s),   # [4,H,W]
            "bin":     self.bins[idx],           # np. "x32"
            "size":    self.sizes[idx],          # (H,W)
        }
        return sample


def pad_to_target(x: torch.Tensor, Ht: int, Wt: int) -> torch.Tensor:
    # x: [C,H,W]
    _, H, W = x.shape
    pt = max((Ht - H) // 2, 0); pb = Ht - H - pt
    pl = max((Wt - W) // 2, 0); pr = Wt - W - pl
    return F.pad(x.unsqueeze(0), (pl, pr, pt, pb), mode='constant', value=0).squeeze(0)


BUCKET_TARGET = {"x8":8,"x16":16,"x32":32,"x64":64,"x128":128,"x256":256,"x512":512}

def targets_from_bucket(bucket_key: str) -> Tuple[int,int]:
    # "hshort|x8|x128" albo "wshort|x16|x64"
    ori, sbin, lbin = bucket_key.split("|")
    S = BUCKET_TARGET[sbin]
    L = BUCKET_TARGET[lbin]
    Ht, Wt = (S, L) if ori == "hshort" else (L, S)
    # dopnij do wielokrotności 4 i min 16 (jak wcześniej)
    Ht = max(16, (Ht + 3)//4*4)
    Wt = max(16, (Wt + 3)//4*4)
    return Ht, Wt

def collate_pad(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    bucket_key = batch[0]["bin"]
    Ht, Wt = targets_from_bucket(bucket_key)

    V, S, M = [], [], []
    for b in batch:
        v, s = b["vanilla"], b["styled"]
        mask = torch.ones(1, s.shape[-2], s.shape[-1], dtype=s.dtype)
        V.append(pad_to_target(v, Ht, Wt))
        S.append(pad_to_target(s, Ht, Wt))
        M.append(pad_to_target(mask, Ht, Wt))

    return {
        "vanilla": torch.stack(V, 0),
        "styled":  torch.stack(S, 0),
        "mask":    torch.stack(M, 0),
        "bin":     bucket_key,
    }


class BucketBatchSampler(Sampler[List[int]]):
    """
    Tworzy batch'e tylko z jednego kosza (np. x16), a kosze podaje round-robin.
    Dla rzadkich koszy robi oversampling, żeby domknąć pełne batch'e.
    """
    def __init__(self, dataset: TexturePairs, batch_size: int = 4, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        # indeksy per kosz
        buckets = defaultdict(list)
        for i, b in enumerate(dataset.bins):
            buckets[b].append(i)
        self.buckets = {k: v[:] for k, v in buckets.items()}
        self.bucket_keys = sorted(self.buckets.keys(), key=lambda k: (len(self.buckets[k])==0, k))

    def __iter__(self):
        # przygotuj kolejki per kosz
        qs = {}
        for k in self.bucket_keys:
            q = self.buckets[k][:]
            if self.shuffle: random.shuffle(q)
            # oversampling do wielokrotności batch_size
            need = (-len(q)) % self.batch_size
            if need and len(q) > 0:
                q += random.choices(q, k=need)
            qs[k] = q

        # ile batchy per kosz
        per_bucket_batches = {k: len(v)//self.batch_size for k, v in qs.items()}
        maxb = max(per_bucket_batches.values()) if per_bucket_batches else 0

        # round-robin po koszach
        ptr = {k: 0 for k in self.bucket_keys}

        for _ in range(maxb):
            for k in self.bucket_keys:
                nb = per_bucket_batches[k]
                if nb == 0: continue
                if ptr[k] >= len(qs[k]): continue
                batch = qs[k][ptr[k]:ptr[k]+self.batch_size]
                ptr[k] += self.batch_size
                if batch: yield batch

    def __len__(self):
        total = 0
        for v in self.buckets.values():
            total += math.ceil(len(v)/self.batch_size) if v else 0
        return total

class BlockBucketBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset: TexturePairs, batch_size: int = 4, block_batches: int = 32, shuffle: bool = True):
        self.batch_size = batch_size; self.block_batches = block_batches; self.shuffle = shuffle
        buckets = defaultdict(list)
        for i, b in enumerate(dataset.bins): buckets[b].append(i)
        self.buckets = {k: v[:] for k,v in buckets.items()}
        self.bucket_keys = list(self.buckets.keys())

    def __iter__(self):
        qs = {}
        for k in self.bucket_keys:
            q = self.buckets[k][:]
            if self.shuffle: random.shuffle(q)
            need = (-len(q)) % self.batch_size
            if need and len(q)>0: q += random.choices(q, k=need)
            qs[k] = [q[i:i+self.batch_size] for i in range(0, len(q), self.batch_size)]
        keys = self.bucket_keys[:]
        if self.shuffle: random.shuffle(keys)
        for k in keys:
            batches = qs[k]
            for i in range(0, len(batches), self.block_batches):
                for b in batches[i:i+self.block_batches]:
                    if b: yield b

    def __len__(self):
        total = 0
        for v in self.buckets.values():
            total += math.ceil(len(v)/self.batch_size) if v else 0
        return total


train_pairs = TexturePairs(
    root_vanilla="data/augmented/train/vanilla",
    root_styled="data/augmented/train/styled",
)

valid_pairs = TexturePairs(
    root_vanilla="data/augmented/valid/vanilla",
    root_styled="data/augmented/valid/styled",
)

train_sampler = BlockBucketBatchSampler(train_pairs, batch_size=BATCH_SIZE, block_batches=128, shuffle=True)
val_sampler   = BlockBucketBatchSampler(valid_pairs,  batch_size=BATCH_SIZE, block_batches=128, shuffle=False)

train_dl = DataLoader(train_pairs,
                      batch_sampler=train_sampler,
                      num_workers=4, pin_memory=True,
                      persistent_workers=True, pin_memory_device="cuda",
                      prefetch_factor=8, collate_fn=collate_pad)

val_dl   = DataLoader(valid_pairs,
                      batch_sampler=val_sampler,
                      num_workers=4, pin_memory=True,
                      persistent_workers=True, pin_memory_device="cuda",
                      prefetch_factor=8, collate_fn=collate_pad)

import wandb
from collections import defaultdict
from model import TinyUNetLite, l1_masked

device = torch.device("cuda")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

model = TinyUNetLite(in_ch=4, base=BASE, out_ch=4, padding_mode='zeros').to(device)
model = model.to(memory_format=torch.channels_last)

# print("Compiling...")
# compile_mode = "reduce-overhead"
# import torch._dynamo as dynamo
# dynamo.config.cache_size_limit = 64 
# model = torch.compile(model, mode=compile_mode, dynamic=False)
# print("Done")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, fused=True)

wandb.init(project="MineTexture", config={
    "batch_size": BATCH_SIZE,
    "lr": 1e-3,
    "model": f"TinyUNetLite(base={BASE})",
    "loss": "L1_masked",
})

def depth_from_bin(bucket_key: str) -> int:
    # "hshort|x8|x128" → weź długi bok (lbin)
    _, _sbin, lbin = bucket_key.split("|")
    if lbin in ("x8","x16"):   return 0
    if lbin in ("x32","x64"):  return 1
    return 2  # x128, x256, x512...


def run_epoch(dloader, train=True):
    model.train(train)
    ctx = torch.enable_grad() if train else torch.no_grad()

    total_loss, total_count = 0.0, 0
    bin_loss_sum, bin_count = defaultdict(float), defaultdict(int)

    major, _ = torch.cuda.get_device_capability()
    amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    amp_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype)

    with ctx:
        for batch in tqdm(dloader):
            x = batch["vanilla"].to(device, non_blocking=True).to(torch.float32).div_(255.0).contiguous(memory_format=torch.channels_last)
            y = batch["styled" ].to(device, non_blocking=True).to(torch.float32).div_(255.0).contiguous(memory_format=torch.channels_last)
            m = batch["mask"   ].to(device, non_blocking=True).to(torch.float32).contiguous(memory_format=torch.channels_last)

            bname = batch["bin"]; depth = depth_from_bin(bname)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                yhat = model(x, depth=depth)
                loss = l1_masked(yhat, y, m)

            if train:
                loss.backward()
                optimizer.step()

            bs = x.size(0)
            total_loss += loss.item() * bs; total_count += bs
            bin_loss_sum[bname] += loss.item() * bs; bin_count[bname] += bs

    avg = total_loss / max(1,total_count)
    per_bin = {k: bin_loss_sum[k]/max(1,bin_count[k]) for k in bin_loss_sum}
    return avg, per_bin


NUM_EPOCHS = 10
for epoch in range(1, NUM_EPOCHS+1):
    train_avg, train_bins = run_epoch(train_dl, train=True)
    val_avg,   val_bins   = run_epoch(val_dl,   train=False)

    log = {
        "epoch": epoch,
        "train/avg": train_avg,
        "valid/avg": val_avg,
    }
    for k,v in sorted(train_bins.items()): log[f"train/bin/{k}"] = v
    for k,v in sorted(val_bins.items()):   log[f"valid/bin/{k}"] = v
    # wandb.log(log)

    print(f"[{epoch:03d}] train={train_avg:.6f}  valid={val_avg:.6f}")
