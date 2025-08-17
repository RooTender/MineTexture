# train_adapter_texture.py
import os, random, re
from typing import List
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.schedulers import DDPMScheduler
from peft import LoraConfig, TaskType, get_peft_model
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
from tqdm import tqdm

# === Ścieżki ===
VANILLA = "data/vanilla/1.21.4/assets/minecraft/textures/block"
STYLED  = "data/styled/Faithful 32x/assets/minecraft/textures/block"

MODEL_BASE   = "segmind/tiny-sd"
OUT_DIR      = "output/adapter_only"

# === Hiperparametry ===
SEED = 42
TRAIN_SIZE = 512         # 256 dla pikselartu; wielokrotność 8
BATCH = 4               # 3060 Ti mobile powinna uciągnąć 8-16 przy xFormers
EPOCHS = 3
LR_ADAPTER = 5e-5        # spokojny LR dla adaptera
LR_LORA    = 5e-5
RANK = 16
USE_DORA = True
USE_ADAPTER   = True     # << główna rzecz
USE_UNET_LORA = True    # włącz dopiero gdy adapter nie domaga

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# === Dataset parujący vanilla->styled ===
class TexturePairs(Dataset):
    def __init__(self, vanilla_root: str, styled_root: str):
        self.pairs: List[tuple[str,str]] = []
        for root, _, files in os.walk(vanilla_root):
            for f in files:
                if not f.lower().endswith(".png"): continue
                vpath = os.path.join(root, f)
                rel   = os.path.relpath(vpath, vanilla_root)
                spath = os.path.join(styled_root, rel)
                if os.path.exists(spath):
                    self.pairs.append((vpath, spath))
        if not self.pairs:
            raise RuntimeError("Nie znaleziono par (vanilla vs styled). Sprawdź ścieżki.")

    def __len__(self): return len(self.pairs)

    def _load_rgba_to_rgb_norm(self, path: str):
        img = Image.open(path).convert("RGBA")
        # integer NEAREST-upscale do TRAIN_SIZE
        img = img.resize((TRAIN_SIZE, TRAIN_SIZE), Image.Resampling.NEAREST)
        rgba = np.array(img)
        rgb  = rgba[:, :, :3]
        rgb  = torch.from_numpy(rgb).float().permute(2,0,1) / 255.0
        rgb  = rgb * 2.0 - 1.0
        alpha = torch.from_numpy(rgba[:, :, 3]).float().unsqueeze(0) / 255.0
        return rgb, alpha

    def __getitem__(self, idx):
        vpath, spath = self.pairs[idx]
        v_rgb, v_a = self._load_rgba_to_rgb_norm(vpath)
        s_rgb, s_a = self._load_rgba_to_rgb_norm(spath)

        # prosty prompt z nazwą
        base = os.path.splitext(os.path.basename(spath))[0]
        parts = [p for p in base.split("_") if not p.isdigit()]
        name  = re.sub(r"\s+", " ", " ".join(parts)).strip()

        return {
            "vanilla_rgb": v_rgb, "vanilla_a": v_a,
            "styled_rgb":  s_rgb, "styled_a":  s_a,
            "prompt":      f"Minecraft {name} in faithful 32x style",
        }

# === TinyAdapter: z obrazu (RGB 256x256) -> residua dla bloków UNeta ===
class TinyAdapter(nn.Module):
    def __init__(self, unet_block_out_channels: list[int], in_ch=3, hidden=256):
        super().__init__()
        # 256 -> 128 -> 64 -> 32
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
            nn.Conv2d(128, hidden, 3, 2, 1), nn.GELU(),
        )
        # projekcje 32/16/8; kanały: c0, c1, c2
        self.proj32 = nn.Conv2d(hidden,  unet_block_out_channels[0], 1)
        self.down16 = nn.Sequential(nn.Conv2d(hidden, hidden, 3, 2, 1), nn.GELU())
        self.proj16 = nn.Conv2d(hidden,  unet_block_out_channels[1], 1)
        self.down8  = nn.Sequential(nn.Conv2d(hidden, hidden, 3, 2, 1), nn.GELU())
        self.proj8  = nn.Conv2d(hidden,  unet_block_out_channels[2], 1)

        # 8 -> 4; kanały takie jak ostatni poziom (c2), BEZ odwoływania się do [3]
        self.down4  = nn.Sequential(nn.Conv2d(hidden, hidden, 3, 2, 1), nn.GELU())
        self.proj4  = nn.Conv2d(hidden,  unet_block_out_channels[-1], 1)  # <- [-1], nie [3]

        # mid też w kanałach ostatniego poziomu
        self.mid    = nn.Conv2d(unet_block_out_channels[-1], unet_block_out_channels[-1], 1)

    def forward(self, x):
        h32 = self.stem(x)             # 32x32
        r32 = self.proj32(h32)

        h16 = self.down16(h32)         # 16x16
        r16 = self.proj16(h16)

        h8  = self.down8(h16)          # 8x8
        r8  = self.proj8(h8)

        h4  = self.down4(h8)           # 4x4
        r4  = self.proj4(h4)

        mid = self.mid(r4)             # 4x4
        return [r32, r16, r8, r4], mid



# === Modele bazowe ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_BASE, torch_dtype=torch.float16, use_safetensors=False
)
vae, tokenizer, text_encoder, unet = pipe.vae, pipe.tokenizer, pipe.text_encoder, pipe.unet
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear")

vae.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)

pipe.enable_vae_slicing()
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

# Freeze bazowych wag
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

unet_lora_cfg = LoraConfig(
    r=RANK,
    lora_alpha=RANK,
    lora_dropout=0.05,
    init_lora_weights="gaussian",
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
)
unet.add_adapter(unet_lora_cfg)
unet.enable_adapters()

for p in unet.parameters():
    if p.requires_grad and p.dtype != torch.float32:
        p.data = p.data.float()

accelerator = Accelerator(mixed_precision="fp16")

# Adapter
if USE_ADAPTER:
    adapter = TinyAdapter(unet.config.block_out_channels).to(device=accelerator.device, dtype=torch.float32) # FP32!
else:
    adapter = None

# === Dataloader ===
ds = TexturePairs(VANILLA, STYLED)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

# === Accelerator (fp16) ===
lora_params = [p for p in unet.parameters() if p.requires_grad]
opt_groups = []

if USE_ADAPTER:
    opt_groups.append({"params": adapter.parameters(), "lr": LR_ADAPTER})

if USE_UNET_LORA:
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    opt_groups.append({"params": lora_params, "lr": LR_LORA})

optimizer = torch.optim.AdamW(opt_groups, weight_decay=1e-2, eps=1e-8)

if USE_ADAPTER:
    unet, adapter, text_encoder, vae, optimizer, dl = accelerator.prepare(
        unet, adapter, text_encoder, vae, optimizer, dl
    )
else:
    unet, text_encoder, vae, optimizer, dl = accelerator.prepare(
        unet, text_encoder, vae, optimizer, dl
    )

unet.train()
if USE_ADAPTER: adapter.train()

# === Helper: VRAM print ===
def print_vram(prefix=""):
    if not torch.cuda.is_available(): return
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved  = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    max_res   = torch.cuda.max_memory_reserved() / 1024**2
    print(f"{prefix} VRAM allocated={allocated:.0f}MB reserved={reserved:.0f}MB "
          f"(max_alloc={max_alloc:.0f}MB max_res={max_res:.0f}MB)")

# === Trening ===
global_step = 0
for epoch in range(EPOCHS):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for batch in tqdm(dl, desc=f"Epoch ({epoch + 1}/{EPOCHS})"):
        with accelerator.accumulate(unet):
            # 1) Tekst -> embedding
            with torch.no_grad():
                tokens = tokenizer(
                    batch["prompt"], padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt",
                )
                input_ids = tokens.input_ids.to(device)
                embeddings = text_encoder(input_ids)[0]

            # 2) Styled -> latenty (target do denoise)
            with torch.no_grad():
                styled_rgb = batch["styled_rgb"].to(accelerator.device, dtype=torch.float16)
                latents = vae.encode(styled_rgb).latent_dist.sample() * vae.config.scaling_factor

            # 3) Dodaj szum według losowego t
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=latents.device, dtype=torch.long
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 4) Adapter: przetwórz VANILLA i przygotuj residua
            # 4) Adapter: przetwórz VANILLA i przygotuj residua DOPASOWANE do UNeta
            down_residuals = None
            if USE_ADAPTER:
                vanilla_rgb = batch["vanilla_rgb"].to(noisy_latents.device, dtype=noisy_latents.dtype)
                (r32, r16, r8, _), _ = adapter(vanilla_rgb)  # ignoruj r4 i mid_r
                down_residuals = [r32, r16, r8]

            # 5) UNet forward z dodatkowymi residualami z adaptera
            embeddings = embeddings.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
            out = unet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=embeddings,
                down_intrablock_additional_residuals=down_residuals,
                mid_block_additional_residual=None,
                return_dict=True,
            )
            model_pred = out.sample


            # 6) Loss: MSE do prawdziwego szumu (epsilon prediction)
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = [p
                    for group in optimizer.param_groups
                    for p in group["params"]
                    if p.requires_grad and p.grad is not None]
                if params_to_clip:
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1
        if accelerator.is_main_process and global_step % 20 == 0:
            print_vram(f"[epoch {epoch} step {global_step}]")
        if accelerator.is_main_process and global_step % 50 == 0:
            print(f"[epoch {epoch}] step {global_step} loss {loss.item():.4f}")

    if accelerator.is_main_process:
        os.makedirs(OUT_DIR, exist_ok=True)
        # zapisz adapter + (ew.) lory
        if USE_ADAPTER:
            torch.save(accelerator.unwrap_model(adapter).state_dict(), os.path.join(OUT_DIR, "tiny_adapter.pt"))
        if USE_UNET_LORA:
            accelerator.unwrap_model(unet).save_pretrained(os.path.join(OUT_DIR, "unet_lora"))

# zapis końcowy
if accelerator.is_main_process:
    os.makedirs(OUT_DIR, exist_ok=True)
    if USE_ADAPTER:
        torch.save(accelerator.unwrap_model(adapter).state_dict(), os.path.join(OUT_DIR, "tiny_adapter.pt"))
    if USE_UNET_LORA:
        accelerator.unwrap_model(unet).save_pretrained(os.path.join(OUT_DIR, "unet_lora"))

print("Done.")
