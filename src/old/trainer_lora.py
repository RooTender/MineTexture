# train_dora_texture.py
import os, math, itertools, random
from dataclasses import dataclass
from typing import List, Tuple
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# === Diffusers/PEFT bits ===
from diffusers import (
    AutoencoderKL,
    StableDiffusionImg2ImgPipeline
)
from diffusers.schedulers import DDPMScheduler
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import torch.nn as nn

from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
import re

from tqdm import tqdm

# === Konfiguracja ścieżek ===
VANILLA = "data/vanilla/1.21.4/assets/minecraft/textures/block"
STYLED  = "data/styled/Faithful 32x/assets/minecraft/textures/block"

MODEL_BASE   = "segmind/tiny-sd"
OUTPUT_DIR_UNET  = "output/dora_unet"

SEED = 42
TRAIN_SIZE = 512           # trenowanie w 512 dla stabilności SD1.x
BATCH = 4
EPOCHS = 3
LR = 1e-4
RANK = 8
USE_DORA = True      # jeśli Twoje diffusers/peft nie wspiera, ustaw False

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# === Dataset parujący pliki ===
class TexturePairs(Dataset):
    def __init__(self, vanilla_root: str, styled_root: str):
        self.styled_textures: List[str] = []

        for root, _, files in os.walk(vanilla_root):
            for f in files:
                if not f.lower().endswith(".png"):
                    continue

                vpath = os.path.join(root, f)
                rel   = os.path.relpath(vpath, vanilla_root)
                spath = os.path.join(styled_root, rel)

                if os.path.exists(spath):
                    self.styled_textures.append(spath)

        if not self.styled_textures:
            raise RuntimeError("Nie znaleziono par (vanilla vs styled). Sprawdź ścieżki.")

    def __len__(self): 
        return len(self.styled_textures)

    def __getitem__(self, idx):
        spath = self.styled_textures[idx]

        # Wczytaj vanilla i styled jako RGBA (żeby NIE zgubić kolorów przy alfa=0)
        s_img = Image.open(spath).convert("RGBA")
        s_img = s_img.resize((TRAIN_SIZE, TRAIN_SIZE), Image.Resampling.NEAREST)
        
        s_rgb = Image.fromarray(np.array(s_img)[:, :, :3])
        s_rgb = torch.from_numpy(np.array(s_rgb)).float().permute(2,0,1) / 255.0
        s_rgb = s_rgb * 2.0 - 1.0

        base = os.path.splitext(os.path.basename(spath))[0]
        parts = base.split("_")
        parts = [p for p in parts if not p.isdigit()]
        name  = " ".join(parts)
        name  = re.sub(r"\s+", " ", name).strip()

        return {
            "styled_rgb":  s_rgb,
            "prompt":      f"Minecraft {name} converted to Faithful style",
        }

# === Inicjalizacja modeli/pipeline ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    MODEL_BASE, torch_dtype=torch.float16, use_safetensors=False
)

vae, tokenizer, text_encoder, unet = pipe.vae, pipe.tokenizer, pipe.text_encoder, pipe.unet
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear")

pipe.enable_vae_slicing()
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

# cele LoRA/DoRA w UNet SD1.x:
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


# --- PEFT DoRA config ---
peft_unet_cfg = LoraConfig(
    r=RANK,
    lora_alpha=RANK,
    lora_dropout=0.0,
    use_dora=USE_DORA,          # <<-- wymusza DoRA
    target_modules=LORA_TARGETS,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION    # dowolne tutaj, dla plain nn.Module nie ma znaczenia
)

unet = get_peft_model(unet, peft_unet_cfg)

def trainable_params(module: nn.Module):
    return [p for p in module.parameters() if p.requires_grad]

params = []
params += trainable_params(unet)

optimizer = torch.optim.AdamW(params, lr=LR)

# === Dataloader ===
ds = TexturePairs(VANILLA, STYLED)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

# === Accelerator (fp16) ===
accelerator = Accelerator(mixed_precision="fp16")
unet, text_encoder, vae, optimizer, dl = accelerator.prepare(
    unet, text_encoder, vae, optimizer, dl
)

trainable = [p for p in unet.parameters() if p.requires_grad]

unet.train()

def print_vram(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved  = torch.cuda.memory_reserved() / 1024**2
    max_alloc = torch.cuda.max_memory_allocated() / 1024**2
    max_res   = torch.cuda.max_memory_reserved() / 1024**2
    print(f"{prefix} VRAM allocated={allocated:.0f}MB reserved={reserved:.0f}MB "
          f"(max_alloc={max_alloc:.0f}MB max_res={max_res:.0f}MB)")

# === Trening ===
global_step = 0
for epoch in range(EPOCHS):
    torch.cuda.reset_peak_memory_stats()

    for batch in tqdm(dl, desc=f"Epoch ({epoch + 1}/{EPOCHS})"):
        with accelerator.accumulate(unet):
            # 1) Tekst -> embedding
            with torch.no_grad():
                input_ids = tokenizer(
                    batch["prompt"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)
                encoder_hidden_states = text_encoder(input_ids)[0]

            # 2) Obraz docelowy (styled) -> latenty VAE
            with torch.no_grad():
                styled_rgb = batch["styled_rgb"].to(accelerator.device, dtype=torch.float16)
                latents = vae.encode(styled_rgb).latent_dist.sample() * vae.config.scaling_factor

            # 3) Dodaj szum według losowego t
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = encoder_hidden_states.to(
                device=noisy_latents.device,
                dtype=noisy_latents.dtype
            )

            # 6) UNet z residualami z ControlNet
            model_pred = unet.base_model(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=True
            ).sample

            # 7) Loss: MSE do prawdziwego szumu (epsilon prediction)
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        global_step += 1
        if accelerator.is_main_process and global_step % 20 == 0:
            print_vram(f"[epoch {epoch} step {global_step}]")

        if accelerator.is_main_process and global_step % 50 == 0:
            print(f"[epoch {epoch}] step {global_step} loss {loss.item():.4f}")

    # checkpoint po każdej epoce
    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR_UNET, exist_ok=True)
        accelerator.unwrap_model(unet).save_pretrained(OUTPUT_DIR_UNET)

# zapis końcowy (analogicznie)
if accelerator.is_main_process:
    os.makedirs(OUTPUT_DIR_UNET, exist_ok=True)
    accelerator.unwrap_model(unet).save_pretrained(OUTPUT_DIR_UNET)


print("Done. Adapters saved.")
