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
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
)
from diffusers.schedulers import DDPMScheduler
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import torch.nn as nn

from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator

from tqdm import tqdm

import cv2
def canny_edges(pil_rgb_512: Image.Image) -> Image.Image:
    arr = np.array(pil_rgb_512)
    edges = cv2.Canny(arr, 100, 200)
    edges = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges)

# === Konfiguracja ścieżek ===
VANILLA = "data/vanilla/1.21.4/assets/minecraft/textures/block"
STYLED  = "data/styled/Faithful 32x - 1.21.4 Experimental/assets/minecraft/textures/block"

MODEL_BASE   = "segmind/tiny-sd"
CONTROL_BASE = "lllyasviel/sd-controlnet-canny"

OUTPUT_DIR_UNET  = "output/dora_unet"
OUTPUT_DIR_CTRL  = "output/dora_controlnet"   # opcjonalnie, jeśli trenowany

SEED = 42
TRAIN_SIZE = 512           # trenowanie w 512 dla stabilności SD1.x
BATCH = 1
EPOCHS = 3
LR = 1e-4
RANK = 8
USE_DORA = True      # jeśli Twoje diffusers/peft nie wspiera, ustaw False
TRAIN_CONTROLNET = False  # True jeśli chcesz też DoRA na ControlNet (wolniej, więcej VRAM)
PROMPT = "Minecraft block converted to Faithful style"

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# === Dataset parujący pliki ===
class TexturePairs(Dataset):
    def __init__(self, vanilla_root: str, styled_root: str):
        self.pairs: List[Tuple[str,str]] = []
        for root, _, files in os.walk(vanilla_root):
            for f in files:
                if not f.lower().endswith(".png"):
                    continue
                vpath = os.path.join(root, f)
                rel   = os.path.relpath(vpath, vanilla_root)
                spath = os.path.join(styled_root, rel)
                if os.path.exists(spath):
                    self.pairs.append((vpath, spath))
        if not self.pairs:
            raise RuntimeError("Nie znaleziono par (vanilla vs styled). Sprawdź ścieżki.")

    def __len__(self): 
        return len(self.pairs)

    def __getitem__(self, idx):
        vpath, spath = self.pairs[idx]

        # Wczytaj vanilla i styled jako RGBA (żeby NIE zgubić kolorów przy alfa=0)
        v_img = Image.open(vpath).convert("RGBA")
        s_img = Image.open(spath).convert("RGBA")

        # RGB trzymamy "as-is" (odrzucamy alfa, ale kolory pod alfa=0 zostają)
        v_rgb = Image.fromarray(np.array(v_img)[:, :, :3])
        s_rgb = Image.fromarray(np.array(s_img)[:, :, :3])
        s_a   = Image.fromarray(np.array(s_img)[:, :, 3])  # alfa przydaje się przy ew. walidacji/zapisie

        # Krawędzie z vanilla (ControlNet canny)
        ctrl  = canny_edges(v_rgb)

        # Na tensory
        v_rgb = torch.from_numpy(np.array(v_rgb)).float().permute(2,0,1) / 255.0     # [3,H,W] w [0,1]
        s_rgb = torch.from_numpy(np.array(s_rgb)).float().permute(2,0,1) / 255.0
        s_a   = torch.from_numpy(np.array(s_a)).float().unsqueeze(0) / 255.0
        ctrl  = torch.from_numpy(np.array(ctrl)).float().permute(2,0,1) / 255.0

        # Normalizacja jak w diffusers: [-1,1]
        s_rgb = s_rgb * 2.0 - 1.0
        ctrl  = ctrl  * 2.0 - 1.0

        return {
            "vanilla_rgb": v_rgb,     # nie używamy bezpośrednio w tej wersji treningu (bo sterujemy przez ctrl)
            "styled_rgb":  s_rgb,
            "styled_a":    s_a,
            "control":     ctrl,
            "path":        vpath,
        }

# === Inicjalizacja modeli/pipeline ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

controlnet = ControlNetModel.from_pretrained(CONTROL_BASE, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    MODEL_BASE, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=False
)

vae: AutoencoderKL       = pipe.vae
tokenizer: CLIPTokenizer = pipe.tokenizer
text_encoder: CLIPTextModel = pipe.text_encoder
unet = pipe.unet
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="scaled_linear")

pipe.enable_vae_slicing()
pipe.enable_attention_slicing()

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
pipe.controlnet.requires_grad_(False)
unet.requires_grad_(False)

# cele LoRA/DoRA w UNet SD1.x:
LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]

# Jeśli chcesz też ControlNet – zwykle te same cele:
CTRL_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]


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

if TRAIN_CONTROLNET:
    pipe.controlnet.requires_grad_(True)  # włączamy, bo będziemy dokładać adaptery
    peft_ctrl_cfg = LoraConfig(
        r=RANK,
        lora_alpha=RANK,
        lora_dropout=0.0,
        use_dora=USE_DORA,
        target_modules=CTRL_TARGETS,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    controlnet = get_peft_model(pipe.controlnet, peft_ctrl_cfg)  # nadpisz wskaźnik lokalny
else:
    controlnet = pipe.controlnet  # alias, żeby niżej było spójnie

def trainable_params(module: nn.Module):
    return [p for p in module.parameters() if p.requires_grad]

params = []
params += trainable_params(unet)
if TRAIN_CONTROLNET:
    params += trainable_params(controlnet)

optimizer = torch.optim.AdamW(params, lr=LR)

# === Dataloader ===
ds = TexturePairs(VANILLA, STYLED)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

# === Accelerator (fp16) ===
accelerator = Accelerator(mixed_precision="fp16")
unet, controlnet, text_encoder, vae, optimizer, dl = accelerator.prepare(
    unet, controlnet, text_encoder, vae, optimizer, dl
)

trainable = [p for p in unet.parameters() if p.requires_grad]
if TRAIN_CONTROLNET:
    trainable += [p for p in controlnet.parameters() if p.requires_grad]

unet.train()
if TRAIN_CONTROLNET:
    controlnet.train()

# === Trening ===
global_step = 0
for epoch in range(EPOCHS):
    for batch in tqdm(dl):
        with accelerator.accumulate(unet):
            # 1) Tekst -> embedding
            input_ids = tokenizer(
                [PROMPT] * batch["styled_rgb"].shape[0],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)
            encoder_hidden_states = text_encoder(input_ids)[0]

            # 2) Obraz docelowy (styled) -> latenty VAE
            styled_rgb = batch["styled_rgb"].to(accelerator.device, dtype=torch.float16)
            control = batch["control"].to(accelerator.device, dtype=torch.float16)

            if TRAIN_SIZE is not None and TRAIN_SIZE != styled_rgb.shape[-1]:
                styled_rgb = F.interpolate(styled_rgb, size=(TRAIN_SIZE, TRAIN_SIZE), mode="nearest")
                control    = F.interpolate(control, size=(TRAIN_SIZE, TRAIN_SIZE), mode="nearest")
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

            # 5) ControlNet forward -> residuals
            down_samples, mid_sample = controlnet(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=control,
                return_dict=False
            )

            down_samples    = [d.to(dtype=noisy_latents.dtype) for d in down_samples]
            mid_sample      = mid_sample.to(dtype=noisy_latents.dtype)

            # 6) UNet z residualami z ControlNet
            model_pred = unet.base_model(
                sample=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample,
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
        if accelerator.is_main_process and global_step % 50 == 0:
            print(f"[epoch {epoch}] step {global_step} loss {loss.item():.4f}")

    # checkpoint po każdej epoce
    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR_UNET, exist_ok=True)
        accelerator.unwrap_model(unet).save_pretrained(OUTPUT_DIR_UNET)
        if TRAIN_CONTROLNET:
            os.makedirs(OUTPUT_DIR_CTRL, exist_ok=True)
            accelerator.unwrap_model(controlnet).save_pretrained(OUTPUT_DIR_CTRL)

# zapis końcowy (analogicznie)
if accelerator.is_main_process:
    os.makedirs(OUTPUT_DIR_UNET, exist_ok=True)
    accelerator.unwrap_model(unet).save_pretrained(OUTPUT_DIR_UNET)
    if TRAIN_CONTROLNET:
        os.makedirs(OUTPUT_DIR_CTRL, exist_ok=True)
        accelerator.unwrap_model(controlnet).save_pretrained(OUTPUT_DIR_CTRL)


print("Done. Adapters saved.")
