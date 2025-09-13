from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
import torch
from PIL import Image

vanilla_path    = "data/vanilla/1.21.4/"
styled_path     = "data/styled/Faithful 32x - 1.21.4 Experimental/"

control = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "segmind/tiny-sd", controlnet=control, torch_dtype=torch.float16, use_safetensors=False
).to("cuda")
pipe.enable_attention_slicing()

init = Image.open(f"{vanilla_path}assets/minecraft/textures/block/acacia_log_top.png")
init.resize((512, 512), resample=Image.Resampling.NEAREST)
init.save("test.png")

canny = Image.open(f"{styled_path}assets/minecraft/textures/block/acacia_log_top.png")
target_size = canny.size

canny.resize((512, 512), resample=Image.Resampling.NEAREST)
canny.save("test2.png")

out = pipe(
    prompt="Minecraft block converted to Faithful style",
    image=init, control_image=canny,
    height=512, width=512,
    controlnet_conditioning_scale=0.8,
    strength=0.10, guidance_scale=2.0, num_inference_steps=25
).images[0]
out.save("styled.png")
