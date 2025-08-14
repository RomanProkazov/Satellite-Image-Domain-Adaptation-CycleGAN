import torch
import numpy as np
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import warnings
warnings.filterwarnings('ignore')

# --- Load Model ---
model_path = "pretrained_models/realesrgan/RealESRGAN_x4plus.pth"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))['params_ema']

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
model.load_state_dict(state_dict, strict=True)

upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    pre_pad=0,
    half=True  # Use FP16 for faster processing (if supported)
)

# --- Load & Process Image ---
img = Image.open('recovered_img.png').convert('RGB')
img = np.array(img)  # Shape: (H, W, 3)

# Upscale with RealESRGAN (e.g., 4×)
output, _ = upsampler.enhance(img, outscale=4)  # Output shape: (4H, 4W, 3)

# --- Resize to 1280x720 (if needed) ---
target_width, target_height = 1280, 720

# If RealESRGAN output is larger than 1280x720, downsample
if output.shape[0] > target_height or output.shape[1] > target_width:
    output = Image.fromarray(output)
    output = output.resize((target_width, target_height), Image.LANCZOS)  # High-quality downscaling
else:
    output = Image.fromarray(output)

# --- Save ---
output.save("recovered_image_enhanced_1280x720.png")
print(f"Saved enhanced image at {target_width}x{target_height} resolution.")