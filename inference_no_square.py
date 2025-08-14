import torch
from torchvision import transforms
from PIL import Image
from generator_model import Generator
import config
import os
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

def load_generator(checkpoint_path, device):
    model = Generator(img_channels=3, num_residuals=9).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def preprocess_image(image_path):
    transform = A.Compose([
        A.Resize(height=360, width=640),  # Match your training resolution
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ])
    image = np.array(Image.open(image_path).convert("RGB"))
    return transform(image=image)["image"].unsqueeze(0).to(config.DEVICE)  # Add batch dim


# def postprocess_tensor(tensor):
#     tensor = tensor.squeeze(0).detach().cpu()
#     tensor = (tensor * 0.5) + 0.5  # [-1,1] -> [0,1]
#     return transforms.ToPILImage()(tensor)


def postprocess_tensor(tensor, original_size=(720, 1280)):
    # Step 1: Convert tensor to numpy array
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = (tensor * 0.5) + 0.5  # [-1,1] -> [0,1]
    numpy_img = tensor.permute(1, 2, 0).numpy()  # (H,W,C)
    numpy_img = (numpy_img * 255).astype(np.uint8)  # [0,1] -> [0,255]
    
    # Step 2: Upscale using Lanczos
    h_orig, w_orig = original_size
    upscaled = cv2.resize(numpy_img, (w_orig, h_orig), 
                         interpolation=cv2.INTER_LANCZOS4)
    
    # Step 3: Convert back to PIL
    return Image.fromarray(upscaled)

@torch.no_grad()
def run_inference_image(image_path, generator, output_path=None):
    input_image = preprocess_image(image_path)
    fake_image = generator(input_image)
    result = postprocess_tensor(fake_image)

    if output_path:
        result.save(output_path)
        print(f"Saved translated image to {output_path}")
    else:
        plt.imshow(result)
        plt.axis('off')
        plt.show()


def run_inference_folder(input_folder, output_folder, generator):
    os.makedirs(output_folder, exist_ok=True)
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            run_inference_image(input_path, generator, output_path)


def run_cycle_on_image(image_path, gen_real, gen_synth):
    input_image = preprocess_image(image_path)
    fake_image = gen_real(input_image)
    cycle_image = gen_synth(fake_image)
    
    return (
        postprocess_tensor(input_image),  # Original
        postprocess_tensor(fake_image),   # fake
        postprocess_tensor(cycle_image)   # Reconstructed
    )


def visualize_cycle(orig, fake, rec, title=None, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, lbl in zip(
        axes, (orig, fake, rec),
        ("Synthetic", " Fake Real", "Reconstructed")
    ):
        ax.imshow(img)
        ax.set_title(lbl, fontsize=12)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig("/home/roman/cycle-gans/saved_images_reconstr_close.png", bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Update these paths in config.py:
    # REAL_VAL_DIR = "path/to/real_validation_images"
    # SYNTH_VAL_DIR = "path/to/synthetic_validation_images"
    # CHECKPOINT_GEN_REAL = "path/to/gen_real.pth.tar"
    # CHECKPOINT_GEN_SYNTH = "path/to/gen_synth.pth.tar"
    
    gen_real = load_generator(config.CHECKPOINT_GEN_REAL, config.DEVICE)
    gen_synth = load_generator(config.CHECKPOINT_GEN_SYNTH, config.DEVICE)

    # Single image test
    test_image_path = "/home/roman/spacecraft-pose-estimation-trajectories/data/images/image_01750.jpg"
    orig, fake, rec = run_cycle_on_image(test_image_path, gen_real, gen_synth)
    visualize_cycle(orig, fake, rec, "Spacecraft Domain Translation", "/home/roman/cycle-gans/saved_images-reconstr.png")

    # # Batch process folder
    # run_inference_folder(
    #     input_folder="/home/roman/spacecraft-pose-estimation-trajectories/data/images",
    #     output_folder="/home/roman/spacecraft-pose-estimation-trajectories/data/images_cyclegan",
    #     generator=gen_real
    # )