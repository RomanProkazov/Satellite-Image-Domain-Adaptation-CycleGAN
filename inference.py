import torch
from torchvision import transforms
from PIL import Image
from generator_model import Generator
import config
import os
from matplotlib import pyplot as plt


def load_generator(checkpoint_path, device):
    model = Generator(img_channels=3, num_residuals=9).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


def postprocess_tensor(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = (tensor * 0.5) + 0.5  # Unnormalize
    return transforms.ToPILImage()(tensor)


@torch.no_grad()
def run_inference_image(image_path, generator, output_path=None):
    input_image = preprocess_image(image_path).to(config.DEVICE)
    fake_image = generator(input_image)

    # Convert to PIL image
    result = postprocess_tensor(fake_image)

    # Save or show
    if output_path:
        result.save(output_path)
        print(f"Saved translated image to {output_path}")
    else:
        result.show()


def run_inference_folder(input_folder, output_folder, generator):
    for filename in os.listdir(input_folder)[:100]:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            run_inference_image(input_path, generator, output_path)


def run_cycle_on_image(image_path, gen_Z, gen_H):
    input_image = preprocess_image(image_path).to(config.DEVICE)
    fake_image = gen_Z(input_image)
    cycle_image = gen_H(fake_image)
    result_fake = postprocess_tensor(fake_image)
    result_cycle = postprocess_tensor(cycle_image)
    input_image_orig = postprocess_tensor(input_image)

    return input_image_orig, result_fake, result_cycle


def visualize_cycle(orig, fake, rec, title=None, save_path=None):
    """Plot a 1×3 row and optionally save."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, lbl in zip(
        axes, (orig, fake, rec),
        ("Original", "Fake", "Reconstructed")
    ):
        ax.imshow(img); ax.set_title(lbl)
        ax.axis("off")

    if title:
        fig.suptitle(title, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    input_image_path_Z = config.Z_VAL_IMG_PATH
    input_image_path_H = config.H_VAL_IMG_PATH
    input_folder = config.H_VAL_DIR
    gen_H = load_generator(config.CHECKPOINT_GEN_H, config.DEVICE)
    gen_Z = load_generator(config.CHECKPOINT_GEN_Z, config.DEVICE)

    # run_inference_image(input_image_path, gen_H, output_path=None)
    # run_inference_folder(input_folder, config.Z_INFERENCE_DIR, gen_Z)
    img_orig, img_fake, img_rec = run_cycle_on_image(input_image_path_H, gen_Z, gen_H)
    visualize_cycle(img_orig, img_fake, img_rec)
