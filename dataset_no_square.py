import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config
from matplotlib import pyplot as plt

class SpacecraftDataset(Dataset):
    def __init__(self, root_real, root_synthetic, transform=None, original_size=(720, 1280), intermediate_size=(360, 640)):
        self.root_real = root_real
        self.root_synthetic = root_synthetic
        self.transform = transform
        self.original_size = original_size
        self.intermediate_size = intermediate_size

        self.real_images = os.listdir(root_real)[:540]
        self.synthetic_images = os.listdir(root_synthetic)[:540]
        self.length_dataset = max(len(self.real_images), len(self.synthetic_images))
        self.real_len = len(self.real_images)
        self.synthetic_len = len(self.synthetic_images)

        if self.transform is None:
            self.transform = A.Compose(
                [
                    A.Resize(height=360, width=640),
                    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ToTensorV2(),
                ],
                additional_targets={"image0": "image"},
            )

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        real_image = self.real_images[index % self.real_len]
        synthetic_image = self.synthetic_images[index % self.synthetic_len]

        real_path = os.path.join(self.root_real, real_image)
        synthetic_path = os.path.join(self.root_synthetic, synthetic_image)

        real_img = np.array(Image.open(real_path).convert("RGB"))
        synthetic_img = np.array(Image.open(synthetic_path).convert("RGB"))

        # Resize to 640x360 (no padding)
        h_inter, w_inter = self.intermediate_size
        real_img = cv2.resize(real_img, (w_inter, h_inter), interpolation=cv2.INTER_AREA)
        synthetic_img = cv2.resize(synthetic_img, (w_inter, h_inter), interpolation=cv2.INTER_AREA)

        if self.transform:
            augmentations = self.transform(image=real_img, image0=synthetic_img)
            real_img = augmentations["image"]
            synthetic_img = augmentations["image0"]

        return real_img, synthetic_img

    def _crop_to_original(self, image):
        if not isinstance(image, np.ndarray):
            image = image.detach().cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize
        image = (image * 0.5 + 0.5) * 255
        image = image.astype(np.uint8)

        # Resize to 1280x720 using Lanczos interpolation
        h_orig, w_orig = self.original_size
        image = cv2.resize(image, (w_orig, h_orig), interpolation=cv2.INTER_LANCZOS4)

        return image

if __name__ == "__main__":
    dataset = SpacecraftDataset(
        root_real=config.SL_DIR_SPEED,
        root_synthetic=config.SYNTH_DIR_SPEED,
        transform=config.transforms,
        original_size=(720, 1280),
        intermediate_size=(360, 640)
    )

    # Load sample images
    sample_real, sample_synthetic = dataset[0]

    # Convert tensors to numpy arrays and upscale to original size
    real_img = dataset._crop_to_original(sample_real)
    synthetic_img = dataset._crop_to_original(sample_synthetic)
    plt.imsave("recovered_img.png", synthetic_img)
    # Plot images using matplotlib
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Real")
    plt.imshow(real_img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Synthetic")
    plt.imshow(synthetic_img)
    plt.axis("off")

    plt.show()
