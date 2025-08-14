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
    def __init__(self, root_real, root_synthetic, transform=None, pad_color=(0, 0, 0), original_size=(720, 1280)):
        self.root_real = root_real
        self.root_synthetic = root_synthetic
        self.transform = transform
        self.pad_color = pad_color  # RGB tuple for padding color
        self.original_size = original_size  # (height, width) of original images

        self.real_images = os.listdir(root_real)[:2000]
        self.synthetic_images = os.listdir(root_synthetic)[:2000]
        self.length_dataset = max(len(self.real_images), len(self.synthetic_images))
        self.real_len = len(self.real_images)
        self.synthetic_len = len(self.synthetic_images)

        # Define default transform if none provided
        if self.transform is None:
            self.transform = A.Compose(
                [
                    A.Resize(height=256, width=256),
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

        # Apply square padding
        real_img = self._pad_to_square(real_img)
        synthetic_img = self._pad_to_square(synthetic_img)

        if self.transform:
            augmentations = self.transform(image=real_img, image0=synthetic_img)
            real_img = augmentations["image"]
            synthetic_img = augmentations["image0"]

        return real_img, synthetic_img

    def _pad_to_square(self, image):
        h, w = image.shape[:2]
        
        if h == w:
            return image

        # Calculate padding dimensions
        max_side = max(h, w)
        pad_top = (max_side - h) // 2
        pad_bottom = max_side - h - pad_top
        pad_left = (max_side - w) // 2
        pad_right = max_side - w - pad_left

        cv_pad_color = self.pad_color[::-1]
        
        padded_image = cv2.copyMakeBorder(
            image,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=cv_pad_color
        )
        
        return padded_image

    def _crop_to_original(self, image):
        # If image is a tensor, convert to numpy
        if not isinstance(image, np.ndarray):
            image = image.cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize image
        image = (image * 0.5 + 0.5) * 255
        image = image.astype(np.uint8)

        # Resize to padded size (1280x1280)
        padded_size = max(self.original_size)
        image = cv2.resize(image, (padded_size, padded_size), interpolation=cv2.INTER_LANCZOS4)

        # Calculate crop dimensions to restore 1280x720
        h_orig, w_orig = self.original_size
        pad_top = (padded_size - h_orig) // 2
        pad_left = (padded_size - w_orig) // 2

        # Crop to original size
        cropped_image = image[pad_top:pad_top + h_orig, pad_left:pad_left + w_orig]

        return cropped_image

# Example script to generate and save fake-real images
if __name__ == "__main__":
    # Initialize dataset
    dataset = SpacecraftDataset(
        root_real=config.SL_DIR_SPEED,
        root_synthetic=config.SYNTH_DIR_SPEED,
        transform=config.transforms,
        original_size=(720, 1280)
    )

    # Placeholder for CycleGAN model (replace with your actual model)
    # Assume generator_synthetic_to_real is the trained CycleGAN generator
    # generator_synthetic_to_real = load_cyclegan_generator("path_to_model")

    # Example: Generate fake-real image from a synthetic image
    synthetic_img, _ = dataset[2]  # Get a synthetic image
    synthetic_img = synthetic_img.unsqueeze(0)  # Add batch dimension for model

    # Generate fake-real image (placeholder)
    # fake_real_img = generator_synthetic_to_real(synthetic_img)

    # For demonstration, use the synthetic image as a placeholder for fake_real_img
    fake_real_img = synthetic_img  # Replace with actual model output

    # Crop to original 1280x720 resolution
    fake_real_img_cropped = dataset._crop_to_original(fake_real_img[0])

    # Save the generated image
    output_path = "fake_real_spacecraft.png"
    Image.fromarray(fake_real_img_cropped).save(output_path)

    # Visualize the result
    plt.figure(figsize=(10, 5))
    plt.imshow(fake_real_img_cropped)
    plt.title("Generated Fake-Real Spacecraft (1280x720)")
    plt.axis("off")
    plt.show()