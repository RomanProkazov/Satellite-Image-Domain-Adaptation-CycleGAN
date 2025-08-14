import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import config
from matplotlib import pyplot as plt


class HorseZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, transform=None, pad_color=(0, 0, 0)):
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform
        self.pad_color = pad_color  # RGB tuple for padding color

        self.zebra_images = os.listdir(root_zebra)[:2000]
        self.horse_images = os.listdir(root_horse)[:2000]
        self.length_dataset = max(len(self.zebra_images), len(self.horse_images))
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

    def __len__(self):
        return self.length_dataset
    
    def __getitem__(self, index):
        zebra_image = self.zebra_images[index % self.zebra_len]
        horse_image = self.horse_images[index % self.horse_len]

        zebra_path = os.path.join(self.root_zebra, zebra_image)
        horse_path = os.path.join(self.root_horse, horse_image)

        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))

        # Apply square padding
        zebra_img = self._pad_to_square(zebra_img)
        horse_img = self._pad_to_square(horse_img)

        if self.transform:
            augumentations = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augumentations["image"]
            horse_img = augumentations["image0"]

        return zebra_img, horse_img

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
    

if __name__ == "__main__":
    dataset = HorseZebraDataset(
        root_zebra=config.SL_DIR_SPEED,
        root_horse=config.SYNTH_DIR_SPEED,
        transform=config.transforms,
    )
    # print(len(dataset))
    # exit()
    sample_zebra, sample_horse = dataset[2]

    # Convert tensors to numpy arrays if necessary
    if isinstance(sample_zebra, np.ndarray):
        zebra_img = sample_zebra
        horse_img = sample_horse
    else:
        zebra_img = sample_zebra.cpu().numpy().transpose(1, 2, 0)
        horse_img = sample_horse.cpu().numpy().transpose(1, 2, 0)

    # Denormalize images for visualization
    zebra_img = (zebra_img * 0.5 + 0.5) * 255
    horse_img = (horse_img * 0.5 + 0.5) * 255

    # Plot images using matplotlib
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Sunlamp")
    plt.imshow(zebra_img.astype(np.uint8))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Synthetic")
    plt.imshow(horse_img.astype(np.uint8))
    plt.axis("off")

    plt.show()