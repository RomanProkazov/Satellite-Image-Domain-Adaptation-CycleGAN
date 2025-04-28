import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 16
NUM_EPOCHS = 400
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "saved_models/gens/genh_v3s.pth.tar"
CHECKPOINT_GEN_Z = "saved_models/gens/genz_v3s.pth.tar"
CHECKPOINT_CRITIC_H = "saved_models/discrs/disch_v3s.pth.tar"
CHECKPOINT_CRITIC_Z = "saved_models/discrs/discz_v3s.pth.tar"

# directories for training and validation zebras and horses
H_TRAIN_DIR = "horse2zebra/horse2zebra/train/horses"
Z_TRAIN_DIR = "horse2zebra/horse2zebra/train/zebras"
H_VAL_DIR = "horse2zebra/horse2zebra/val/horses"
Z_VAL_DIR = "horse2zebra/horse2zebra/val/zebras"

# validation image paths
H_VAL_IMG_PATH = "/home/roman/Desktop/LUXEMBOURG PROJECT/speedplusv2/synthetic/images/img003352.jpg"
Z_VAL_IMG_PATH = "horse2zebra/horse2zebra/val/zebras/n02391049_180.jpg"

# directories SPEED+ horses = synthetic, zebras = lightbox
SYNTH_DIR_SPEED = "/home/roman/Desktop/LUXEMBOURG PROJECT/speedplusv2/synthetic/images" 
SL_DIR_SPEED = "/home/roman/Desktop/LUXEMBOURG PROJECT/speedplusv2/sunlamp/images"
LB_DIR_SPEED = "/home/roman/Desktop/LUXEMBOURG PROJECT/speedplusv2/lightbox/images"




# inference directories
H_INFERENCE_DIR = "inference_images/horses"
Z_INFERENCE_DIR = "inference_images/zebras"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)