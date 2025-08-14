import torch
from dataset_no_square import SpacecraftDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model_no_square import Discriminator
from generator_model_no_square import Generator


def train_fn(
    disc_real, disc_synth, gen_synth, gen_real, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    real_reals = 0
    real_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (real_img, synth_img) in enumerate(loop):
        real_img = real_img.to(config.DEVICE)
        synth_img = synth_img.to(config.DEVICE)

        # Verify input dimensions (360x640)
        assert real_img.shape[2:] == (360, 640), f"Expected 360x640, got {real_img.shape[2:]}"
        assert synth_img.shape[2:] == (360, 640), f"Expected 360x640, got {synth_img.shape[2:]}"

        # Train Discriminators
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # Generate fake images
            fake_real = gen_real(synth_img)
            fake_synth = gen_synth(real_img)
            
            # Discriminator real/fake
            D_real_real = disc_real(real_img)
            D_real_fake = disc_real(fake_real.detach())
            D_synth_real = disc_synth(synth_img)
            D_synth_fake = disc_synth(fake_synth.detach())

            # Loss calculations
            real_reals += D_real_real.mean().item()
            real_fakes += D_real_fake.mean().item()
            D_real_real_loss = mse(D_real_real, torch.ones_like(D_real_real))
            D_real_fake_loss = mse(D_real_fake, torch.zeros_like(D_real_fake))
            D_synth_real_loss = mse(D_synth_real, torch.ones_like(D_synth_real))
            D_synth_fake_loss = mse(D_synth_fake, torch.zeros_like(D_synth_fake))
            
            D_loss = (D_real_real_loss + D_real_fake_loss + D_synth_real_loss + D_synth_fake_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # Adversarial loss
            D_real_fake = disc_real(fake_real)
            D_synth_fake = disc_synth(fake_synth)
            loss_G_real = mse(D_real_fake, torch.ones_like(D_real_fake))
            loss_G_synth = mse(D_synth_fake, torch.ones_like(D_synth_fake))

            # Cycle consistency
            cycle_synth = gen_synth(fake_real)
            cycle_real = gen_real(fake_synth)
            cycle_synth_loss = l1(synth_img, cycle_synth)
            cycle_real_loss = l1(real_img, cycle_real)

            # Total loss
            G_loss = (
                loss_G_synth + loss_G_real +
                cycle_synth_loss * config.LAMBDA_CYCLE +
                cycle_real_loss * config.LAMBDA_CYCLE
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Save sample images
        if idx % 200 == 0:
            save_image(fake_real * 0.5 + 0.5, f"saved_images/real_{idx}.png")
            save_image(fake_synth * 0.5 + 0.5, f"saved_images/synth_{idx}.png")

        loop.set_postfix(real_real=real_reals/(idx+1), real_fake=real_fakes/(idx+1))

def main():
    # Initialize models
    disc_real = Discriminator(in_channels=3).to(config.DEVICE)
    disc_synth = Discriminator(in_channels=3).to(config.DEVICE)
    gen_synth = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_real = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    
    # Optimizers
    opt_disc = optim.Adam(
        list(disc_real.parameters()) + list(disc_synth.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_synth.parameters()) + list(gen_real.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    # Loss functions
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    # Load dataset
    dataset = SpacecraftDataset(
        root_real=config.SL_DIR_SPEED,
        root_synthetic=config.SYNTH_DIR_SPEED,
        transform=config.transforms,
        original_size=(720, 1280),
        intermediate_size=(360, 640)
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Training loop
    g_scaler = torch.amp.GradScaler()
    d_scaler = torch.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
        train_fn(
            disc_real,
            disc_synth,
            gen_synth,
            gen_real,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_real, opt_gen, filename=config.CHECKPOINT_GEN_REAL)
            save_checkpoint(gen_synth, opt_gen, filename=config.CHECKPOINT_GEN_SYNTH)
            save_checkpoint(disc_real, opt_disc, filename=config.CHECKPOINT_DISC_REAL)
            save_checkpoint(disc_synth, opt_disc, filename=config.CHECKPOINT_DISC_SYNTH)

if __name__ == "__main__":
    main()