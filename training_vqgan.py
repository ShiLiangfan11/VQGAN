import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import load_data, weights_init


class TrainVQGAN:
    def __init__(self, args):
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir="logs")

        # Load pre-trained model if available
        if args.pretrained_model_path:
            self.load_pretrained_model(args.pretrained_model_path)

        self.prepare_training()
        self.train(args)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr, eps=1e-08, betas=(args.beta1, args.beta2))

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def load_pretrained_model(self, pretrained_model_path):
        """Load pre-trained weights for VQGAN and Discriminator."""
        if os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path, map_location=torch.device(args.device))
            
            # Load VQGAN model
            self.vqgan.load_state_dict(checkpoint['vqgan_state_dict'])
            print(f"Loaded VQGAN model from {pretrained_model_path}")
            
            # Optionally, load discriminator model if required
            if 'discriminator_state_dict' in checkpoint:
                self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                print(f"Loaded Discriminator model from {pretrained_model_path}")
            
            # Optionally, load optimizers' state if you want to resume training
            if 'optimizer_vq_state_dict' in checkpoint:
                self.opt_vq.load_state_dict(checkpoint['optimizer_vq_state_dict'])
                print("Loaded VQGAN optimizer state.")
                
            if 'optimizer_disc_state_dict' in checkpoint:
                self.opt_disc.load_state_dict(checkpoint['optimizer_disc_state_dict'])
                print("Loaded Discriminator optimizer state.")
                
            # Optionally load the epoch if you are resuming training
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                print(f"Resuming training from epoch {start_epoch}.")
            else:
                start_epoch = 0
        else:
            print(f"No pre-trained model found at {pretrained_model_path}. Starting from scratch.")
            start_epoch = 0

        return start_epoch

    def train(self, args):
        # Load data
        train_dataset = load_data(args)
        validation_size = len(train_dataset) - args.test_images_count  # Subtract the test images
        train_data, val_data = random_split(train_dataset, [validation_size, args.test_images_count])

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

        steps_per_epoch = len(train_loader)
        
        for epoch in range(args.epochs):
            self.vqgan.train()
            self.discriminator.train()

            # Training loop
            with tqdm(range(steps_per_epoch)) as pbar:
                for i, imgs in enumerate(train_loader):
                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(args.disc_factor, epoch * steps_per_epoch + i, threshold=args.disc_start)

                    perceptual_loss = self.perceptual_loss(imgs, decoded_images)
                    rec_loss = torch.abs(imgs - decoded_images)
                    perceptual_rec_loss = args.perceptual_loss_factor * perceptual_loss + args.rec_loss_factor * rec_loss
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    if i % 500 == 0:
                        self.test_and_log(val_loader, args, epoch, i)  # Run test every 500 steps

                    pbar.set_postfix(
                        VQ_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 5),
                        GAN_Loss=np.round(gan_loss.cpu().detach().numpy().item(), 3)
                    )
                    pbar.update(0)

            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'vqgan_state_dict': self.vqgan.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'optimizer_vq_state_dict': self.opt_vq.state_dict(),
                'optimizer_disc_state_dict': self.opt_disc.state_dict(),
            }
            torch.save(checkpoint, os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt"))

    def test_and_log(self, val_loader, args, epoch, step):
        # Switch to evaluation mode
        self.vqgan.eval()
        self.discriminator.eval()

        with torch.no_grad():
            # Get the first `args.test_images_count` images from validation data
            imgs = next(iter(val_loader))[:args.test_images_count].to(device=args.device)
            decoded_images, _, _ = self.vqgan(imgs)

            # Save image to TensorBoard
            real_fake_images = torch.cat((imgs[:4], decoded_images.add(1).mul(0.5)[:4]))  # Save first 4 real + fake images
            self.writer.add_image(f"Validation/epoch_{epoch}_step_{step}", real_fake_images, global_step=step)

            # Optionally, log validation loss (e.g., perceptual loss)
            val_loss = self.perceptual_loss(imgs, decoded_images).mean()
            self.writer.add_scalar('Validation/Perceptual_Loss', val_loss.item(), global_step=step)

            print(f"Validation at step {step}: Perceptual Loss = {val_loss.item():.4f}")

        # Switch back to training mode
        self.vqgan.train()
        self.discriminator.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=16384, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images (default: 3)')
    parser.add_argument('--dataset-path', type=str, default='/data', help='Path to data (default: /data)')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--batch-size', type=int, default=6, help='Input batch size for training (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate for optimizers')
    parser.add_argument('--test-images-count', type=int, default=16, help='Number of images to test every 500 steps')
    parser.add_argument('--pretrained-model-path', type=str, default='', help='Path to pre-trained model')
    parser.add_argument('--disc-factor', type=float, default=0.2, help='Factor to scale discriminator loss')
    parser.add_argument('--disc-start', type=int, default=5000, help='Step to start applying the discriminator loss')
    parser.add_argument('--perceptual-loss-factor', type=float, default=0.1, help='Factor for perceptual loss')
    parser.add_argument('--rec-loss-factor', type=float, default=0.9, help='Factor for reconstruction loss')

    args = parser.parse_args()

    trainer = TrainVQGAN(args)
