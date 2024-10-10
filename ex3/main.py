import os
import time
import torch
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

from datasets.fashion_mnist import FashionMNISTDataModule
from models import DCGAN, WGAN


# Hyperparameters
model_name = "dcgan"  # Choose between "dcgan" and "wgan"

batch_size = 128
image_size = 64  # Resize images to 64x64 as per DCGAN requirements
latent_dim = 100  # Latent vector size (generator input)
img_channels = 1  # Number of image channels (FashionMNIST is grayscale)
num_classes = 10  # FashionMNIST has 10 classes
learning_rate = 0.0002  # Learning rate
beta1 = 0.5  # Beta1 hyperparameter for Adam optimizer
epochs = 20  # Number of training epochs
n_critic = 5  # Number of critic iterations per generator iteration
weight_clip_value = 0.01  # Weight clipping value for WGAN


def create_model(model_name):
    if model_name == "dcgan":
        model = DCGAN(
            latent_dim=latent_dim,
            num_classes=num_classes,
            img_channels=img_channels,
            learning_rate=learning_rate,
            beta1=beta1,
        )
    elif model_name == "wgan":
        model = WGAN(
            latent_dim=latent_dim,
            num_classes=num_classes,
            img_channels=img_channels,
            learning_rate=learning_rate,
            n_critic=n_critic,
            weight_clip_value=weight_clip_value,
        )
    else:
        raise ValueError("Invalid model name")
    return model


def main():
    print(f"Training {model_name.upper()} on FashionMNIST dataset")
    print("===============================================")
    data_module = FashionMNISTDataModule(batch_size=batch_size, image_size=image_size)
    data_module.setup()
    model = create_model(model_name)
    print(f"Using {model.device} for training")

    # Train the model
    train_loader = data_module.train_dataloader()

    # Fixed noise for generating images with specific labels during testing
    fixed_noise = torch.randn(100, latent_dim, 1, 1).to(model.device)
    fixed_labels = torch.arange(0, 10).repeat(10).to(model.device)

    # Lists to store losses
    G_losses = []
    D_losses = []

    # Make output and checkpoints directories if they don't exist
    os.makedirs("output", exist_ok=True)
    os.makedirs(f"output/{model_name}", exist_ok=True)
    os.makedirs(f"checkpoints/{model_name}", exist_ok=True)

    for epoch in range(epochs):
        epoch_G_losses = []
        epoch_D_losses = []
        # Training steps
        for i, batch in enumerate(train_loader):
            errG, errD = model.training_step(batch, i)
            epoch_G_losses.append(errG.item())
            epoch_D_losses.append(errD.item())
        G_losses += epoch_G_losses
        D_losses += epoch_D_losses
        print(
            f"Epoch [{epoch+1}/{epochs}] \t G_loss: {sum(epoch_G_losses)/len(epoch_G_losses):.4f} \t D_loss: {sum(epoch_D_losses)/len(epoch_D_losses):.4f}"
        )

        # Save generated images after each epoch (with fixed labels)
        with torch.no_grad():
            fake_images = model.forward(fixed_noise, fixed_labels).detach().cpu()
        fake_images = make_grid(fake_images, nrow=10, normalize=True)
        save_image(fake_images, f"output/{model_name}/images_epoch_{epoch+1}.png")

        # Save model checkpoints for each epoch
        torch.save(
            model.netG.state_dict(),
            f"checkpoints/{model_name}/netG_epoch_{epoch+1}.pth",
        )
        torch.save(
            model.netD.state_dict(),
            f"checkpoints/{model_name}/netD_epoch_{epoch+1}.pth",
        )

    # Save losses plot as two subplots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(G_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output/{model_name}/losses.png")
    plt.show()


if __name__ == "__main__":
    # Record the time taken for everything
    start_time = time.time()
    main()
    # Print the time taken in minutes plus seconds
    elapsed_time = time.time() - start_time
    elapsed_mins = int(elapsed_time // 60)
    elapsed_secs = int(elapsed_time % 60)
    print(f"Training took {elapsed_mins} minutes and {elapsed_secs} seconds.")
