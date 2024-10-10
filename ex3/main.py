import os
import torch
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

from datasets.fashion_mnist import FashionMNISTDataModule
from models.dcgan import DCGAN


# Hyperparameters
batch_size = 128
image_size = 64  # Resize images to 64x64 as per DCGAN requirements
latent_dim = 100  # Latent vector size (generator input)
img_channels = 1  # Number of image channels (FashionMNIST is grayscale)
num_classes = 10  # FashionMNIST has 10 classes
learning_rate = 0.0002  # Learning rate
beta1 = 0.5  # Beta1 hyperparameter for Adam optimizer
epochs = 20  # Number of training epochs


def main():
    data_module = FashionMNISTDataModule(batch_size=batch_size, image_size=image_size)
    data_module.setup()
    model = DCGAN(
        latent_dim=latent_dim,
        num_classes=num_classes,
        img_channels=img_channels,
        learning_rate=learning_rate,
        beta1=beta1,
    )
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
    os.makedirs("checkpoints/dcgan", exist_ok=True)

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
            f"Epoch [{epoch+1}/{epochs}], G_loss: {sum(epoch_G_losses)/len(epoch_G_losses)}, D_loss: {sum(epoch_D_losses)/len(epoch_D_losses)}"
        )

        # Save generated images after each epoch (with fixed labels)
        with torch.no_grad():
            fake_images = model.forward(fixed_noise, fixed_labels).detach().cpu()
        fake_images = make_grid(fake_images, nrow=10, normalize=True)
        save_image(fake_images, f"output/dcgan_images_epoch_{epoch+1}.png")

        # Save model checkpoints for each epoch
        torch.save(
            model.netG.state_dict(), f"checkpoints/dcgan/netG_epoch_{epoch+1}.pth"
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
    plt.savefig("output/dcgan_losses.png")
    plt.show()


if __name__ == "__main__":
    main()
