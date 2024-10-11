# Create a GAN model interface
from typing import Tuple
import torchvision
import torch


class GAN:
    def __init__(self, latent_dim=100, num_classes=10, img_channels=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration for the model
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_channels = img_channels

    def training_step(
        self,
        batch: Tuple[torch.Tensor],
        batch_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs=100,
        output_dir="output",
    ):
        """
        Train the model on the given dataloader for the specified number of epochs.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The DataLoader containing the training data.
        num_epochs : int
            The number of epochs to train the model.
        """
        print(f"Training the model on {self.device}")
        
        for epoch in range(num_epochs):
            for i, batch in enumerate(dataloader):
                errG, errD = self.training_step(batch, i)

                if i % 100 == 0:
                    print(
                        f"Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}]"
                        f"\tLoss_D: {errD.item():.4f} \t Loss_G: {errG.item():.4f}"
                    )

            torchvision.utils.save_image(
                self.netG(torch.randn(64, self.latent_dim, 1, 1).to(self.device)),
                f"{output_dir}_{epoch}.png",
                normalize=True,
            )

    def generate_images_by_label(self, num_images: int, label: int) -> torch.Tensor:
        """
        Generate fake images in the format of torchvision grid
        given a class label.

        Parameters
        ----------
        num_images : int
            The number of images to generate.
        label: int
            The class label for the images.

        Returns
        -------
        grid : torch.Tensor (3, H, W)
            The grid of fake images.
        """
        noise = torch.randn(num_images, self.latent_dim, 1, 1).to(self.device)
        labels = torch.full((num_images,), label, dtype=torch.long).to(self.device)

        with torch.no_grad():
            fake_images = self.forward(noise, labels).detach().cpu()

        grid = torchvision.utils.make_grid(fake_images, nrow=10, normalize=True)
        return grid
