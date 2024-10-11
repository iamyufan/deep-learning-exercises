from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from .components import Generator, Discriminator
from .gan import GAN


class DCGAN(GAN):
    def __init__(
        self,
        latent_dim=100,
        num_classes=10,
        img_channels=1,
        learning_rate=0.0002,
        beta1=0.2,
        num_layers_G=4,
        num_layers_D=4,
        nonlinearity_G="ReLU",
        nonlinearity_D="LeakyReLU",
    ):
        """
        The DCGAN class that combines the Generator and Discriminator.
        Follows the PyTorch Lightning Module structure that wraps the training loop.

        Parameters
        ----------
        latent_dim : int
            The dimension of the latent random noise vector.
        num_classes : int
            The number of classes in the dataset, used for label supervision.
        img_channels : int
            The number of channels in the input images.
        learning_rate : float
            The learning rate for the optimizer.
        beta1 : float
            The beta1 parameter for the Adam optimizer.
        """
        super().__init__(
            latent_dim=latent_dim,
            num_classes=num_classes,
            img_channels=img_channels,
        )
        # Generator and Discriminator
        self.netG = Generator(
            latent_dim=latent_dim,
            num_classes=num_classes,
            img_channels=img_channels,
            num_layers=num_layers_G,
            nonlinearity=nonlinearity_G,
        ).to(self.device)
        self.netD = Discriminator(
            num_classes=num_classes,
            img_channels=img_channels,
            num_layers=num_layers_D,
            nonlinearity=nonlinearity_D,
        ).to(self.device)

        # Training configurations
        self.learning_rate = learning_rate
        self.beta1 = beta1

    @property
    def criterion(self):
        if not hasattr(self, "_criterion"):
            self._criterion = nn.BCELoss()
        return self._criterion

    @property
    def optimizerG(self):
        if not hasattr(self, "_optimizerG"):
            self._optimizerG = optim.Adam(
                self.netG.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999)
            )
        return self._optimizerG

    @property
    def optimizerD(self):
        if not hasattr(self, "_optimizerD"):
            self._optimizerD = optim.Adam(
                self.netD.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999)
            )
        return self._optimizerD

    def training_step(
        self,
        batch: Tuple[torch.Tensor],
        batch_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run a single training step on a batch of data and
        return the losses of the Generator and Discriminator.

        Parameters
        ----------
        batch : Tuple[torch.Tensor]
            A tuple containing
            - input images (batch_size, img_channels, 64, 64) and
            - labels (batch_size).
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        errG : torch.Tensor
            The loss of the Generator.
        errD : torch.Tensor
            The loss of the Discriminator.
        """
        real_images, class_labels = batch
        real_images = real_images.to(self.device)
        class_labels = class_labels.to(self.device)
        batch_size = real_images.size(0)

        # Labels for real and fake data
        real_labels = torch.ones(batch_size, device=self.device)
        fake_labels = torch.zeros(batch_size, device=self.device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.netD.zero_grad()

        # Real images
        output_real = self.netD(real_images, class_labels).view(-1)
        errD_real = self.criterion(output_real, real_labels)

        # Fake images
        noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        fake_images = self.netG(noise, class_labels)
        output_fake = self.netD(fake_images.detach(), class_labels).view(-1)
        errD_fake = self.criterion(output_fake, fake_labels)

        # Total Discriminator loss
        errD = errD_real + errD_fake
        errD.backward()
        self.optimizerD.step()

        # ---------------------
        #  Train Generator
        # ---------------------
        self.netG.zero_grad()
        output_fake = self.netD(fake_images, class_labels).view(-1)
        errG = self.criterion(output_fake, real_labels)
        errG.backward()
        self.optimizerG.step()

        return errG, errD
