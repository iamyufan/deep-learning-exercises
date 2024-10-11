from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from .components import Generator, Critic
from .gan import GAN


class WGAN(GAN):
    def __init__(
        self,
        latent_dim=100,
        num_classes=10,
        img_channels=1,
        learning_rate=0.0002,
        n_critic=5,
        weight_clip_value=0.01,
        num_layers_G=4,
        num_layers_D=4,
        nonlinearity_G="ReLU",
        nonlinearity_D="LeakyReLU",
        norm_layer_D="none",
    ):
        """
        The WGAN class that combines the Generator and Critic.
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
        n_critic : int
            The number of times to update the Critic per Generator update.
        weight_clip_value : float
            The value to clip the weights of the Critic.
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
        self.netD = Critic(
            num_classes=num_classes,
            img_channels=img_channels,
            num_layers=num_layers_D,
            nonlinearity=nonlinearity_D,
            norm_layer=norm_layer_D,
        ).to(self.device)

        # Training configurations
        self.learning_rate = learning_rate
        self.n_critic = n_critic
        self.weight_clip_value = weight_clip_value

    @property
    def optimizerG(self):
        if not hasattr(self, "_optimizerG"):
            self._optimizerG = optim.RMSprop(
                self.netG.parameters(), lr=self.learning_rate
            )
        return self._optimizerG

    @property
    def optimizerD(self):
        if not hasattr(self, "_optimizerD"):
            self._optimizerD = optim.RMSprop(
                self.netD.parameters(), lr=self.learning_rate
            )
        return self._optimizerD

    def training_step(
        self,
        batch: Tuple[torch.Tensor],
        batch_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run a single training step on a batch of data and
        return the losses of the Generator and Critic.

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
        #  Train Critic
        # ---------------------
        for _ in range(self.n_critic):
            # Generate random noise
            noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
            fake_images = self.netG(noise, class_labels)

            # Compute critic loss
            C_real = self.netD(real_images, class_labels).mean()
            C_fake = self.netD(fake_images, class_labels).mean()
            C_loss = -(C_real - C_fake)

            # Update Critic
            self.optimizerD.zero_grad()
            C_loss.backward()
            self.optimizerD.step()

            # Clip the weights of the Critic
            for p in self.netD.parameters():
                p.data.clamp_(-self.weight_clip_value, self.weight_clip_value)

        # ---------------------
        #  Train Generator
        # ---------------------
        self.netG.zero_grad()
        noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        fake_images = self.netG(noise, class_labels)
        errG = -self.netD(fake_images, class_labels).mean()
        self.optimizerG.step()

        return errG, C_loss
