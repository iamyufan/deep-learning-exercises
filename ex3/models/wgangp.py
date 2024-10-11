from typing import Tuple
import torch
import torch.optim as optim

from .components import Generator, Critic
from .gan import GAN


class WGAN_GP(GAN):
    def __init__(
        self,
        latent_dim=100,
        num_classes=10,
        img_channels=1,
        learning_rate=0.0001,
        n_critic=5,
        lambda_gp=10.0,
        num_layers_G=4,
        num_layers_D=4,
        nonlinearity_G="ReLU",
        nonlinearity_D="LeakyReLU",
        norm_layer_D="none",
    ):
        """
        The WGAN-GP class that combines the Generator and Critic.
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
        lambda_gp : float
            The gradient penalty coefficient.
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
        self.lambda_gp = lambda_gp

    @property
    def optimizerG(self):
        if not hasattr(self, "_optimizerG"):
            self._optimizerG = optim.Adam(
                self.netG.parameters(), lr=self.learning_rate, betas=(0.0, 0.9)
            )
        return self._optimizerG

    @property
    def optimizerD(self):
        if not hasattr(self, "_optimizerD"):
            self._optimizerD = optim.Adam(
                self.netD.parameters(), lr=self.learning_rate, betas=(0.0, 0.9)
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
            The loss of the Critic.
        """
        real_images, class_labels = batch
        real_images = real_images.to(self.device)
        class_labels = class_labels.to(self.device)
        batch_size = real_images.size(0)

        # ---------------------
        #  Train Critic
        # ---------------------
        for _ in range(self.n_critic):
            # Generate random noise
            noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
            fake_images = self.netG(noise, class_labels).detach()

            # Compute critic loss
            C_real = self.netD(real_images, class_labels).mean()
            C_fake = self.netD(fake_images, class_labels).mean()
            C_loss = -(C_real - C_fake)

            # Gradient penalty
            gp = self.compute_gradient_penalty(real_images, fake_images, class_labels)
            C_loss += self.lambda_gp * gp

            # Update Critic
            self.optimizerD.zero_grad()
            C_loss.backward()
            self.optimizerD.step()

        # ---------------------
        #  Train Generator
        # ---------------------
        noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        fake_images = self.netG(noise, class_labels)
        G_loss = -self.netD(fake_images, class_labels).mean()

        self.optimizerG.zero_grad()
        G_loss.backward()
        self.optimizerG.step()

        return G_loss, C_loss

    def compute_gradient_penalty(self, real_images, fake_images, labels):
        """
        Computes the gradient penalty for WGAN-GP.

        Parameters
        ----------
        real_images : torch.Tensor
            A batch of real images.
        fake_images : torch.Tensor
            A batch of generated fake images.
        labels : torch.Tensor
            The class labels corresponding to the images.

        Returns
        -------
        gradient_penalty : torch.Tensor
            The computed gradient penalty.
        """
        batch_size = real_images.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device).expand_as(
            real_images
        )
        interpolates = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(
            True
        )

        critic_interpolates = self.netD(interpolates, labels)

        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(critic_interpolates.size(), device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
