from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import torchvision


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim=100,
        num_classes=10,
        img_channels=1,
    ):
        """
        The Generator class for DCGAN.

        Parameters
        ----------
        latent_dim : int
            The dimension of the latent random noise vector.
        num_classes : int
            The number of classes in the dataset, used for label supervision.
        img_channels : int
            The number of channels in the input images.
        """
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            # Input is the latent vector z + class label, going into a transposed conv layer
            nn.ConvTranspose2d(latent_dim + num_classes, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Layer 2
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Layer 3
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Layer 4
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Output layer: produces 1-channel 64x64 image
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),  # Output range should be [-1, 1]
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Generator to generate fake images.

        Parameters
        ----------
        noise : torch.Tensor (batch_size, latent_dim, 1, 1)
            The random noise vector sampled from a normal distribution.
        labels : torch.Tensor (batch_size)
            The class labels for the images.

        Returns
        -------
        torch.Tensor (batch_size, img_channels, 64, 64)
            The generated fake images.
        """
        # Concatenate noise vector z and class label embedding
        label_embedding = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        z = torch.cat([noise, label_embedding], dim=1)
        return self.main(z)


class Critic(nn.Module):
    def __init__(
        self,
        num_classes=10,
        img_channels=1,
    ):
        """
        The Critic class for WGAN-GP.

        Parameters
        ----------
        num_classes : int
            The number of classes in the dataset, used for label supervision.
        img_channels : int
            The number of channels in the input images.
        """
        super(Critic, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            # Input is the image + class label, going into a conv layer
            nn.Conv2d(img_channels + num_classes, 64, 4, 2, 1, bias=False),
            nn.LayerNorm([64, 32, 32]),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.LayerNorm([128, 16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.LayerNorm([256, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.LayerNorm([512, 4, 4]),
            nn.LeakyReLU(0.2, inplace=True),
            # Output layer
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, img, labels) -> torch.Tensor:
        label_embedding = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        label_embedding = label_embedding.expand(-1, -1, img.size(2), img.size(3))
        img = torch.cat([img, label_embedding], dim=1)
        return self.main(img).view(-1)


class WGAN_GP:
    def __init__(
        self,
        latent_dim=100,
        num_classes=10,
        img_channels=1,
        learning_rate=0.0002,
        n_critic=5,
        weight_clip_value=0.01,
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
        n_critic : int
            The number of critic iterations per generator iteration.
        weight_clip_value : float
            The value to clip the weights of the discriminator.
        """
        super(WGAN_GP, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration for the model
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.n_critic = n_critic
        self.weight_clip_value = weight_clip_value

        # Generator and Discriminator
        self.netG = Generator(self.latent_dim, self.num_classes, self.img_channels).to(
            self.device
        )
        self.netC = Critic(self.num_classes, self.img_channels).to(self.device)

        # Training configurations
        self.learning_rate = learning_rate

    @property
    def optimizerG(self):
        if not hasattr(self, "_optimizerG"):
            self._optimizerG = optim.Adam(
                self.netG.parameters(), lr=self.learning_rate, betas=(0.0, 0.9)
            )
        return self._optimizerG

    @property
    def optimizerC(self):
        if not hasattr(self, "_optimizerC"):
            self._optimizerC = optim.Adam(
                self.netC.parameters(), lr=self.learning_rate, betas=(0.0, 0.9)
            )
        return self._optimizerC

    def gradient_penalty(self, real_data, fake_data, labels):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(
            True
        )
        d_interpolates = self.netC(interpolates, labels)
        gradients = grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        return gradient_penalty

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Generator to generate fake images.

        Parameters
        ----------
        noise : torch.Tensor (batch_size, latent_dim, 1, 1)
            The random noise vector sampled from a normal distribution.
        labels : torch.Tensor (batch_size)
            The class labels for the images.

        Returns
        -------
        torch.Tensor (batch_size, img_channels, 64, 64)
            The generated fake images.
        """
        return self.netG(noise, labels)

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
        real_images, real_labels = batch
        real_images = real_images.to(self.device)
        real_labels = real_labels.to(self.device)
        batch_size = real_images.size(0)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for _ in range(self.n_critic):
            self.netD.zero_grad()

            # Train on real images
            d_real = self.netC(real_images, real_labels).mean()
            
            # Train on fake images
            noise = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
            fake_labels = torch.randint(0, self.num_classes, (batch_size,)).to(self.device)
            fake_images = self.netG(noise, fake_labels)
            d_fake = self.netC(fake_images.detach(), fake_labels).mean()
            
            # Gradient penalty
            gp = self.gradient_penalty(real_images, fake_images, real_labels)
            
            # Critic loss
            errC = -d_real + d_fake + gp
            errC.backward()
            self.optimizerC.step()
            
            output_real = self.netD(real_images, real_labels).view(-1)
            errD_real = -torch.mean(output_real)

            # Train on fake images
            noise = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
            fake_labels = torch.randint(0, self.num_classes, (batch_size,)).to(
                self.device
            )
            fake_images = self.netG(noise, fake_labels)
            output_fake = self.netD(fake_images.detach(), fake_labels).view(-1)
            errD_fake = torch.mean(output_fake)

            # Total discriminator loss
            errD = errD_real + errD_fake
            errD.backward()
            self.optimizerD.step()

            # Weight clipping
            for p in self.netD.parameters():
                p.data.clamp_(-self.weight_clip_value, self.weight_clip_value)

        # ---------------------
        #  Train Generator
        # ---------------------
        self.netG.zero_grad()

        # Generator wants to maximize the discriminator's output for fake images
        output = self.netD(fake_images, fake_labels).view(-1)
        errG = -torch.mean(output)
        errG.backward()
        self.optimizerG.step()

        return errG, errD

    def configure_optimizers(self):
        return [self.optimizerG, self.optimizerD], []

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
