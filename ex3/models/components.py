import torch
import torch.nn as nn


# Function to select nonlinearity
def get_nonlinearity(name) -> nn.Module:
    if name == "ReLU":
        return nn.ReLU(True)
    elif name == "LeakyReLU":
        return nn.LeakyReLU(0.2, inplace=True)
    elif name == "softplus":
        return lambda x: (torch.nn.functional.softplus(2 * x + 2) / 2) - 1
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Invalid nonlinearity selection.")


# Function to select normalization layer
def get_norm_layer(norm_layer, num_features, img_shape) -> nn.Module:
    if norm_layer == "bn":
        return nn.BatchNorm2d(num_features)
    elif norm_layer == "ln":
        return nn.LayerNorm([num_features, img_shape[0], img_shape[1]])
    elif norm_layer == "none":
        return nn.Identity()
    else:
        raise ValueError("Invalid normalization layer selection.")


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim=100,
        num_classes=10,
        img_channels=1,
        num_layers=4,
        nonlinearity="ReLU",
    ):
        """
        The Generator class for DCGAN, WGAN, and WGAN-GP.

        Parameters
        ----------
        latent_dim : int
            The dimension of the latent random noise vector.
        num_classes : int
            The number of classes in the dataset, used for label supervision.
        img_channels : int
            The number of channels in the input images.
        num_layers : int
            The number of layers in the Generator, not including the output layer.
        nonlinearity : str
            The nonlinearity to use in the Generator.
        """
        super().__init__()
        self.nonlinearity = nonlinearity
        self.input_dim = latent_dim + num_classes

        # The embedding layer for the class labels
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # Get the main body of the Generator
        if num_layers == 4:
            main = self._get_4_layer()
        elif num_layers == 8:
            main = self._get_8_layer()
        else:
            raise ValueError("Invalid number of layers for Generator.")

        # Output layer: produces 1-channel 64x64 image in [-1, 1]
        main.add_module(
            "output",
            nn.Sequential(
                nn.ConvTranspose2d(64, img_channels, 4, 2, 1, bias=False),
                nn.Tanh(),
            ),
        )

        self.main = nn.Sequential(*main)

    def _get_4_layer(self) -> nn.Sequential:
        main = nn.Sequential()

        # Layer 1: (1x1) -> (4x4)
        main.add_module(
            "block1",
            nn.Sequential(
                nn.ConvTranspose2d(self.input_dim, 512, 4, 1, 0, bias=False),
                nn.BatchNorm2d(512),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 2: (4x4) -> (8x8)
        main.add_module(
            "block2",
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 3: (8x8) -> (16x16)
        main.add_module(
            "block3",
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 4: (16x16) -> (32x32)
        main.add_module(
            "block4",
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                get_nonlinearity(self.nonlinearity),
            ),
        )

        return main

    def _get_8_layer(self) -> nn.Sequential:
        main = nn.Sequential()

        # Layer 1: (1x1) -> (4x4)
        main.add_module(
            "block1",
            nn.Sequential(
                nn.ConvTranspose2d(self.input_dim, 512, 4, 1, 0, bias=False),
                nn.BatchNorm2d(512),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 2: (4x4) -> (4x4)
        main.add_module(
            "block2",
            nn.Sequential(
                nn.ConvTranspose2d(512, 512, 3, 1, 1, bias=False),
                nn.BatchNorm2d(512),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 3: (4x4) -> (8x8)
        main.add_module(
            "block3",
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 4: (8x8) -> (8x8)
        main.add_module(
            "block4",
            nn.Sequential(
                nn.ConvTranspose2d(256, 256, 3, 1, 1, bias=False),
                nn.BatchNorm2d(256),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 5: (8x8) -> (16x16)
        main.add_module(
            "block5",
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 6: (16x16) -> (16x16)
        main.add_module(
            "block6",
            nn.Sequential(
                nn.ConvTranspose2d(128, 128, 3, 1, 1, bias=False),
                nn.BatchNorm2d(128),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 7: (16x16) -> (32x32)
        main.add_module(
            "block7",
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 8: (32x32) -> (32x32)
        main.add_module(
            "block8",
            nn.Sequential(
                nn.ConvTranspose2d(64, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64),
                get_nonlinearity(self.nonlinearity),
            ),
        )

        return main

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
        label_embedding = (
            self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        )  # (batch_size, num_classes, 1, 1)
        z = torch.cat(
            [noise, label_embedding], dim=1
        )  # (batch_size, latent_dim + num_classes, 1, 1)
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(
        self,
        num_classes=10,
        img_channels=1,
        num_layers=4,
        nonlinearity="LeakyReLU",
    ):
        """
        The Discriminator class for DCGAN.

        Parameters
        ----------
        num_classes : int
            The number of classes in the dataset, used for label supervision.
        img_channels : int
            The number of channels in the input images.
        num_layers : int
            The number of layers in the Discriminator, not including the output layer.
        nonlinearity : str
            The nonlinearity to use in the Discriminator.
        """
        super().__init__()
        self.input_dim = img_channels + num_classes
        self.nonlinearity = nonlinearity

        # The embedding layer for the class labels
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # Get the main body of the Discriminator
        if num_layers == 4:
            main = self._get_4_layer()
        elif num_layers == 8:
            main = self._get_8_layer()

        # Output layer: produces probability of input image being real
        main.append(
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )
        main.append(nn.Sigmoid())
        self.main = nn.Sequential(*main)

    def _get_4_layer(self) -> nn.Sequential:
        main = nn.Sequential()

        # Layer 1: (64x64) -> (32x32)
        main.add_module(
            "block1",
            nn.Sequential(
                nn.Conv2d(self.input_dim, 64, 4, 2, 1, bias=False),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 2: (32x32) -> (16x16)
        main.add_module(
            "block2",
            nn.Sequential(
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 3: (16x16) -> (8x8)
        main.add_module(
            "block3",
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 4: (8x8) -> (4x4)
        main.add_module(
            "block4",
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                get_nonlinearity(self.nonlinearity),
            ),
        )

        return main

    def _get_8_layer(self) -> nn.Sequential:
        main = nn.Sequential()

        # Layer 1: (64x64) -> (32x32)
        main.add_module(
            "block1",
            nn.Sequential(
                nn.Conv2d(self.input_dim, 64, 4, 2, 1, bias=False),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 2: (32x32) -> (32x32)
        main.add_module(
            "block2",
            nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 3: (32x32) -> (16x16)
        main.add_module(
            "block3",
            nn.Sequential(
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 4: (16x16) -> (16x16)
        main.add_module(
            "block4",
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                nn.BatchNorm2d(128),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 5: (16x16) -> (8x8)
        main.add_module(
            "block5",
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 6: (8x8) -> (8x8)
        main.add_module(
            "block6",
            nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1, bias=False),
                nn.BatchNorm2d(256),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 7: (8x8) -> (4x4)
        main.add_module(
            "block7",
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 8: (4x4) -> (4x4)
        main.add_module(
            "block8",
            nn.Sequential(
                nn.Conv2d(512, 512, 3, 1, 1, bias=False),
                nn.BatchNorm2d(512),
                get_nonlinearity(self.nonlinearity),
            ),
        )

        return main

    def forward(self, img, labels) -> torch.Tensor:
        """
        Forward pass of the Discriminator to classify real/fake images.

        Parameters
        ----------
        img : torch.Tensor (batch_size, img_channels, 64, 64)
            The input images to be classified.
        labels : torch.Tensor (batch_size)
            The class labels for the images.

        Returns
        -------
        torch.Tensor (batch_size, 1, 1, 1)
            The probability of the input images being real.
        """
        # Concatenate image and class label embedding
        label_embedding = (
            self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        )  # (batch_size, num_classes, 1, 1)
        label_embedding = label_embedding.expand(
            -1, -1, img.size(2), img.size(3)
        )  # (batch_size, num_classes, 64, 64)
        img = torch.cat(
            [img, label_embedding], dim=1
        )  # (batch_size, img_channels + num_classes, 64, 64)
        return self.main(img)


class Critic(nn.Module):
    def __init__(
        self,
        num_classes=10,
        img_channels=1,
        num_layers=4,
        nonlinearity="LeakyReLU",
        norm_layer="none",
    ):
        """
        The Critic class for WGAN and WGAN-GP.

        Parameters
        ----------
        num_classes : int
            The number of classes in the dataset, used for label supervision.
        img_channels : int
            The number of channels in the input images.
        num_layers : int
            The number of layers in the Critic, not including the output layer.
        nonlinearity : str
            The nonlinearity to use in the Critic.
        norm_layer : str
            The normalization layer to use in the Critic.
            "bn" for BatchNorm, "ln" for LayerNorm, and "none" for no normalization.
            For WGAN, use "none". For WGAN-GP, use "ln".
        """
        super().__init__()
        self.input_dim = img_channels + num_classes
        self.nonlinearity = nonlinearity
        self.norm_layer = norm_layer

        # The embedding layer for the class labels
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # Get the main body of the Discriminator
        if num_layers == 4:
            main = self._get_4_layer()
        elif num_layers == 8:
            main = self._get_8_layer()

        # Output layer: produces probability of input image being real
        main.append(
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )
        main.append(nn.Sigmoid())
        self.main = nn.Sequential(*main)

    def _get_4_layer(self) -> nn.Sequential:
        main = nn.Sequential()

        # Layer 1: (64x64) -> (32x32)
        main.add_module(
            "block1",
            nn.Sequential(
                nn.Conv2d(self.input_dim, 64, 4, 2, 1, bias=False),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 2: (32x32) -> (16x16)
        main.add_module(
            "block2",
            nn.Sequential(
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                get_norm_layer(self.norm_layer, 128, (16, 16)),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 3: (16x16) -> (8x8)
        main.add_module(
            "block3",
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                get_norm_layer(self.norm_layer, 256, (8, 8)),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 4: (8x8) -> (4x4)
        main.add_module(
            "block4",
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                get_norm_layer(self.norm_layer, 512, (4, 4)),
                get_nonlinearity(self.nonlinearity),
            ),
        )

        return main

    def _get_8_layer(self) -> nn.Sequential:
        main = nn.Sequential()

        # Layer 1: (64x64) -> (32x32)
        main.add_module(
            "block1",
            nn.Sequential(
                nn.Conv2d(self.input_dim, 64, 4, 2, 1, bias=False),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 2: (32x32) -> (32x32)
        main.add_module(
            "block2",
            nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                get_norm_layer(self.norm_layer, 64, (32, 32)),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 3: (32x32) -> (16x16)
        main.add_module(
            "block3",
            nn.Sequential(
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                get_norm_layer(self.norm_layer, 128, (16, 16)),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 4: (16x16) -> (16x16)
        main.add_module(
            "block4",
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                get_norm_layer(self.norm_layer, 128, (16, 16)),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 5: (16x16) -> (8x8)
        main.add_module(
            "block5",
            nn.Sequential(
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                get_norm_layer(self.norm_layer, 256, (8, 8)),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 6: (8x8) -> (8x8)
        main.add_module(
            "block6",
            nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1, bias=False),
                get_norm_layer(self.norm_layer, 256, (8, 8)),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 7: (8x8) -> (4x4)
        main.add_module(
            "block7",
            nn.Sequential(
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                get_norm_layer(self.norm_layer, 512, (4, 4)),
                get_nonlinearity(self.nonlinearity),
            ),
        )
        # Layer 8: (4x4) -> (4x4)
        main.add_module(
            "block8",
            nn.Sequential(
                nn.Conv2d(512, 512, 3, 1, 1, bias=False),
                get_norm_layer(self.norm_layer, 512, (4, 4)),
                get_nonlinearity(self.nonlinearity),
            ),
        )

        return main

    def forward(self, img, labels) -> torch.Tensor:
        """
        Forward pass of the Discriminator to classify real/fake images.

        Parameters
        ----------
        img : torch.Tensor (batch_size, img_channels, 64, 64)
            The input images to be classified.
        labels : torch.Tensor (batch_size)
            The class labels for the images.

        Returns
        -------
        torch.Tensor (batch_size, 1, 1, 1)
            The probability of the input images being real.
        """
        # Concatenate image and class label embedding
        label_embedding = (
            self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        )  # (batch_size, num_classes, 1, 1)
        label_embedding = label_embedding.expand(
            -1, -1, img.size(2), img.size(3)
        )  # (batch_size, num_classes, 64, 64)
        img = torch.cat(
            [img, label_embedding], dim=1
        )  # (batch_size, img_channels + num_classes, 64, 64)
        return self.main(img)
