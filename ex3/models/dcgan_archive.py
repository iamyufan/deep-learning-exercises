import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os


# Generator
class Generator(nn.Module):
    def __init__(
        self, latent_dim=100, base_filter_count=32, num_classes=10, img_channels=1
    ):
        """
        The Generator class for DCGAN.

        Parameters
        ----------
        latent_dim : int
            The dimension of the latent random noise vector.
        base_filter_count : int
            The base number of filters for the convolutional layers
            i.e., the number of filters in the first layer, which is doubled in each subsequent layer.
        num_classes : int
            The number of classes in the dataset, used for label supervision.
        img_channels : int
            The number of channels in the input images.
        """
        super(Generator, self).__init__()

        # Embedding for the label supervision
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # Blocks of the transposed convolutional layers
        self.model = nn.Sequential(
            self.block(
                latent_dim + num_classes, base_filter_count * 4, 4, 1, 0
            ),  # Depth 1
            self.block(base_filter_count * 4, base_filter_count * 2),  # Depth 2
            self.block(base_filter_count * 2, base_filter_count),  # Depth 3
            nn.ConvTranspose2d(base_filter_count, img_channels, 4, 2, 1),  # Depth 4
            nn.Tanh(),  # Output layer for [-1, 1] images
        )

    @staticmethod
    def block(
        input_channels: int,
        output_channels: int,
        kernel_size=4,
        stride=2,
        padding=1,
        batch_norm=True,
    ):
        """
        A helper function to create a transposed convolutional block.
        """
        layers = [
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(output_channels))
        layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((noise, label_embedding), -1)
        gen_input = gen_input.view(gen_input.size(0), gen_input.size(1), 1, 1)
        return self.model(gen_input)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, base_filter_count=32, num_classes=10, img_channels=1):
        """
        The Discriminator class for DCGAN.

        Parameters
        ----------
        base_filter_count : int
            The base number of filters for the convolutional layers
            i.e., the number of filters in the first layer, which is doubled in each subsequent layer.
        num_classes : int
            The number of classes in the dataset, used for label supervision.
        img_channels : int
            The number of channels in the input images.
        """
        super(Discriminator, self).__init__()

        # Embedding for the label supervision
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            self.block(
                img_channels + num_classes, base_filter_count, batch_norm=False
            ),  # Depth 1
            self.block(base_filter_count, base_filter_count * 2),  # Depth 2
            self.block(base_filter_count * 2, base_filter_count * 4),  # Depth 3
            nn.Conv2d(base_filter_count * 4, 1, 4, 1, 0),  # Depth 4
            nn.Sigmoid(),  # For probability estimation
        )

    @staticmethod
    def block(
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        batch_norm=True,
    ):
        layers = [
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(output_channels))
        layers.append(nn.ReLU(True))
        return nn.Sequential(*layers)

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels)
        label_embedding = label_embedding.view(
            label_embedding.size(0), label_embedding.size(1), 1, 1
        )
        label_embedding = label_embedding.repeat(1, 1, img.size(2), img.size(3))
        d_input = torch.cat((img, label_embedding), 1)
        return self.model(d_input).view(-1, 1)


# DCGAN Training Loop
def train_dcgan(
    epochs,
    dataloader,
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    criterion,
    latent_dim,
    device,
):
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.size(0)

            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # Adversarial ground truths
            valid = torch.ones((batch_size, 1), requires_grad=False).to(device)
            fake = torch.zeros((batch_size, 1), requires_grad=False).to(device)

            # ---------------------
            #  Train Generator
            # ---------------------
            optimizer_g.zero_grad()

            # Sample noise and labels as generator input
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_labels = torch.randint(0, 10, (batch_size,)).to(device)

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            g_loss = criterion(discriminator(gen_imgs, gen_labels), valid)

            g_loss.backward()
            optimizer_g.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()

            # Loss for real images
            real_loss = criterion(discriminator(real_imgs, labels), valid)

            # Loss for fake images
            fake_loss = criterion(discriminator(gen_imgs.detach(), gen_labels), fake)

            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_d.step()

            # Print training progress
            if i % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] Batch {i}/{len(dataloader)} \
                      Loss D: {d_loss.item()}, loss G: {g_loss.item()}"
                )

        # Save generated samples every epoch
        save_image(gen_imgs.data[:25], f"images/{epoch}.png", nrow=5, normalize=True)


# Hyperparameters
latent_dim = 100
batch_size = 64
epochs = 50
lr = 0.0002
b1, b2 = 0.5, 0.999
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize data module
data_module = FashionMNISTDataModule(batch_size)
data_module.setup()

# Initialize generator and discriminator
generator = Generator(latent_dim=latent_dim).to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# Loss function
criterion = nn.BCELoss()

# Train the DCGAN
train_dcgan(
    epochs,
    data_module.train_dataloader(),
    generator,
    discriminator,
    optimizer_g,
    optimizer_d,
    criterion,
    latent_dim,
    device,
)
