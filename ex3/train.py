from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import DCGAN


# Load the FashionMNIST dataset
dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    ),
)

dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


dcgan_1 = DCGAN(
    latent_dim=100,
    num_classes=10,
    img_channels=1,
    learning_rate=0.0002,
    beta1=0.2,
    num_layers_G=4,
    num_layers_D=4,
    nonlinearity_G="ReLU",
    nonlinearity_D="LeakyReLU",
)
print(dcgan_1.netG)
print(dcgan_1.netD)
print("===============================================")


dcgan_1.train(dataloader, num_epochs=10, output_dir="./output/dcgan_1")
