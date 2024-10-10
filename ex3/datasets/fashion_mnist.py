from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class FashionMNISTDataModule:
    """
    Fashion MNIST DataModule for handling the data loaders.
    Follows the PyTorch Lightning DataModule structure.
    """

    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # Normalize images between -1 and 1
            ]
        )

    def setup(self):
        # Load the training set
        self.train_dataset = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=self.transform
        )
        # Load the test set
        self.test_dataset = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )
