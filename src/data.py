import torch
import torchvision
import torchvision.transforms as transforms
import warnings


def get_synthetic_sample(batch_size=1, S=32, D=64, scale=1.0, seed=None):
    """Generates synthetic random normal data."""
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(batch_size, S, D) * scale


def load_cifar10_sample(batch_size=16, data_dir="./data"):
    """Load CIFAR-10 images and preprocess them into (S, D) sequences.
    Downloads the dataset to data_dir if it doesn't exist.
    """
    warnings.filterwarnings(
        "ignore", message=".*align should be passed as Python or NumPy boolean.*"
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    images, _ = next(iter(loader))
    B, C, H, W = images.shape
    S = H
    D = W * C
    sequences = images.permute(0, 2, 3, 1).reshape(B, S, D)
    return sequences
