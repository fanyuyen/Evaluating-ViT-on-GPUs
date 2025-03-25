import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_cifar10_dataloader(batch_size, train=True, subset_size=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )

    # Use a subset of data for debugging
    if subset_size:
        dataset = Subset(dataset, list(range(subset_size)))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True
    )

    return dataloader
