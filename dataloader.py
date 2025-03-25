import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloader(batch_size, train=True):
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

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True
    )

    return dataloader
