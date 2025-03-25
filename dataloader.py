import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
import os
from PIL import Image




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

def get_imagenet100_dataloader(batch_size, train=True, subset_size=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
    ])

    split = "train" if train else "validation"
    dataset = load_dataset("clane9/imagenet-100", split=split)

    def transform_fn(example):
        image = example["image"]
        if isinstance(image, list):
            image = image[0]  # In case a list slipped through
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)  # Convert numpy to PIL if needed
        return {
            "image": transform(image),
            "label": example["label"]
        }

    dataset.set_transform(transform_fn)

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

