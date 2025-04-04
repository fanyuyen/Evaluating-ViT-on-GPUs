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
        num_workers=0,
        pin_memory=True
    )

    return dataloader

def get_imagenet100_dataloader(batch_size, train=True, subset_size=None):
    split = "train" if train else "validation"
    
    try:
        print(f"Loading ImageNet100 dataset - {split} split...")
        dataset = load_dataset("clane9/imagenet-100", split=split)
        print(f"Dataset loaded successfully. Size: {len(dataset)}")
        
        # Debugging info
        keys = list(dataset[0].keys())
        print(f"Example keys: {keys}")
        
        # Apply subset before transform
        if subset_size and subset_size < len(dataset):
            print(f"Using subset of {subset_size} examples")
            indices = list(range(subset_size))
            dataset = dataset.select(indices)
            print(f"Subset created. New size: {len(dataset)}")
        
        # Apply transform to each example individually
        transformed_dataset = []
        for i in range(len(dataset)):
            try:
                example = dataset[i]
                image = example["image"]
                
                # Convert to proper format
                if isinstance(image, list):
                    image = image[0]
                if isinstance(image, str):
                    image = Image.open(image).convert('RGB')
                elif not isinstance(image, Image.Image):
                    image = Image.fromarray(image).convert('RGB')
                
                # Resize and convert to tensor
                image = image.resize((224, 224), Image.BILINEAR)
                image = transforms.ToTensor()(image)
                
                transformed_dataset.append({
                    "image": image,
                    "label": example["label"]
                })
                
            except Exception as e:
                print(f"Error transforming example {i}: {str(e)}")
                if i > 0:
                    continue
                else:
                    raise
        
        print(f"Transformed {len(transformed_dataset)} examples successfully")
        
        # Create simple dataset wrapper
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        # Create dataloader
        dataset = SimpleDataset(transformed_dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=lambda batch: {
                'image': torch.stack([item['image'].repeat(3, 1, 1) if item['image'].size(0) == 1 else item['image'] for item in batch]),
                'label': torch.tensor([item['label'] for item in batch])
            }
        )
        
        return dataloader
        
    except Exception as e:
        print(f"Error in get_imagenet100_dataloader: {str(e)}")
        raise

def get_food101_dataloader(batch_size, train=True, subset_size=None):
    split = "train" if train else "validation"
    
    try:
        print(f"Loading Food101 dataset - {split} split...")
        dataset = load_dataset("food101", split=split)
        print(f"Dataset loaded successfully. Size: {len(dataset)}")
        
        # Debugging info
        keys = list(dataset[0].keys())
        print(f"Example keys: {keys}")
        
        # Apply subset before transform
        if subset_size and subset_size < len(dataset):
            print(f"Using subset of {subset_size} examples")
            indices = list(range(subset_size))
            dataset = dataset.select(indices)
            print(f"Subset created. New size: {len(dataset)}")
        
        # Apply transform to each example individually
        transformed_dataset = []
        for i in range(len(dataset)):
            try:
                example = dataset[i]
                image = example["image"]
                
                # Convert to proper format
                if isinstance(image, list):
                    image = image[0]
                if isinstance(image, str):
                    image = Image.open(image).convert('RGB')
                elif not isinstance(image, Image.Image):
                    image = Image.fromarray(image).convert('RGB')
                
                # Resize and convert to tensor
                image = image.resize((224, 224), Image.BILINEAR)
                image = transforms.ToTensor()(image)
                
                transformed_dataset.append({
                    "image": image,
                    "label": example["label"]
                })
                
            except Exception as e:
                print(f"Error transforming example {i}: {str(e)}")
                if i > 0:
                    continue
                else:
                    raise
        
        print(f"Transformed {len(transformed_dataset)} examples successfully")
        
        # Create simple dataset wrapper
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        # Create dataloader
        dataset = SimpleDataset(transformed_dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=lambda batch: {
                'image': torch.stack([item['image'] for item in batch]),
                'label': torch.tensor([item['label'] for item in batch])
            }
        )
        
        return dataloader
        
    except Exception as e:
        print(f"Error in get_food101_dataloader: {str(e)}")
        raise

def get_brain_tumor_dataloader(batch_size, train=True, subset_size=None):
    """
    Create dataloader for Brain Tumor MRI dataset
    Dataset structure:
    - Training/
      - glioma/
      - meningioma/
      - notumor/
      - pituitary/
    - Testing/
      - glioma/
      - meningioma/
      - notumor/
      - pituitary/
    """
    # Basic transforms for ViT
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Use the correct dataset path
    dataset_dir = './data/brain-tumor-mri-dataset'
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found at {dataset_dir}")
    
    # Use Training or Testing directory based on train parameter
    data_dir = os.path.join(dataset_dir, 'Training' if train else 'Testing')
    
    # Create dataset
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )
    
    # Print dataset information
    print(f"Loading {'Training' if train else 'Testing'} set:")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Class names: {dataset.classes}")
    print(f"Total images: {len(dataset)}")
    
    # Print class distribution
    class_counts = {dataset.classes[i]: 0 for i in range(len(dataset.classes))}
    for _, label in dataset.samples:
        class_counts[dataset.classes[label]] += 1
    print("Class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images")
    
    # Use a subset of data if specified
    if subset_size:
        dataset = Subset(dataset, list(range(subset_size)))
        print(f"Using subset of {subset_size} examples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader

