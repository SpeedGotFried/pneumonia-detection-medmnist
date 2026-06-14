import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO, PneumoniaMNIST

def get_dataset_info():
    """
    Returns information about the PneumoniaMNIST dataset from the MedMNIST metadata.
    """
    return INFO['pneumoniamnist']

def get_transforms(image_size=128, augment=False):
    """
    Creates transform pipelines for the training and validation/testing datasets.
    We convert the 1-channel grayscale images to 3 channels using Grayscale(3)
    so they are fully compatible with pretrained models from torchvision (ResNet, Inception, MobileNet).
    """
    base_transforms = [
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),  # Convert 1 channel to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet stats
    ]
    
    if augment:
        # Data augmentation to prevent overfitting on PneumoniaMNIST
        train_transforms = [
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        return transforms.Compose(train_transforms)
    
    return transforms.Compose(base_transforms)

def get_data_loaders(batch_size=64, image_size=128, augment_train=True):
    """
    Downloads (if necessary) and loads PneumoniaMNIST.
    Returns: train_loader, val_loader, test_loader
    """
    train_transform = get_transforms(image_size, augment=augment_train)
    eval_transform = get_transforms(image_size, augment=False)
    
    train_dataset = PneumoniaMNIST(split='train', transform=train_transform, download=True)
    val_dataset = PneumoniaMNIST(split='val', transform=eval_transform, download=True)
    test_dataset = PneumoniaMNIST(split='test', transform=eval_transform, download=True)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def print_dataset_statistics():
    """
    Analyzes and prints class distributions and sizes of the PneumoniaMNIST dataset.
    Useful for 'Problem Interpretation' requirements.
    """
    # Load raw datasets (no transforms) to inspect labels
    train_dataset = PneumoniaMNIST(split='train', download=True)
    val_dataset = PneumoniaMNIST(split='val', download=True)
    test_dataset = PneumoniaMNIST(split='test', download=True)
    
    info = get_dataset_info()
    label_dict = info['label']  # {'0': 'normal', '1': 'pneumonia'}
    
    print("="*50)
    print("PneumoniaMNIST Dataset Statistics")
    print("="*50)
    print(f"Description: {info['description']}")
    print(f"Task: {info['task']}")
    print(f"Classes: {label_dict}")
    print("-"*50)
    
    for name, ds in [('Train', train_dataset), ('Validation', val_dataset), ('Test', test_dataset)]:
        labels = ds.labels.flatten()
        total = len(labels)
        normal_count = sum(labels == 0)
        pneumonia_count = sum(labels == 1)
        
        normal_pct = (normal_count / total) * 100
        pneumonia_pct = (pneumonia_count / total) * 100
        
        print(f"{name} Split:")
        print(f"  Total samples: {total}")
        print(f"  Normal (0):     {normal_count:4d} ({normal_pct:.2f}%)")
        print(f"  Pneumonia (1):  {pneumonia_count:4d} ({pneumonia_pct:.2f}%)")
        print("-"*50)
    print("="*50)

if __name__ == "__main__":
    print_dataset_statistics()
