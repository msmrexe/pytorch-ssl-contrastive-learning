import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image, ImageOps
import random
import logging


class CIFAR10PretextDataset(Dataset):
    """
    A custom dataset class for applying self-supervised learning pretext tasks
    on the CIFAR-10 dataset.
    
    Applies random transformations and returns the image along with labels
    for the transformations applied (rotation, shear, color).
    """
    def __init__(self, root, train=True, transform=None):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform

        self.rotation_angles = [0, 90, 180, 270]
        self.shear_factors = [0.0, 0.2, 0.4]
        self.color_modes = ['original', 'grayscale', 'color_inverted']
        logging.info(f"CIFAR10PretextDataset created. Train={train}, {len(self.dataset)} samples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx] # Original label is ignored

        # 1. Choose random transformations and their labels
        rot_idx = random.randint(0, len(self.rotation_angles) - 1)
        shear_idx = random.randint(0, len(self.shear_factors) - 1)
        color_idx = random.randint(0, len(self.color_modes) - 1)

        rotation_angle = self.rotation_angles[rot_idx]
        shear_factor = self.shear_factors[shear_idx]
        color_mode = self.color_modes[color_idx]

        # 2. Apply transformations
        augmented_image = image.rotate(rotation_angle)

        if shear_factor > 0:
            augmented_image = augmented_image.transform(
                augmented_image.size, Image.AFFINE, (1, shear_factor, 0, 0, 1, 0)
            )

        if color_mode == 'grayscale':
            augmented_image = ImageOps.grayscale(augmented_image).convert('RGB')
        elif color_mode == 'color_inverted':
            augmented_image = ImageOps.invert(augmented_image)

        # 3. Apply standard transformations (ToTensor, Normalize)
        if self.transform:
            augmented_image = self.transform(augmented_image)

        labels = {
            'rotation': torch.tensor(rot_idx, dtype=torch.long),
            'shear': torch.tensor(shear_idx, dtype=torch.long),
            'color': torch.tensor(color_idx, dtype=torch.long)
        }
        return augmented_image, labels


class SimSiamTransform:
    """
    Applies strong augmentations twice to generate two views for SimSiam.
    This implementation is based on the augmentations used in the SimSiam paper.
    """
    def __init__(self, image_size=32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


def get_baseline_loaders(batch_size, subset_size=5000, data_root='./data'):
    """
    Returns train and test loaders for the baseline supervised task.
    The train loader uses a subset of the data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        train_dataset = torchvision.datasets.CIFAR10(download=True, root=data_root, train=True, transform=transform)
        
        if subset_size > len(train_dataset):
            logging.warning(f"Subset size {subset_size} is larger than dataset {len(train_dataset)}. Using full dataset.")
            subset_size = len(train_dataset)
        
        train_subset = Subset(train_dataset, list(range(subset_size)))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        logging.info(f"Baseline train loader created with {subset_size} samples.")

        test_dataset = torchvision.datasets.CIFAR10(download=True, root=data_root, train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        logging.info(f"Baseline test loader created with {len(test_dataset)} samples.")
        
        return train_loader, test_loader, len(train_subset), len(test_dataset)

    except Exception as e:
        logging.error(f"Failed to load baseline datasets: {e}")
        raise
      

def get_pretext_loaders(batch_size, data_root='./data'):
    """
    Returns train and test loaders for the self-supervised pretext task.
    Uses the custom CIFAR10PretextDataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    try:
        pretext_train_dataset = CIFAR10PretextDataset(root=data_root, train=True, transform=transform)
        pretext_train_loader = DataLoader(pretext_train_dataset, batch_size=batch_size, shuffle=True)

        pretext_test_dataset = CIFAR10PretextDataset(root=data_root, train=False, transform=transform)
        pretext_test_loader = DataLoader(pretext_test_dataset, batch_size=batch_size, shuffle=False)
        
        logging.info("Pretext task loaders created.")
        return pretext_train_loader, pretext_test_loader

    except Exception as e:
        logging.error(f"Failed to load pretext datasets: {e}")
        raise


def get_simsiam_loader(batch_size, data_root='./data'):
    """
    Returns a train loader for SimSiam pre-training.
    Uses the SimSiamTransform to get two augmented views.
    """
    try:
        simsiam_train_dataset = torchvision.datasets.CIFAR10(
            download=True, 
            root=data_root, 
            train=True, 
            transform=SimSiamTransform()
        )
        simsiam_loader = DataLoader(
            simsiam_train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True, 
            drop_last=True
        )
        logging.info(f"SimSiam pre-training loader created with {len(simsiam_train_dataset)} samples.")
        return simsiam_loader

    except Exception as e:
        logging.error(f"Failed to load SimSiam dataset: {e}")
        raise
