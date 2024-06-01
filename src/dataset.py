from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from pathlib import Path

import os

PARENT_DIR = os.path.abspath(os.path.join(os.getcwd()))


# Transformations for training set
train_transformer = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor()])

val_transformer = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor()])

test_transformer = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor()])


def load_train_dataset() -> Dataset:
    return ImageFolder(
                    os.path.join(PARENT_DIR, 'data', 'processed', 'trash_images_train_resized'), 
                    transform=transforms.ToTensor(),
                    target_transform=None)

def load_val_dataset() -> Dataset:
    return ImageFolder(
                    os.path.join(PARENT_DIR, 'data', 'processed', 'trash_images_val_resized'),
                    transform=transforms.ToTensor(),
                    target_transform=None)

def load_test_dataset() -> Dataset:
    return ImageFolder(
                    os.path.join(PARENT_DIR, 'data', 'processed', 'trash_images_test_resized'),
                    transform=transforms.ToTensor(),
                    target_transform=None)

def transform_train_dataset(train_Dataset: Dataset) -> Dataset:
    train_Dataset.transform = train_transformer
    return train_Dataset

def transform_val_dataset(val_Dataset: Dataset) -> Dataset:
    val_Dataset.transform = train_transformer
    return val_Dataset

def transform_test_dataset(test_Dataset: Dataset) -> Dataset:
    test_Dataset.transform = train_transformer
    return test_Dataset

def get_train_dataloader(train_Dataset: Dataset, batch_size: int) -> DataLoader:
# Turn datasets into iterables (batches)
    return DataLoader(train_Dataset, # dataset to turn into iterable
                        batch_size=batch_size, # how many samples per batch? 
                        shuffle=True # shuffle data every epoch?
                    )

def get_val_dataloader(val_Dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(val_Dataset,
                        batch_size=batch_size,
                        shuffle=False # don't necessarily have to shuffle the testing data
                    )

def get_test_dataloader(test_Dataset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(test_Dataset,
                        batch_size=batch_size,
                        shuffle=False # don't necessarily have to shuffle the testing data
                    )
