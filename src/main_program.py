import os
import sys

import torch
from torch import nn
from torch import optim
from dotenv import load_dotenv

from dataset import load_train_dataset, load_val_dataset, load_test_dataset
from dataset import transform_train_dataset, transform_val_dataset, transform_test_dataset
from dataset import get_train_dataloader, get_val_dataloader, get_test_dataloader

from models import modelling
from model_training import train
from save_model import model_checkpoint

# Load environment variables from .env file
load_dotenv()

# Parameters
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# checkpoint_dir = os.getenv('CHECKPOINT_DIR')
num_epochs = int(os.getenv('NUM_EPOCHS'))
batch_size = int(os.getenv('BATCH_SIZE'))
learning_rate = float(os.getenv('LEARNING_RATE'))
target_dir = os.path.join(parent_dir, "models" , "ResNet18")

## Verify that the PyTorch can gain an access to CUDA device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def main():
    train_dataset = load_train_dataset()
    val_dataset = load_val_dataset()
    test_dataset = load_test_dataset()

    # # What's the shape of the image?
    # image, label = train_dataset[0]
    # print(f'Image shape of the first image is {image.shape}')
    # print(f'Image format is {type(image)}')
    # print(f"class names {train_dataset.classes}")

    train_dataset_transformed = transform_train_dataset(train_dataset)
    val_dataset_transformed = transform_val_dataset(val_dataset)
    test_dataset_transformed = transform_test_dataset(test_dataset)

    # # What's the shape of the image?
    # image, label = train_dataset_transformed[0]
    # print(f'Image shape of the first image is {image.shape}')
    # print(f'Image format is {type(image)}')

    train_dataloader = get_train_dataloader(train_dataset_transformed, batch_size)
    val_dataloader = get_val_dataloader(val_dataset_transformed, batch_size)
    test_dataloader = get_test_dataloader(test_dataset_transformed, batch_size)

    # # Let's check out what we've created
    # print(f"Dataloaders: {train_dataloader, val_dataloader, test_dataloader}") 
    # print(f"Length of train dataloader: {len(train_dataloader)} batches of {batch_size}")
    # print(f"Length of validation dataloader: {len(val_dataloader)} batches of {batch_size}")
    # print(f"Length of test dataloader: {len(test_dataloader)} batches of {batch_size}")

    pretrained_resnet18 = modelling(device)

    # Make the loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pretrained_resnet18.parameters(), lr=learning_rate)

    # Print a summary using torchinfo (uncomment for actual output)
#     print(summary(model=pretrained_resnet18, 
#         input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
#         # col_names=["input_size"], # uncomment for smaller output
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"]
# ))
    # Start training with help from engine.py
    train(pretrained_resnet18, 
        train_dataloader, val_dataloader, 
        optimizer, loss_func,
        num_epochs, device)
    
    model_name = "model_resnet_checkpoint_1445.pth"
    model_checkpoint(pretrained_resnet18, target_dir, model_name)


if __name__ == '__main__':
    print(f"Using {device} device")
    main()