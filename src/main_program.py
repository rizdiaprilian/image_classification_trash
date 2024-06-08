import os

import torch
from dotenv import load_dotenv
from torch import nn, optim

from dataset import (get_test_dataloader, get_train_dataloader,
                     get_val_dataloader, load_test_dataset, load_train_dataset,
                     load_val_dataset, transform_test_dataset,
                     transform_train_dataset, transform_val_dataset)
from evaluation_ import accuracy_fn, test_model
from model_training import train
from models import modelling
from save_model import model_checkpoint

# Load environment variables from .env file
load_dotenv()

# Parameters
parent_dir = os.path.abspath(os.path.join(os.getcwd()))
num_epochs = int(os.getenv("NUM_EPOCHS"))
batch_size = int(os.getenv("BATCH_SIZE"))
learning_rate = float(os.getenv("LEARNING_RATE"))
target_dir = os.path.join(parent_dir, "models", "ResNet18")

## Verify that the PyTorch can gain an access to CUDA device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def main():
    train_dataset = load_train_dataset()
    val_dataset = load_val_dataset()
    test_dataset = load_test_dataset()

    train_dataset_transformed = transform_train_dataset(train_dataset)
    val_dataset_transformed = transform_val_dataset(val_dataset)
    test_dataset_transformed = transform_test_dataset(test_dataset)

    train_dataloader = get_train_dataloader(train_dataset_transformed, batch_size)
    val_dataloader = get_val_dataloader(val_dataset_transformed, batch_size)
    test_dataloader = get_test_dataloader(test_dataset_transformed, batch_size)

    pretrained_resnet18 = modelling(device)

    # Make the loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pretrained_resnet18.parameters(), lr=learning_rate)

    # Start training with help from engine.py
    train(
        pretrained_resnet18,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_func,
        num_epochs,
        device,
    )

    model_1_results = test_model(
        model=pretrained_resnet18,
        data_loader=test_dataloader,
        loss_fn=loss_func,
        accuracy_fn=accuracy_fn,
        device=device,
    )
    print(model_1_results)

    model_name = "model_resnet_checkpoint_1200.pth"
    model_checkpoint(pretrained_resnet18, target_dir, model_name)


if __name__ == "__main__":
    print(f"Using {device} device")
    main()
