from timeit import default_timer as timer

import torch

num_epochs = 10


def model_training(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    loss_hist_train = 0
    accuracy_hist_train = 0
    train_time_start_on_gpu = timer()
    model.train()
    for x_batch, y_batch in train_dataloader:
        # Send data to GPU
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # 1. Forward pass
        pred = model(x_batch)

        # 2. Calculate loss
        loss = loss_fn(pred, y_batch)

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # 3. Optimizer zero grad
        optimizer.zero_grad()
        loss_hist_train += loss.item() * y_batch.size(0)
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        accuracy_hist_train += is_correct.sum().cpu()

    loss_hist_train /= len(train_dataloader.dataset)
    accuracy_hist_train /= len(train_dataloader.dataset)

    print(type(accuracy_hist_train))
    print(f"Accuracy: {accuracy_hist_train:.4f}")

    train_time_end_on_gpu = timer()
    periods = train_time_end_on_gpu - train_time_start_on_gpu
    print(f"Train time on GPU: {periods:.4f} seconds")

    return loss_hist_train, accuracy_hist_train


def model_validation(
    model: torch.nn.Module,
    val_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
):

    loss_hist_valid = 0
    accuracy_hist_valid = 0

    # Set the model in eval mode
    model.eval()

    with torch.no_grad():
        for x_batch, y_batch in val_dataloader:
            # Send data to GPU
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)

            loss = loss_fn(pred, y_batch)

            # Accumulate the loss and accuracy values per batch
            loss_hist_valid += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_valid += is_correct.sum().cpu()

    loss_hist_valid /= len(val_dataloader.dataset)
    accuracy_hist_valid /= len(val_dataloader.dataset)

    print(type(accuracy_hist_valid))
    print(f"Validation Accuracy: {accuracy_hist_valid:.4f}")

    return loss_hist_valid, accuracy_hist_valid
