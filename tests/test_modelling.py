import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import modelling
from src.training_validation_step import model_training, model_validation


# Fixture to load the dataset
def test_modelling():
    model = modelling(device="cpu")
    assert isinstance(model, torch.nn.Module)


# Define a simple mock model for testing
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.layer(x)


# Fixture to set up the device
@pytest.fixture
def device():
    return torch.device("cpu")


# Fixture to create a mock model
@pytest.fixture
def model():
    # model = SimpleModel().to(device)
    model = modelling(device="cpu")
    return model


# Fixture to create a mock dataset and dataloader
@pytest.fixture
def train_dataloader():
    # Define the dimensions of the images
    batch_size = 10
    num_samples = 100
    num_channels = 3  # For RGB images
    height = 224  # Height of the image
    width = 224  # Width of the image

    # Create random image data
    x_train = torch.randn(num_samples, num_channels, height, width)
    y_train = torch.randint(0, 2, (num_samples,))  # Binary classification

    # Create a dataset and dataloader
    dataset = TensorDataset(x_train, y_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


@pytest.fixture
def val_dataloader():
    # Define the dimensions of the images
    batch_size = 10
    num_samples = 100
    num_channels = 3  # For RGB images
    height = 224  # Height of the image
    width = 224  # Width of the image

    # Create random image data
    x_train = torch.randn(num_samples, num_channels, height, width)
    y_train = torch.randint(0, 2, (num_samples,))  # Binary classification

    # Create a dataset and dataloader
    dataset = TensorDataset(x_train, y_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Fixture to create a loss function
@pytest.fixture
def loss_fn():
    return torch.nn.CrossEntropyLoss()


# Fixture to create an optimizer
@pytest.fixture
def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=1e-4)


# Test the training function
def test_model_training(model, train_dataloader, loss_fn, optimizer, device):
    loss_hist_train, accuracy_hist_train = model_training(
        model, train_dataloader, loss_fn, optimizer, device
    )

    # Convert accuracy_hist_train to float if it's a torch.Tensor
    if isinstance(accuracy_hist_train, torch.Tensor):
        accuracy_hist_train = accuracy_hist_train.item()

    assert isinstance(loss_hist_train, float), "Training loss should be a float"
    assert isinstance(accuracy_hist_train, float), "Training accuracy should be a float"


# Test the validation function
def test_model_validation(model, val_dataloader, loss_fn, device):
    loss_hist_valid, accuracy_hist_valid = model_validation(
        model, val_dataloader, loss_fn, device
    )

    # Convert accuracy_hist_train to float if it's a torch.Tensor
    if isinstance(accuracy_hist_valid, torch.Tensor):
        accuracy_hist_valid = accuracy_hist_valid.item()

    assert isinstance(loss_hist_valid, float), "Validation loss should be a float"
    assert isinstance(
        accuracy_hist_valid, float
    ), "Validation accuracy should be a float"


if __name__ == "__main__":
    pytest.main()
