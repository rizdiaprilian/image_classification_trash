import os
import pytest
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from src.training_validation_step import model_training, model_validation
from src.models import modelling

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
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fixture to create a mock model
@pytest.fixture
def model(device):
    model = SimpleModel().to(device)
    return model

# Fixture to create a mock dataset and dataloader
@pytest.fixture
def train_dataloader():
    # Create a random dataset
    x_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    dataset = TensorDataset(x_train, y_train)
    return DataLoader(dataset, batch_size=10, shuffle=True)

@pytest.fixture
def val_dataloader():
    # Create a random dataset
    x_val = torch.randn(50, 10)
    y_val = torch.randint(0, 2, (50,))
    dataset = TensorDataset(x_val, y_val)
    return DataLoader(dataset, batch_size=10)

# Fixture to create a loss function
@pytest.fixture
def loss_fn():
    return torch.nn.CrossEntropyLoss()

# Fixture to create an optimizer
@pytest.fixture
def optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=0.01)

# Test the training function
def test_model_training(model, train_dataloader, loss_fn, optimizer, device):
    loss_hist_train, accuracy_hist_train = model_training(
        model, train_dataloader, loss_fn, optimizer, device)
    
    assert isinstance(loss_hist_train, float or int), "Training loss should be a float"
    assert isinstance(accuracy_hist_train, float or int), "Training accuracy should be a float"

# Test the validation function
def test_model_validation(model, val_dataloader, loss_fn, device):
    loss_hist_valid, accuracy_hist_valid = model_validation(
        model, val_dataloader, loss_fn, device)
    
    assert isinstance(loss_hist_valid, float or int), "Validation loss should be a float"
    assert isinstance(accuracy_hist_valid, float or int), "Validation accuracy should be a float"

if __name__ == "__main__":
    pytest.main()