import pytest
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.dataset import (get_train_dataloader, load_test_dataset,
                         load_train_dataset, load_val_dataset,
                         transform_train_dataset)


# Fixture to load the dataset
@pytest.fixture
def train_dataset():
    return load_train_dataset()


@pytest.fixture
def val_dataset():
    return load_val_dataset()


@pytest.fixture
def test_dataset():
    return load_test_dataset()


def test_load_train_dataset(train_dataset):
    assert isinstance(
        train_dataset, ImageFolder
    ), "Dataset should be an instance of ImageFolder"
    assert len(train_dataset) > 0, "Dataset should not be empty"


def test_transform_train_dataset(train_dataset):
    transformed_dataset = transform_train_dataset(train_dataset)
    assert transformed_dataset.transform is not None, "Transform should not be None"
    assert callable(transformed_dataset.transform), "Transform should be callable"


def test_get_train_dataloader(train_dataset):
    batch_size = 32
    dataloader = get_train_dataloader(train_dataset, batch_size)
    assert isinstance(
        dataloader, DataLoader
    ), "Dataloader should be an instance of DataLoader"
    for images, labels in dataloader:
        assert (
            images.shape[0] == batch_size
        ), "Batch size should match the expected value"
        break


def test_load_val_dataset(val_dataset):
    assert isinstance(
        val_dataset, ImageFolder
    ), "Dataset should be an instance of ImageFolder"
    assert len(val_dataset) > 0, "Dataset should not be empty"


def test_transform_val_dataset(val_dataset):
    transformed_dataset = transform_train_dataset(val_dataset)
    assert transformed_dataset.transform is not None, "Transform should not be None"
    assert callable(transformed_dataset.transform), "Transform should be callable"


def test_get_val_dataloader(val_dataset):
    batch_size = 32
    dataloader = get_train_dataloader(val_dataset, batch_size)
    assert isinstance(
        dataloader, DataLoader
    ), "Dataloader should be an instance of DataLoader"
    for images, labels in dataloader:
        assert (
            images.shape[0] == batch_size
        ), "Batch size should match the expected value"
        break


def test_load_test_dataset(test_dataset):
    assert isinstance(
        test_dataset, ImageFolder
    ), "Dataset should be an instance of ImageFolder"
    assert len(test_dataset) > 0, "Dataset should not be empty"


def test_transform_test_dataset(test_dataset):
    transformed_dataset = transform_train_dataset(test_dataset)
    assert transformed_dataset.transform is not None, "Transform should not be None"
    assert callable(transformed_dataset.transform), "Transform should be callable"


def test_get_test_dataloader(test_dataset):
    batch_size = 32
    dataloader = get_train_dataloader(test_dataset, batch_size)
    assert isinstance(
        dataloader, DataLoader
    ), "Dataloader should be an instance of DataLoader"
    for images, labels in dataloader:
        assert (
            images.shape[0] == batch_size
        ), "Batch size should match the expected value"
        break


if __name__ == "__main__":
    pytest.main()
