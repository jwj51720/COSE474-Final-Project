import torch
import torchvision
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from .utils import img_transform


def train_dataloader(config):
    transform = img_transform()

    cifar10_dataset = torchvision.datasets.CIFAR10(
        root=f"{config['save_path']}/data",
        train=True,
        download=True,
        transform=transform,
    )

    num_splits = 5  # 20%
    skf = StratifiedKFold(
        n_splits=num_splits, shuffle=True, random_state=config["seed"]
    )

    fold_loaders = []
    for fold, (train_indices, valid_indices) in enumerate(
        skf.split(cifar10_dataset.data, cifar10_dataset.targets), 1
    ):
        train_dataset = Subset(cifar10_dataset, train_indices)
        valid_dataset = Subset(cifar10_dataset, valid_indices)

        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
        )
        validloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
        )

        fold_loaders.append((trainloader, validloader))
        print(
            f"Fold {fold}: Train set size - {len(train_dataset)}, Valid set size - {len(valid_dataset)}"
        )
    return fold_loaders


def test_dataloader(config):
    transform = img_transform()

    test_dataset = torchvision.datasets.CIFAR10(
        root=f"{config['save_path']}/data",
        train=False,
        download=True,
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["test_batch_size"], shuffle=False, num_workers=2
    )
    print(f"Test set size - {len(test_dataset)}")
    return test_loader
