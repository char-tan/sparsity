from dataclasses import dataclass

import torch

import torchvision
import torchvision.transforms as T


def mask_checkpoint(checkpoint, masked_model):

    for key, value in masked_model.state_dict().items():
        if ".__weight_mma_mask" in key:
            layer_name = key.replace(".__weight_mma_mask", ".weight")
            checkpoint[layer_name] *= value

    return checkpoint


# little bit hacky
def find_downsample_layers(model):

    downsample_layers = []

    for name, _ in model.named_parameters():

        if "downsample" in name and "weight" in name:

            downsample_layers.append(name[:-7])

    return downsample_layers


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs: int = 50
    batch_size: int = 256
    momentum: float = 0.9
    lr: float = 0.1
    weight_decay: float = 5e-4
    seed: int = 42


def cifar10_dataloaders(config):

    train_transforms = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_transforms = T.Compose(
        [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    train_kwargs = {
        "root": "data",
        "train": True,
        "download": True,
        "transform": train_transforms,
    }
    test_kwargs = {
        "root": "data",
        "train": False,
        "download": False,
        "transform": test_transforms,
    }

    train_set = torchvision.datasets.CIFAR10(**train_kwargs)
    test_set = torchvision.datasets.CIFAR10(**test_kwargs)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size * 4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


def resnet18_small_input():

    # init model and change first layers for smaller input images
    model = torchvision.models.resnet18(num_classes=10, zero_init_residual=True)
    model.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
    )
    model.maxpool = torch.nn.Identity()

    return model


def epoch_summary(info: dict):

    summary = ""

    for key, val in info.items():

        summary += "{0}: {1:.4g} | ".format(key, val)

    print(summary)
