# from apex.contrib.sparsity import ASP
# ASP.prune_trained_model(model, optimizer) #pruned a trained model

import numpy as np

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T

import time as t


def train_epoch(model, train_loader, optimizer, scheduler, device, verbose=0):

    train_loss = []
    train_acc = []

    for i, (data, target) in enumerate(train_loader):

        optimizer.zero_grad(set_to_none=False)

        data, target = data.to(device), target.to(device)

        output = model(data)

        loss = F.cross_entropy(output, target)
        train_loss.append(loss.item())

        acc = (output.max(1)[1].detach() == target).sum() / output.shape[0] * 100
        train_acc.append(acc.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose:
            print(f"train iteration: {i} | loss: {loss} | acc: {acc}")

    return np.mean(train_loss), np.mean(train_acc)


def test_epoch(model, test_loader, device, verbose=0):

    test_loss = []
    test_acc = []

    with torch.no_grad():

        for i, (data, target) in enumerate(test_loader):

            data, target = data.to(device), target.to(device)

            output = model(data)

            loss = F.cross_entropy(output, target)
            test_loss.append(loss.item())

            acc = (output.max(1)[1].detach() == target).sum() / output.shape[0] * 100
            test_acc.append(acc.item())

            if verbose:
                print(f"train iteration: {i} | loss: {loss} | acc: {acc}")

    return np.mean(test_loss), np.mean(test_acc)


def print_info(info: dict):

    statement = ""

    for key, val in info.items():

        statement += "{0}: {1:.4g} | ".format(key, val)

    print(statement)


def train_model(model):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs = 50
    batch_size = 256

    momentum = 0.9
    lr = 0.1
    weight_decay = 5e-4
    # seed?

    verbose = False

    model = model.to(device)

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
        train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        div_factor=1e5,
        epochs=n_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=1 / n_epochs,
        verbose=verbose,
    )

    time = t.time()

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(
            model.train(), train_loader, optimizer, scheduler, device
        )
        test_loss, test_acc = test_epoch(model.eval(), test_loader, device)

        info = {
            "epoch": epoch,
            "time": t.time() - time,
            "train loss": train_loss,
            "train acc": train_acc,
            "test loss": test_loss,
            "test acc": test_acc,
        }

        print_info(info)
        time = t.time()


if __name__ == "__main__":
    train_model()

