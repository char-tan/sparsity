import numpy as np
import time as t

import torch
import torch.nn.functional as F

from training.utils import cifar10_dataloaders
from training.utils import epoch_summary


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


def train_phase(model, optimizer, train_loader, test_loader, config):

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
        max_lr=config.lr,
        div_factor=1e5,
        epochs=config.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=1 / config.num_epochs,
    )

    time = t.time()

    for epoch in range(config.num_epochs):
        train_loss, train_acc = train_epoch(
            model.train(), train_loader, optimizer, scheduler, config.device
        )
        test_loss, test_acc = test_epoch(model.eval(), test_loader, config.device)

        info = {
            "epoch": epoch,
            "time": t.time() - time,
            "train loss": train_loss,
            "train acc": train_acc,
            "test loss": test_loss,
            "test acc": test_acc,
        }

        epoch_summary(info)
        time = t.time()
