#from apex.contrib.sparsity import ASP     
#ASP.prune_trained_model(model, optimizer) #pruned a trained model

import numpy as np

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T

import time as t


class PreLoadCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = torch.tensor(self.data, device=device)
        self.targets = torch.tensor(self.targets, device=device)
        
        self.data = self.data.to(torch.float32)
        self.data = self.data.permute(0, 3, 1, 2)
        self.data = self.data / 255

        self.data = T.Normalize(
          (0.4914, 0.4822, 0.4465),
          (0.2023, 0.1994, 0.2010),
          )(self.data)

        if kwargs['train']:
            transforms = [T.RandomHorizontalFlip()]
        else:
            transforms = []

        self.transform = T.Compose(transforms)
       
    def __getitem__(self, index):

        image = self.data[index % len(self.data)]
        image = self.transform(image)
        target = self.targets[index % len(self.data)]

        return image, target

    def __len__(self):
        return self.data.shape[0]


def train_epoch(model, train_loader, optimizer, scheduler, verbose=0):

    train_loss = []
    train_acc = []

    for i, batch in enumerate(train_loader):
    
        optimizer.zero_grad(set_to_none=False)

        data, target = batch

        output = model(data)

        loss = F.cross_entropy(output, target)
        train_loss.append(loss.item())

        acc = (output.max(1)[1].detach() == target).sum() / output.shape[0] * 100
        train_acc.append(acc.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose:
            print(f'train iteration: {i} | loss: {loss} | acc: {acc}')

    return np.mean(train_loss), np.mean(train_acc)


def test_epoch(model, test_loader, verbose=0):

    test_loss = []
    test_acc = []

    with torch.no_grad():

        for i, batch in enumerate(test_loader):
        
            data, target = batch

            output = model(data)

            loss = F.cross_entropy(output, target)
            test_loss.append(loss.item())

            acc = (output.max(1)[1].detach() == target).sum() / output.shape[0] * 100
            test_acc.append(acc.item())

            if verbose:
                print(f'train iteration: {i} | loss: {loss} | acc: {acc}')

    return np.mean(test_loss), np.mean(test_acc)


def print_info(info: dict):

    statement = ''

    for key, val in info.items():

        statement += '{0}: {1:.4g} | '.format(key, val)

    print(statement)


def train_model():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_epochs = 50
    batch_size = 128

    momentum=0.9
    lr=0.1
    weight_decay=5e-4
    # seed?

    verbose = False

    model = torchvision.models.resnet18(num_classes=10).to(device)

    train_kwargs = {'root': 'data', 'train': True, 'download': True}
    test_kwargs = {'root': 'data', 'train': False, 'download': False}

    train_set = PreLoadCIFAR10(device, **train_kwargs)
    test_set = PreLoadCIFAR10(device, **test_kwargs)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size*4, shuffle=False)

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
            pct_start=1/n_epochs,
            verbose=verbose,
            )

    time = t.time()

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model.train(), train_loader, optimizer, scheduler)
        test_loss, test_acc = test_epoch(model.eval(), test_loader)

        info = {
          'epoch': epoch,
          'time': t.time() - time,
          'train loss': train_loss,
          'train acc': train_acc,
          'test loss': test_loss,
          'test acc': test_acc,
        }

        print_info(info)
        time = t.time()

if __name__ == '__main__':
    train_model()
