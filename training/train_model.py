#from apex.contrib.sparsity import ASP     
#ASP.prune_trained_model(model, optimizer) #pruned a trained model

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

def main():

    device = 'cuda'
    n_epochs = 50
    batch_size = 128

    momentum=0.9
    lr=0.1
    weight_decay=5e-4
    # seed?

    model = torchvision.models.resnet18().to(device)

    train_kwargs = {'root': 'data', 'train': True, 'download': True}
    test_kwargs = {'root': 'data', 'train': False, 'download': False}

    train_set = PreLoadCIFAR10(device, **train_kwargs)
    test_set = PreLoadCIFAR10(device, **test_kwargs)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size*4, shuffle=False)

    print(len(train_loader))

    optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=lr,
            momentum=momentum, 
            weight_decay=weight_decay,
            )

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            div_factor=1e5,
            epochs=n_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=1/n_epochs,
            verbose=True,
            )

    time = t.time()

    for epoch in range(n_epochs):

        for i, batch in enumerate(train_loader):

            data, target = batch

            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            print(f'iteration {i} | loss {loss}')

        print(t.time() - time)

        time = t.time()

#torch.save(...) # saves the pruned checkpoint with sparsity masks 
