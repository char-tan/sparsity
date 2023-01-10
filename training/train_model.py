#from apex.contrib.sparsity import ASP     
#ASP.prune_trained_model(model, optimizer) #pruned a trained model

import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T


class PreLoadCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = torch.tensor(self.data, device=device)
        self.targets = torch.tensor(self.targets, device=device)

        transforms = [T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]

        if kwargs['train']:
            transforms += [T.RandomHorizontalFlip()]

        self.transform = T.Compose(transforms)

    def __getitem__(self, index):

        index %= len(self.data)

        image = self.data[index].to(torch.float32).permute(2, 0, 1)

        target = self.targets[index]

        image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':

    device = 'cpu'
    n_epochs = 50
    batch_size = 128

    momentum=0.9
    lr=0.1
    weight_decay=5e-4
    # seed?

    model = torchvision.models.resnet18()

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

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=n_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=1/len(train_loader),
            )

    for epoch in range(n_epochs):

        for i, batch in enumerate(train_loader):

            data, target = batch

            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            print(f'iteration {i} | loss {loss}')

#torch.save(...) # saves the pruned checkpoint with sparsity masks 
