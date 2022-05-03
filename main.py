# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import copy
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torchvision import transforms

import helper
import pytorch_unet
import simulation
from loss import dice_loss


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e-10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                torch.save(model.state_dict(), 'exp/best.pth')
            else:
                torch.save(model.state_dict(), 'exp/latest.pth')

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = simulation.generate_random_data(
            192, 192, count=count)
        self.transform = transform

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]

        if self.transform:
            image = self.transform(image)

        return [image, mask]

    def __len__(self):
        return len(self.input_images)


if __name__ == '__main__':
    # Generate some random images
    input_images, target_masks = simulation.generate_random_data(192, 192, count=3)

    # Change channel-order and make 3 channels for matplot
    input_images_rgb = [x.astype(np.uint8) for x in input_images]

    # Map each channel (i.e. class) to each color
    target_masks_rgb = [helper.masks_to_colorimg(x) for x in target_masks]

    # use same transform for train/val for this example
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = SimDataset(2000, transform=trans)
    val_set = SimDataset(200, transform=trans)

    image_datasets = {'train': train_set, 'val': val_set}

    batch_size = 48

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size,
                            shuffle=True, num_workers=4),

        'val': DataLoader(val_set, batch_size=batch_size,
                          shuffle=False, num_workers=0)
    }

    dataset_sizes = {
        x: len(image_datasets[x]) for x in image_datasets.keys()
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_class = 6

    model = pytorch_unet.ResNetUNet(num_class).to(device)
    summary(model, input_size=(3, 224, 224))

    optimizer_ft = optim.Adam(model.parameters(), lr=1e-3)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25,
                                           gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=40)

