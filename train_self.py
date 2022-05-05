# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import cv2
import time
import copy
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import pytorch_unet
from loss import dice_loss

import os
gpu_id = ['0', '1']
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_id)


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
    time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S',
                               time.localtime(time.time()))

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

            save_dir = Path(f'exp/{time_stamp}')
            save_dir.mkdir(parents=True, exist_ok=True)

            # deep copy the model
            if phase == 'val':
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                torch.save(model.state_dict(), f'{save_dir}/best.pth')
            else:
                torch.save(model.state_dict(), f'{save_dir}/latest.pth')

        time_elapsed = time.time() - since
        print('cost: {:.0f} m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


class SubtitleDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_list = list((Path(data_dir) / 'images').iterdir())
        self.gt_dir = Path(data_dir) / 'masks'

        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = cv2.imread(str(img_path))
        image = self.scale_resize(image).astype(np.float32)
        image /= 255.0

        mask_path = self.gt_dir / Path(img_path).name
        mask = cv2.imread(str(mask_path))
        mask = self.scale_resize(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask[np.newaxis, ...].astype(np.float32)
        mask /= 255

        if self.transform:
            image = self.transform(image)

        return [image, mask]

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def scale_resize(img, resize_value=(480, 480)):
        '''
        @params:
        img: ndarray
        resize_value: (width, height)
        '''
        # padding
        ratio = resize_value[0] / resize_value[1]  # w / h
        h, w = img.shape[:2]
        if w / h < ratio:
            # 补宽
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = cv2.copyMakeBorder(img, 0, 0, w_padding, w_padding,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            # 补高  (left, upper, right, lower)
            t = int(w / ratio)
            h_padding = (t - h) // 2
            color = tuple([int(i) for i in img[0, 0, :]])
            img = cv2.copyMakeBorder(img, h_padding, h_padding, 0, 0,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img = cv2.resize(img, resize_value,
                            interpolation=cv2.INTER_LANCZOS4)
        return img


if __name__ == '__main__':

    # use same transform for train/val for this example
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = SubtitleDataset(data_dir='datasets/gen_datasets/train',
                                transform=trans)
    val_set = SubtitleDataset(data_dir='datasets/gen_datasets/val',
                              transform=trans)

    image_datasets = {'train': train_set, 'val': val_set}

    batch_size = 16

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

    num_class = 1

    model = pytorch_unet.ResNetUNet(num_class)
    model = torch.nn.DataParallel(model, device_ids=list(range(len(gpu_id))))
    model.to(device)

    optimizer_ft = optim.Adam(model.parameters(), lr=1e-3)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25,
                                           gamma=0.1)

    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=40)

