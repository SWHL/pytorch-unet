# !/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2

import helper
import pytorch_unet
import simulation

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


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
    num_class = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trans = transforms.Compose([
        transforms.ToTensor(),
    ])


    model = pytorch_unet.ResNetUNet(num_class).to(device)
    model.eval()   # Set model to evaluate mode
    model.load_state_dict(torch.load('exp/latest.pth'))

    test_dataset = SimDataset(3, transform=trans)
    test_loader = DataLoader(test_dataset, batch_size=1,
                                shuffle=False, num_workers=0)

    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    pred = model(inputs)

    pred = pred.data.cpu().numpy()
    print(pred.shape)

    # Change channel-order and make 3 channels for matplot
    input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

    # Map each channel (i.e. class) to each color
    target_masks_rgb = [helper.masks_to_colorimg(
        x) for x in labels.cpu().numpy()]
    pred_rgb = [helper.masks_to_colorimg(x) for x in pred]

    cv2.imwrite('tmp/input.jpg', input_images_rgb[0])
    cv2.imwrite('tmp/target.jpg', target_masks_rgb[0])
    cv2.imwrite('tmp/pred.jpg', pred_rgb[0])
