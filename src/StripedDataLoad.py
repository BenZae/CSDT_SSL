import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np

class StripedDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None, label_rate=1.0):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir), key=lambda x: int(x.split('.')[0]))  # 按文件名数字排序

        num_labeled = int(len(self.images) * label_rate)
        self.has_label = {img: True for img in self.images[:num_labeled]}
        self.has_label.update({img: False for img in self.images[num_labeled:]})

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        has_label = self.has_label[img_name]
        if has_label:
            mask_path = os.path.join(self.label_dir, img_name)
            mask = Image.open(mask_path).convert("L") if os.path.exists(mask_path) else Image.fromarray(
                np.zeros((img.height, img.width), dtype=np.uint8))
        else:
            mask = Image.fromarray(np.zeros((img.height, img.width), dtype=np.uint8))

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask, has_label, img_name