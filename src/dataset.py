import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class StripedDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None, indices=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        # List all files and sort them to ensure consistency
        self.all_images = sorted(os.listdir(image_dir))
        self.all_labels = sorted(os.listdir(label_dir))

        # If indices are provided, filter images and labels accordingly
        if indices is not None:
            self.images = [self.all_images[i] for i in indices if i < len(self.all_images)]
            self.labels = [self.all_labels[i] for i in indices if i < len(self.all_labels)]
        else:
            self.images = self.all_images
            self.labels = self.all_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label_name = self.labels[idx]  # Assuming label and image names are aligned

        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)

        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("L") if os.path.exists(label_path) else Image.fromarray(np.zeros((img.height, img.width), dtype=np.uint8))

        if self.transform:
            img = self.transform(img)
            label = self.transform(label)

        return img, label





