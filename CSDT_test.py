import torch
import torch.nn as nn
from src.dataset import StripedDataset
from nets.UNet_base import UNet_base
from nets.UCTransNet import UCTransNet
from nets.MRSA_Net import Unet_Attention
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import ml_collections
from src.utils import *

def test(model, device, test_loader, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    dice_scores = []
    iou_scores = []
    total_detection_successes = 0
    total_false_alarm_pixels = 0
    total_tested_pixels = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            for j, output in enumerate(outputs):
                filename = test_loader.dataset.images[i * test_loader.batch_size + j]
                save_path = os.path.join(save_dir, filename)
                save_image(output, save_path)
                output_bin = (output > 0.6).float()
                label_byte = labels[j].byte()
                dice_score = dice_coef(labels[j], output_bin)
                iou_score_value = iou_score(label_byte, output_bin.byte())

                dice_scores.append(dice_score.item())
                iou_scores.append(iou_score_value.item())
                # Calculate detection and false alarm
                detection, false_alarms, pixels = calculate_detection_and_false_alarm_rates(output_bin, labels[j])
                if detection:
                    total_detection_successes += 1
                total_false_alarm_pixels += false_alarms
                total_tested_pixels += pixels

    average_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0
    average_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
    detection_rate = total_detection_successes / len(test_loader.dataset)
    false_alarm_rate = (total_false_alarm_pixels / total_tested_pixels) * 1e4 if total_tested_pixels else 0

    print(f"Average Dice: {average_dice:.4f}")
    print(f"Average IoU: {average_iou:.4f}")
    print(f"Detection Rate: {detection_rate:.4f}")
    print(f"False Alarm Rate: {false_alarm_rate:.2f}")

    # Save only average scores and rates to a text file
    with open(os.path.join(save_dir, 'average_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Average Dice: {average_dice:.4f}\n")
        f.write(f"Average IoU: {average_iou:.4f}\n")
        f.write(f"Detection Rate: {detection_rate:.4f}\n")
        f.write(f"False Alarm Rate: {false_alarm_rate:.2f}\n")


def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config_vit = get_CTranS_config()
    # model = UCTransNet(config_vit, n_channels=3, n_classes=1).to(device)
    model = Unet_Attention( n_channels=3, n_classes=1).to(device)
    model_name = model.__class__.__name__
    model_path = f"checkpoint/{model_name}/Teacher/MRSA-Net.pth"
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_dataset = StripedDataset(image_dir="data/test/ALL/images", label_dir="data/test/ALL/masks", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    test_save_dir = "results/CSDT"
    os.makedirs(test_save_dir, exist_ok=True)
    test(model, device, test_loader, test_save_dir)

if __name__ == "__main__":
    main()
