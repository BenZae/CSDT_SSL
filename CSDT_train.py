import torch
import torch.nn as nn
from nets.UNet_base import UNet_base
from nets.UCTransNet import UCTransNet
from nets.MRSA_Net import MRSA_Net
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import random
from src.lossfunction import *
from torchvision.utils import save_image
import os
from torch.backends import cudnn
import time
import ml_collections
from src.utils import *
from src.StripedDataLoad import *
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def train(student_model, teacher_model, static_teacher_model, criterion, consistency_criterion, optimizer, train_loader,
          val_loader, device, alpha=0.99, num_epochs=400):
    student_model.train()
    teacher_model.train()
    best_val_loss = float('inf')

    model_name = student_model.__class__.__name__

    teacher_path = f"checkpoint/{model_name}/Teacher"
    student_path = f"checkpoint/{model_name}/Student"
    os.makedirs(teacher_path, exist_ok=True)
    os.makedirs(student_path, exist_ok=True)

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = 0.0
        consistency_loss_value = 0.0
        rampup_weight = sigmoid_rampup(epoch, num_epochs)

        static_teacher_output_dir = f"output/Binary/static_teacher/epoch_{epoch + 1}"
        teacher_output_dir = f"output/Binary/dynamic_teacher/epoch_{epoch + 1}"
        os.makedirs(static_teacher_output_dir, exist_ok=True)
        os.makedirs(teacher_output_dir, exist_ok=True)

        static_teacher_output_dir1 = f"output/Sigmoid/static_teacher/epoch_{epoch + 1}"
        teacher_output_dir1 = f"output/Sigmoid/dynamic_teacher/epoch_{epoch + 1}"
        os.makedirs(static_teacher_output_dir1, exist_ok=True)
        os.makedirs(teacher_output_dir1, exist_ok=True)

        components_count_path = os.path.join(teacher_output_dir, "components_count.txt")
        with open(components_count_path, "w") as file:
            file.write("Image Name,Connected Components Count\n")

        for i, (images, masks, has_labels, img_names) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            has_labels = has_labels.to(device)
            labeled_idx = has_labels.bool()
            unlabeled_idx = (~has_labels).bool()

            optimizer.zero_grad()
            student_outputs = student_model(images)

            # Generate outputs from both teacher models
            with torch.no_grad():
                static_pseudo_masks = static_teacher_model(images)
                dynamic_pseudo_masks = teacher_model(images)

                static_binary_masks = (static_pseudo_masks > 0.5).float()
                dynamic_binary_masks = (dynamic_pseudo_masks > 0.5).float()

                # Save outputs for unlabeled data only
                for j in range(images.size(0)):
                    if not has_labels[j]:
                        save_image(static_binary_masks[j], f"{static_teacher_output_dir}/{img_names[j]}")
                        save_image(dynamic_binary_masks[j], f"{teacher_output_dir}/{img_names[j]}")

                        save_image(static_pseudo_masks[j], f"{static_teacher_output_dir1}/{img_names[j]}")
                        save_image(dynamic_pseudo_masks[j], f"{teacher_output_dir1}/{img_names[j]}")

                        components_dynamic = count_connected_components(dynamic_binary_masks[j])
                        components_static = count_connected_components(static_binary_masks[j])
                        with open(components_count_path, "a") as file:
                            file.write(f"{img_names[j]}, {components_dynamic}, {components_static}\n")

            if abs(components_dynamic - components_static) > 10:
                masks[~has_labels] = static_binary_masks[~has_labels] if components_static < components_dynamic else \
                dynamic_binary_masks[~has_labels]
            else:
                static_linearity = calculate_linearity(static_binary_masks)
                dynamic_linearity = calculate_linearity(dynamic_binary_masks)
                masks[~has_labels] = static_binary_masks[~has_labels] if static_linearity > dynamic_linearity else \
                dynamic_binary_masks[~has_labels]

            # Compute loss for labeled data if present
            if labeled_idx.any():
                labeled_loss = criterion(student_outputs[labeled_idx], masks[labeled_idx])
            else:
                labeled_loss = 0
                # logging.warning(f"No labeled data in batch {i + 1}")

            # Compute loss for unlabeled data if present
            if unlabeled_idx.any():
                unlabeled_loss = criterion(student_outputs[unlabeled_idx], masks[unlabeled_idx])
            else:
                unlabeled_loss = 0
                # logging.warning(f"No unlabeled data in batch {i + 1}")

            segmentation_loss = labeled_loss + unlabeled_loss * 0.3

            loss = segmentation_loss
            train_loss += segmentation_loss.item()

            # If there is unlabeled data, calculate consistency loss
            if (~has_labels).any():
                consistency_loss = consistency_criterion(student_outputs, dynamic_pseudo_masks)
                weighted_consistency_loss = rampup_weight * consistency_loss
                consistency_loss_value += weighted_consistency_loss.item()
                loss += weighted_consistency_loss

            loss.backward()
            optimizer.step()

            update_teacher_model(student_model, teacher_model, alpha)

        train_loss /= len(train_loader)
        consistency_loss_value /= len(train_loader)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Consistency Loss: {consistency_loss_value:.4f}")
        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Consistency Loss: {consistency_loss_value:.4f}")

        val_loss = validate(teacher_model, criterion, val_loader, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}")
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if epoch > 49:
                teacher_filename = f"Unet_Attention_teacher_model_epoch_{epoch + 1}.pth"
                student_filename = f"Unet_Attention_student_model_epoch_{epoch + 1}.pth"
            else:
                teacher_filename = "Unet_Attention.pth"
                student_filename = "Unet_Attention.pth"

            torch.save(teacher_model.state_dict(), os.path.join(teacher_path, teacher_filename))
            print(f"Saved Best Model at Epoch [{epoch + 1}/{num_epochs}]")
            logging.info(f"Saved Best Model at Epoch [{epoch + 1}/{num_epochs}]")

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_time:.2f} seconds.")

def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64
    config.n_classes = 1
    return config

def main(label_rate=0.25):
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    torch.manual_seed(200024)
    np.random.seed(200024)
    random.seed(200024)
    torch.cuda.manual_seed(200024)
    torch.cuda.manual_seed_all(200024)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = StripedDataset(image_dir="data/train/images", label_dir="data/train/masks", transform=transform,
                                   label_rate=label_rate)
    val_dataset = StripedDataset(image_dir="data/val/images", label_dir="data/val/masks", transform=transform,
                                 label_rate=1.0)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # config_vit = get_CTranS_config()
    # Student = UCTransNet(config_vit,3, 1).to(device)
    # Ddynamic_Teacher = UCTransNet(config_vit,3, 1).to(device)
    # Static_Teacher = UCTransNet(config_vit,3, 1).to(device)
    # Static_Teacher.load_state_dict(torch.load("checkpoint/UCTransNet_0.0625.pth"))

    Student = MRSA_Net(3, 1).to(device)
    Ddynamic_Teacher = MRSA_Net(3, 1).to(device)
    Static_Teacher = MRSA_Net(3, 1).to(device)
    Static_Teacher.load_state_dict(torch.load("checkpoint/MRSA_Net0.25.pth"))


    Ddynamic_Teacher.load_state_dict(Student.state_dict())
    criterion = DiceLoss()
    consistency_criterion = ConsistencyLoss()
    optimizer = torch.optim.Adam(Student.parameters(), lr=1e-4)
    train(Student, Ddynamic_Teacher, Static_Teacher, criterion, consistency_criterion, optimizer, train_loader,
          val_loader, device)

if __name__ == "__main__":
    main()