import torch
import torch.nn as nn
import cv2
import numpy as np
import scipy.ndimage as ndimage
import logging
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def validate(model, criterion, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():
        for images, masks, has_labels, img_names in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            has_labels = has_labels.to(device)
            outputs = model(images)

            # Compute losses based on label availability
            if has_labels.any():
                labeled_idx = has_labels.bool()
                loss = criterion(outputs[labeled_idx], masks[labeled_idx])
                val_loss += loss.item()

    # Average losses
    val_loss /= len(val_loader)
    model.train()  # Set the model back to training mode
    logging.info(f"Validation Loss: {val_loss:.4f}")
    return val_loss  # Return the average validation loss


def update_teacher_model(student_model, teacher_model, alpha):
    for teacher_params, student_params in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_params.data = alpha * teacher_params.data + (1.0 - alpha) * student_params.data

def count_connected_components(image):
    image_np = image.cpu().numpy()
    labeled_array, num_features = ndimage.label(image_np > 0.5)
    return num_features

def calculate_linearity(binary_mask):
    points = np.column_stack(np.where(binary_mask.cpu().numpy() > 0))
    if len(points) < 2:
        return 0
    mean_points = np.mean(points, axis=0)
    centered_points = points - mean_points
    covariance_matrix = np.cov(centered_points, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[sorted_indices]
    eigvecs_sorted = eigvecs[:, sorted_indices]
    linearity = eigvals_sorted[0] / max(eigvals_sorted[1], 1e-10)
    return linearity


def save_image(tensor, filename):
    image = tensor.cpu().clone().squeeze(0)  # Remove batch dimension
    image = image.numpy()  # Convert tensor to numpy array
    image = image * 255  # Scale the image to 0-255
    cv2.imwrite(filename, image)


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


def iou_score(y_true, y_pred, smooth=1e-6):
    y_true_int = y_true.byte()
    y_pred_int = y_pred.byte()
    intersection = (y_true_int & y_pred_int).float().sum()
    union = (y_true_int | y_pred_int).float().sum()
    iou = (intersection + smooth) / (union + smooth)
    return iou

def calculate_detection_and_false_alarm_rates(predicted_mask, ground_truth_mask):
    detection_flag = False
    false_alarm_pixels = 0
    total_pixels = ground_truth_mask.numel()

    predicted_binary = (predicted_mask > 0).byte()
    ground_truth_binary = (ground_truth_mask > 0).byte()

    # Calculate Intersection and Union for IOU
    intersection = (predicted_binary & ground_truth_binary).float().sum().item()
    union = (predicted_binary | ground_truth_binary).float().sum().item()
    iou = intersection / union if union != 0 else 0

    # Detection is successful if IOU is greater than 0.5
    if iou > 0.5:
        detection_flag = True

    # Calculate False Alarm by finding non-intersecting predicted areas
    false_alarm_pixels = (predicted_binary & (~ground_truth_binary)).float().sum().item()

    return detection_flag, false_alarm_pixels, total_pixels