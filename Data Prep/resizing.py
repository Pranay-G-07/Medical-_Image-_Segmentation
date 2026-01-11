import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Define the target size
target_size = (256, 256)

def resize_image(image_path, target_size):
    """
    Resize an image to a specified target size while maintaining aspect ratio.
    """
    # Read the image
    image = cv2.imread(str(image_path))
    
    # Get current dimensions
    height, width = image.shape[:2]
    
    # Calculate new dimensions while maintaining aspect ratio
    scale = min(target_size[0]/width, target_size[1]/height)
    new_size = (int(width*scale), int(height*scale))
    
    # Resize the image
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    
    # If the image has an alpha channel, handle it appropriately
    if image.shape[2] == 4:
        alpha = resized[:, :, 3] / 255.0
        resized = resized[:, :, :3] * alpha
        resized = resized.astype(np.uint8)
    
    return resized

# Example usage:
# image_path = Path("path/to/your/image.jpg")
# resized_image = resize_image(image_path, target_size)
# plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
# plt.show()

# For grayscale images:
def resize_grayscale(image_path, target_size):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    return resized

# Example usage for grayscale:
# gray_image_path = Path("path/to/your/grayscale_image.png")
# resized_gray = resize_grayscale(gray_image_path, target_size)
# plt.imshow(resized_gray, cmap='gray')
# plt.show()

# Normalization
def normalize_image(image):
    """
    Normalize image pixel values to [0,1] range.
    """
    return image / 255.0

# Data Augmentation
def apply_data_augmentation(image):
    """
    Apply random augmentations to the image.
    """
    # Rotate
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    angle = np.random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    # Flip
    if np.random.random() > 0.5:
        flipped = cv2.flip(rotated, 1)
    else:
        flipped = rotated
    
    # Random scaling
    scale = np.random.uniform(0.8, 1.2)
    scaled = cv2.resize(flipped, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    return scaled

# Example usage:
# augmented_image = apply_data_augmentation(resized_image)
# plt.imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB))
# plt.show()

# Dataset Splitting
def split_dataset(image_paths, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split image paths into training, validation, and test sets.
    """
    np.random.seed(42)
    indices = np.random.permutation(len(image_paths))
    
    train_idx = indices[:int(train_ratio * len(image_paths))]
    val_idx = indices[int(train_ratio * len(image_paths)):int((train_ratio + val_ratio) * len(image_paths))]
    test_idx = indices[int((train_ratio + val_ratio) * len(image_paths):]]
    
    train_paths = [image_paths[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    test_paths = [image_paths[i] for i in test_idx]
    
    return train_paths, val_paths, test_paths

# Example usage:
# image_paths = list(Path("path/to/your/images").glob("*.png"))
# train_paths, val_paths, test_paths = split_dataset(image_paths)
