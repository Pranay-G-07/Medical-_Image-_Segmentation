import numpy as np
import cv2
import random

def apply_data_augmentation(image, mask, 
                           rotation_range=30,
                           scaling_range=(0.8, 1.2),
                           flip_prob=0.5,
                           translation_range=(10, 10)):
    """
    Applies random data augmentation transformations to both image and mask.
    
    Args:
        image: Input image (numpy array)
        mask: Corresponding mask (numpy array)
        rotation_range: Range of rotation in degrees
        scaling_range: Tuple of scaling factors (min, max)
        flip_prob: Probability of flipping
        translation_range: Range of translation in pixels (x, y)
        
    Returns:
        Augmented image and mask
    """
    # Create a copy of the original image and mask
    image_aug = image.copy()
    mask_aug = mask.copy()
    
    # Apply rotation
    if rotation_range > 0:
        angle = random.uniform(-rotation_range, rotation_range)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image_aug = cv2.warpAffine(image_aug, M, (w, h), 
                                   interpolation=cv2.INTER_NEAREST)
        mask_aug = cv2.warpAffine(mask_aug, M, (w, h), 
                                 interpolation=cv2.INTER_NEAREST)
    
    # Apply scaling
    if scaling_range is not None:
        scale = random.uniform(scaling_range[0], scaling_range[1])
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image_aug = cv2.resize(image_aug, new_size, 
                              interpolation=cv2.INTER_NEAREST)
        mask_aug = cv2.resize(mask_aug, new_size, 
                             interpolation=cv2.INTER_NEAREST)
    
    # Apply flipping
    if random.random() < flip_prob:
        # Randomly choose between horizontal and vertical flip
        if random.random() < 0.5:
            image_aug = cv2.flip(image_aug, 1)  # Horizontal flip
            mask_aug = cv2.flip(mask_aug, 1)
        else:
            image_aug = cv2.flip(image_aug, 0)  # Vertical flip
            mask_aug = cv2.flip(mask_aug, 0)
    
    # Apply translation
    if translation_range is not None:
        tx = random.randint(-translation_range[0], translation_range[0])
        ty = random.randint(-translation_range[1], translation_range[1])
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image_aug = cv2.warpAffine(image_aug, M, (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        mask_aug = cv2.warpAffine(mask_aug, M, (mask.shape[1], mask.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
    
    return image_aug, mask_aug
