import numpy as np

def dice_coefficient(y_true, y_pred):
    """
    Computes the Dice coefficient for two binary masks.
    """
    # Flatten the masks to 1D arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate intersection and union
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    
    # Handle special cases where both are empty
    if union == 0:
        return 1.0  # If both masks are empty, consider perfect similarity
    
    dice = (2.0 * intersection) / union
    return dice

def iou(y_true, y_pred):
    """
    Computes the Intersection over Union (IoU) for two binary masks.
    """
    # Flatten the masks to 1D arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Calculate intersection and union
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection  # Avoid double counting
    
    # Handle special cases where union is zero
    if union == 0:
        return 1.0  # If both masks are empty, consider perfect overlap
    
    iou = intersection / union
    return iou
