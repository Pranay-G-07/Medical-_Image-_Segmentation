import matplotlib.pyplot as plt
import numpy as np

# Load images (replace 'image_path', 'mask_path', 'pred_path' with actual paths)
original_image = np.load('image_path.npy')
ground_truth_mask = np.load('mask_path.npy')
predicted_mask = np.load('pred_path.npy')

# Normalize images to [0, 1] range
original_image = original_image.astype('float32') / 255
ground_truth_mask = ground_truth_mask.astype('float32') / 255
predicted_mask = predicted_mask.astype('float32') / 255

# Create figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# Overlay predicted mask on original image
overlay = np.copy(original_image)
overlay = np.dstack((overlay, overlay, overlay))
predicted_mask_rgb = np.dstack((predicted_mask, predicted_mask, predicted_mask))
overlay[predicted_mask > 0.5] = predicted_mask_rgb[predicted_mask > 0.5] * 0.3 + overlay[predicted_mask > 0.5] * 0.7

axes[1].imshow(overlay)
axes[1].set_title('Predicted Mask Overlay')
axes[1].axis('off')

axes[2].imshow(predicted_mask, cmap='gray')
axes[2].set_title('Predicted Mask')
axes[2].axis('off')

plt.tight_layout()
plt.show()
