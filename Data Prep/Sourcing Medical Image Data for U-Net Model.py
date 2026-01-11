import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load the NIfTI file
image = nib.load('path/to/image.nii.gz')
mask = nib.load('path/to/mask.nii.gz')

# Convert to numpy arrays
image_data = image.get_fdata()
mask_data = mask.get_fdata()

# Visualize the data
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image_data[:, :, image_data.shape[2]//2], cmap='gray')
plt.title('MRI Scan')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mask_data[:, :, mask_data.shape[2]//2], cmap='gray')
plt.title('Segmentation Mask')
plt.axis('off')

plt.show()
