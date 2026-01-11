import numpy as np
import SimpleITK as sitk

def normalize_mri_image(image_path):
    """
    Normalize an MRI image by clipping intensities and scaling to [0, 1] range.
    """
    # Read the image
    image = sitk.ReadImage(image_path)
    # Convert to numpy array
    image_array = sitk.GetArrayFromImage(image)
    
    # Clip intensities to 1st and 99th percentiles
    lower_percentile = np.percentile(image_array, 1)
    upper_percentile = np.percentile(image_array, 99)
    
    # Clip the image array
    clipped_image = np.clip(image_array, lower_percentile, upper_percentile)
    
    # Scale to 0-1 range
    normalized_image = (clipped_image - lower_percentile) / (upper_percentile - lower_percentile)
    
    return normalized_image

def normalize_mask_image(image_path):
    """
    Normalize a mask image to binary values (0 or 1).
    """
    # Read the image
    image = sitk.ReadImage(image_path)
    # Convert to numpy array
    image_array = sitk.GetArrayFromImage(image)
    
    # Convert to binary
    normalized_mask = np.where(image_array > 0, 1, 0)
    
    return normalized_mask

# Example usage
mri_image_path = "path_to_your_mri_image.nii.gz"
mask_image_path = "path_to_your_mask_image.nii.gz"

# Normalize MRI image
normalized_mri = normalize_mri_image(mri_image_path)

# Normalize mask image
normalized_mask = normalize_mask_image(mask_image_path)

# Save the normalized images
# For MRI
normalized_mri_image = sitk.GetImageFromArray(normalized_mri)
sitk.WriteImage(normalized_mri_image, "normalized_mri_image.nii.gz")

# For Mask
normalized_mask_image = sitk.GetImageFromArray(normalized_mask.astype(np.uint8))
sitk.WriteImage(normalized_mask_image, "normalized_mask_image.nii.gz")
