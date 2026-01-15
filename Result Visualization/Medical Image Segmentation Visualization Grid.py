def load_test_data(image_paths, mask_paths):
    test_images = []
    test_masks = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        image = nib.load(img_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        
        # Normalize image to 0-1 range
        image = image / image.max()
        
        test_images.append(image)
        test_masks.append(mask)
    
    return np.array(test_images), np.array(test_masks)
