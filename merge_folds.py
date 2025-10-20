import os
import shutil

def merge_data(base_dir, target_dir, data_type):
    """
    Merges all images and masks from multiple folds into specific subdirectories within
    a single directory for the specified data type without changing filenames.

    Args:
    base_dir (str): The base directory where folds are located.
    target_dir (str): The target directory where merged images and masks will be stored.
    data_type (str): 'train' or 'val' to specify which type of data to merge.
    """
    # Loop through each fold
    for fold in range(5):  # Assuming there are 5 folds; adjust if necessary
        fold_dir = os.path.join(base_dir, str(fold), data_type)
        fold_images_dir = os.path.join(fold_dir, 'images')
        fold_masks_dir = os.path.join(fold_dir, 'masks')

        # Target directories for this fold
        target_images_fold_dir = os.path.join(target_dir, data_type, 'images')
        target_masks_fold_dir = os.path.join(target_dir, data_type, 'masks')

        # Create target directories if they do not exist
        os.makedirs(target_images_fold_dir, exist_ok=True)
        os.makedirs(target_masks_fold_dir, exist_ok=True)

        # Copy images from each fold to the corresponding target images directory
        for image_file in os.listdir(fold_images_dir):
            src_image_path = os.path.join(fold_images_dir, image_file)
            dest_image_path = os.path.join(target_images_fold_dir, image_file)
            shutil.copy(src_image_path, dest_image_path)

        # Copy masks from each fold to the corresponding target masks directory
        for mask_file in os.listdir(fold_masks_dir):
            src_mask_path = os.path.join(fold_masks_dir, mask_file)
            dest_mask_path = os.path.join(target_masks_fold_dir, mask_file)
            shutil.copy(src_mask_path, dest_mask_path)

        print(f"Finished merging {data_type} fold {fold}")

if __name__ == "__main__":
    base_dir = 'dataset'  # Path to the dataset with folds
    target_dir = 'merged_data'  # Path to the directory where merged data will be stored
    merge_data(base_dir, target_dir, 'train')
    merge_data(base_dir, target_dir, 'val')
    print("All folds have been merged successfully for both train and val.")
