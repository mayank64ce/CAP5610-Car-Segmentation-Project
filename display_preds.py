import matplotlib.pyplot as plt
from PIL import Image
import os

def load_and_plot_images(image_path, truth_mask_path, predicted_mask_path):
    # Load the images
    image = Image.open(image_path).convert('RGB')
    truth_mask = Image.open(truth_mask_path).convert('L')
    predicted_mask = Image.open(predicted_mask_path).convert('L')

    image = image.resize((224, 224))
    truth_mask = truth_mask.resize((224, 224))
    predicted_mask = predicted_mask.resize((224, 224))

    # Create a figure to display the results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Set up a plot with three subplots
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')  # Hide axes for better visualization

    axs[1].imshow(truth_mask, cmap='gray')
    axs[1].set_title('Ground Truth Mask')
    axs[1].axis('off')

    axs[2].imshow(predicted_mask, cmap='gray')
    axs[2].set_title('Predicted Mask')
    axs[2].axis('off')

    # plt.show()
    # Save the figure
    output_path = os.path.join('output.png')
    plt.savefig(output_path)
    plt.close()

# Example usage:
base_dir = 'dataset/0/val/'  # Base directory containing your datasets
image_filename = '0d3adbbc9a8b_03.jpg'  # Example image filename
truth_mask_filename = image_filename.replace('.jpg', '_mask.gif')  # Construct ground truth filename
predicted_mask_filename = image_filename # Construct predicted mask filename

image_path = os.path.join(base_dir, 'images', image_filename)
truth_mask_path = os.path.join(base_dir, 'masks', truth_mask_filename)
predicted_mask_path = os.path.join(base_dir, 'masked_predicted', predicted_mask_filename)  # Adjust if your path differs

load_and_plot_images(image_path, truth_mask_path, predicted_mask_path)
