import torch
import os
from torchvision import transforms
from PIL import Image
from model import create_vit_model
from dataloader import get_dataloader
from tqdm import tqdm

def predict_mask(model, image, device):
    """Apply model prediction to an input image and return a binary mask."""
    # Assuming the transformation is similar to training
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    image_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
    output = output.squeeze(0).cpu()  # Remove batch dimension
    binary_mask = (output > 0.5).byte()  # Binarize the output
    return Image.fromarray(binary_mask.numpy()[0] * 255, mode='L')  # Convert to grayscale image, multiply by 255 for PNG saving

def process_fold(fold_id, segmentation_mode, loss_func, device):
    """Process each fold to predict and save masks using the best model for that fold."""
    model_path = f"saved_models/best_model_fold_{fold_id}_{segmentation_mode}_{loss_func}.pt"
    model = create_vit_model("google/vit-base-patch16-224-in21k", mode=segmentation_mode).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    val_loader = get_dataloader(fold_id, base_dir='dataset', split='val', batch_size=1)

    output_dir = f"dataset/{fold_id}/val/masked_predicted"
    os.makedirs(output_dir, exist_ok=True)

    for i, data in enumerate(tqdm(val_loader, desc=f"Processing fold {fold_id}")):
        image = data['image']
        filename = data['filename'][0]  # Assuming the batch size is 1
        predicted_mask = predict_mask(model, image[0], device)  # Process the first (and only) image in the batch
        mask_path = os.path.join(output_dir, filename)
        predicted_mask.save(mask_path)
        print(f"Saved predicted mask to {mask_path}")

def main():
    """Run mask prediction for each fold."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentation_mode = "conv"
    loss_func = "mse"
    for fold_id in range(1):  # Assuming there are 5 folds
        process_fold(fold_id, segmentation_mode, loss_func, device)

if __name__ == "__main__":
    main()
