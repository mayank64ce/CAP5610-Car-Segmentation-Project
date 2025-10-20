import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import create_vit_model
from dataloader import get_dataloader

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dice_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 2. * intersection / (union + 1e-8)
    return dice.mean()

def main(args):
    # Set up logging
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)
    log_file_path = os.path.join(log_directory, f"train_{args.fold}_{args.segmentation_mode}_bce.log")
    
    logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    logging.info(f"Training on fold {args.fold}")
    logging.info(f"Using loss function: {args.loss_func}")
    logging.info(f"Using seed: {args.seed}")
    logging.info(f"Using segmentation mode: {args.segmentation_mode}")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_vit_model(checkpoint=args.checkpoint, mode=args.segmentation_mode).to(device)
    
    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader = get_dataloader(args.fold, base_dir=args.base_dir, split='train', batch_size=args.batch_size)
    val_loader = get_dataloader(args.fold, base_dir=args.base_dir, split='val', batch_size=args.batch_size)

    best_val_loss = float('inf')
    best_model_path = os.path.join("saved_models", f"best_model_fold_{args.fold}_{args.segmentation_mode}_bce.pt")

    logging.info(f"Starting training with model: {model}")


    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        for data in tqdm(train_loader, desc=f"Training (Fold {args.fold})"):
            images, masks = data['image'].to(device), data['mask'].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        dice_scores = []
        for data in tqdm(val_loader, desc=f"Validation (Fold {args.fold})"):
            images, masks = data['image'].to(device), data['mask'].to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)
            dice_scores.append(dice_score(outputs, masks).item())

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        average_dice_score = sum(dice_scores) / len(dice_scores)

        logging.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Average Dice Score: {average_dice_score:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved with Validation Loss: {val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT segmentation model on a specific fold")
    parser.add_argument("--checkpoint", type=str, default="google/vit-base-patch16-224-in21k", help="Model checkpoint")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--fold", type=int, required=True, help="Fold to train on (0-4)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--base_dir", type=str, default="dataset", help="Base directory for dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    # parser.add_argument("--loss_func", type=str, default="bce", help="Loss function to use")
    parser.add_argument("--segmentation_mode", type=str, default="mlp", help="Segmentation head mode")
    args = parser.parse_args()
    main(args)