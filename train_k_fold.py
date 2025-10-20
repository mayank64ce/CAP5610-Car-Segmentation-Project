import torch
import torch.nn as nn
from dataloader import get_dataset_merged
from torch.utils.data import ConcatDataset, SubsetRandomSampler
from sklearn.model_selection import KFold
from model import create_vit_model
from tqdm import tqdm
import os
import logging

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def dice_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 2. * intersection / (union + 1e-8)
    return dice.mean()

if __name__ == "__main__":
    k_folds = 5
    num_epochs = 5
    batch_size = 32
    criterion = nn.BCELoss()
    checkpoint = "google/vit-base-patch16-224-in21k"
    lr = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set up logging
    logging.basicConfig(filename='training_log.log', level=logging.INFO, 
                format='%(asctime)s - %(levelname)s - %(message)s')

    torch.manual_seed(42)

    dataset_train_part = get_dataset_merged(split='train')
    dataset_val_part = get_dataset_merged(split='val')
    # dataset = ConcatDataset([dataset_train_part, dataset_val_part])
    dataset = dataset_train_part


    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                            dataset, 
                            batch_size=batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=batch_size, sampler=test_subsampler)
        
        best_model_path = os.path.join("saved_models_k_fold", f"best_model_fold_{fold}.pt")
        
        # Initialize the model for this run
        model = create_vit_model(checkpoint).to(device)
        model.apply(reset_weights)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for data in tqdm(trainloader, desc=f"Training (Fold {fold})"):
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
            for data in tqdm(testloader, desc=f"Validation (Fold {fold})"):
                images, masks = data['image'].to(device), data['mask'].to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                dice_scores.append(dice_score(outputs, masks).item())

            train_loss /= len(trainloader.dataset)
            val_loss /= len(testloader.dataset)
            average_dice_score = sum(dice_scores) / len(dice_scores)

            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Average Dice Score: {average_dice_score:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with Validation Loss: {val_loss:.4f}")

        logging.info(f"For Fold {fold}, average train loss: {train_loss:.4f}, average validation loss: {val_loss:.4f}, average dice score: {average_dice_score:.4f}")



        

