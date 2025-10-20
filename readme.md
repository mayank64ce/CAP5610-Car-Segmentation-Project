
# Binary Segmentation with Pretrained ViT Backbone

  

## Steps to run the repo:
### 1. Clone the repo and change to the current directory:
```bash
git clone https://github.com/rajatmodi62/ml_project.git
cd ml_project/mayank
```
### 2. Setup the environment with conda:
```bash
conda env create -f environment.yaml
```
### 3. Downloading and setting up data:
Download the dataset from [here](https://drive.google.com/file/d/1wCJuBxbw166omJS7iMLdOaE5pEaJ-en8/view?usp=sharing). It is already split in 5 folds so just place each fold in the `dataset` directory.



### 4. Using `train.py`:
```bash
usage: train.py [-h] [--checkpoint CHECKPOINT] [--lr LR] [--num_epochs NUM_EPOCHS] --fold FOLD [--batch_size BATCH_SIZE] [--base_dir BASE_DIR] [--seed SEED]
                [--segmentation_mode SEGMENTATION_MODE]

Train ViT segmentation model on a specific fold

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Model checkpoint
  --lr LR               Learning rate
  --num_epochs NUM_EPOCHS
                        Number of epochs
  --fold FOLD           Fold to train on (0-4)
  --batch_size BATCH_SIZE
                        Batch size
  --base_dir BASE_DIR   Base directory for dataset
  --seed SEED           Random seed for reproducibility
  --segmentation_mode SEGMENTATION_MODE
                        Segmentation head mode
```
So for example, if you want to train the model with `conv` head and a learning rate of `2e-5` on fold 0, you can run the following command:
```bash
python train.py --fold 0 --lr 0.00002 --segmentation_mode conv
```

### 5. Logs and saved models:
The training logs can be found in the `logs` directory and the best models can be found in the `saved_models` directory.