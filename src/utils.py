import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def create_splits(csv_file, output_dir, test_size=0.2, val_size=0.1, seed=42):
    df = pd.read_csv(csv_file)
    
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=seed, stratify=train_df['label'])

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"Splits saved in {output_dir}")

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")