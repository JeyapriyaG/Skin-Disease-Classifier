import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits(csv_path, out_dir, test_size=0.15, val_size=0.15):
    df = pd.read_csv(csv_path)
    trainval, test = train_test_split(df, test_size=test_size, stratify=df['dx'], random_state=42)
    train, val = train_test_split(trainval, test_size=val_size/(1-test_size), stratify=trainval['dx'], random_state=42)
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(out_dir, 'val.csv'), index=False)
    test.to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    print("Data splits created successfully!")
