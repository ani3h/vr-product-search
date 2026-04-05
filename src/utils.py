import os
import random
import numpy as np
import torch
import pandas as pd
from PIL import Image

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DeepFashionDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for DeepFashion.
    Assumes a CSV or DataFrame is provided with ['image_path', 'item_id'].
    """
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        image_path = row['image_path']
        item_id = row['item_id']
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Fallback if image corrupted or not found
            image = Image.new('RGB', (224, 224), color='white')
            
        if self.transform is not None:
            image = self.transform(image)
            
        return image, item_id, image_path

def load_deepfashion_metadata(data_dir):
    """
    Scans the data directory for images to mock/build a dataframe mapping
    image paths to item_ids. In DeepFashion, item_ids are usually folders (e.g. id_0000001).
    """
    records = []
    
    # Check if a labels file exists
    labels_path = os.path.join(data_dir, 'list_eval_partition.txt')
    if os.path.exists(labels_path):
        # Format usually: image_name item_id evaluation_status
        df = pd.read_csv(labels_path, sep=r'\s+', skiprows=1)
        # Fix columns based on actual DeepFashion format if needed. Here we assume generic format:
        if 'image_name' in df.columns and 'item_id' in df.columns:
            for _, row in df.iterrows():
                path = os.path.join(data_dir, row['image_name'])
                records.append({
                    'image_path': path,
                    'item_id': row['item_id'],
                    'split': row.get('evaluation_status', 'train')
                })
            return pd.DataFrame(records)
            
    # Fallback: scan folders
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                # item_id derived from the parent folder mapping, e.g. /id_000001/01_1_front.jpg
                item_id = os.path.basename(os.path.dirname(path))
                records.append({'image_path': path, 'item_id': item_id})
                
    return pd.DataFrame(records)

