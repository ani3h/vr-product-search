import os
import random
import numpy as np
import torch
import pandas as pd
from PIL import Image

def set_seed(seed=42):
    # Set seeds for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DeepFashionDataset(torch.utils.data.Dataset):
    # Dataset wrapper for DeepFashion.
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
            
        meta = {}
        if 'bbox' in row and pd.notna(row['bbox']).all() if isinstance(row.get('bbox'), tuple) else pd.notna(row.get('bbox')):
            meta['bbox'] = row['bbox']
        if 'gt_description' in row and pd.notna(row['gt_description']):
            meta['gt_description'] = row['gt_description']
            
        return image, item_id, image_path, meta

def load_deepfashion_metadata(data_dir, split=None):
    # Scans the data directory for images to mock/build a dataframe mapping
    # image paths to item_ids. In DeepFashion, item_ids are usually folders (e.g. id_0000001).
    records = []
    
    labels_path = os.path.join(data_dir, 'list_eval_partition.txt')
    bbox_path = os.path.join(data_dir, 'list_bbox_inshop.txt')
    desc_path = os.path.join(data_dir, 'list_description_inshop.json')

    if os.path.exists(labels_path):
        # Format usually: image_name item_id evaluation_status
        df = pd.read_csv(labels_path, sep=r'\s+', skiprows=1)
        
        # Merge bounding boxes if available
        if os.path.exists(bbox_path):
            df_bbox = pd.read_csv(bbox_path, sep=r'\s+', skiprows=1)
            # image_name  clothes_type  pose_type  x_1  y_1  x_2  y_2
            df = pd.merge(df, df_bbox, on='image_name', how='left')
            
        # Merge descriptions if available
        if os.path.exists(desc_path):
            import json
            with open(desc_path, 'r') as f:
                desc_json = json.load(f)
            # JSON is list of dicts: {"item": id, "color": str, "description": [...] }
            # Convert list of descriptions to single string
            desc_records = []
            for d in desc_json:
                text_desc = " ".join(d.get("description", []))
                desc_records.append({
                    "item_id": d.get("item"),
                    "gt_color": d.get("color"),
                    "gt_description": text_desc
                })
            df_desc = pd.DataFrame(desc_records)
            df = pd.merge(df, df_desc, on='item_id', how='left')

        # Format records for Dataset API
        if 'image_name' in df.columns and 'item_id' in df.columns:
            for _, row in df.iterrows():
                path = os.path.join(data_dir, row['image_name'])
                record = {
                    'image_path': path,
                    'item_id': row['item_id'],
                    'split': row.get('evaluation_status', 'train')
                }
                if 'x_1' in row:
                    record['bbox'] = (row['x_1'], row['y_1'], row['x_2'], row['y_2'])
                if 'gt_description' in row:
                    record['gt_description'] = row['gt_description']
                records.append(record)
            
            df_final = pd.DataFrame(records)
            if split:
                df_final = df_final[df_final['split'] == split].reset_index(drop=True)
            return df_final
            
    # Fallback: scan folders
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                # item_id derived from the parent folder mapping, e.g. /id_000001/01_1_front.jpg
                item_id = os.path.basename(os.path.dirname(path))
                records.append({'image_path': path, 'item_id': item_id})
                
    return pd.DataFrame(records)

