import os
import argparse
import pickle
import hnswlib
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from utils import set_seed, load_deepfashion_metadata, DeepFashionDataset
from detection import YOLODetector
from captioning import BLIP2Captioner
from embedding import CLIPEmbedder

def parse_args():
    parser = argparse.ArgumentParser(description="Offline Indexing Pipeline")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw gallery dataset")
    parser.add_argument("--index_path", type=str, required=True, help="Where to save the HNSW index & metadata")
    parser.add_argument("--alpha", type=float, default=0.5, help="Fusion weight for vision embedding")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--clip_model_path", type=str, default=None, help="Path to fine-tuned CLIP or None for frozen")
    return parser.parse_args()

def build_index():
    args = parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.index_path, exist_ok=True)
    
    print(f"Loading metadata from {args.data_dir} (gallery split)...")
    df = load_deepfashion_metadata(args.data_dir, split='gallery')
    print(f"Found {len(df)} images.")
    dataset = DeepFashionDataset(df)
    
    # Initialize models
    detector = YOLODetector()
    captioner = BLIP2Captioner()
    embedder = CLIPEmbedder()
    
    # If using fine-tuned model
    if args.clip_model_path and os.path.exists(args.clip_model_path):
        print(f"Loading fine-tuned CLIP weights from {args.clip_model_path}...")
        embedder.model.load_state_dict(torch.load(args.clip_model_path, map_location=embedder.device))
        
    dim = embedder.model.visual.output_dim
    # Initialize HNSW index
    # IP (Inner Product) since embeddings are L2 normalized, IP = Cosine Similarity
    index = hnswlib.Index(space='ip', dim=dim)
    index.init_index(max_elements=max(len(df), 100), ef_construction=200, M=16)
    
    metadata = {}
    
    print("Starting offline indexing...")
    for idx in tqdm(range(len(dataset))):
        try:
            image, item_id, path, meta = dataset[idx]
            
            # 1. Detection
            cropped = detector.crop_primary_item(image, gt_bbox=meta.get('bbox'))
            
            # 2. Captioning
            caption = captioner.generate_caption(cropped, gt_caption=meta.get('gt_description'))
            
            # 3. Embedding fusion
            emb_tensor = embedder.compute_fusion_embedding(cropped, caption, args.alpha)
            emb_np = emb_tensor.cpu().numpy()[0] # Shape [dim]
            
            # 4. Add to index
            index.add_items(emb_np, idx)
            
            # Store metadata
            metadata[idx] = {
                'item_id': item_id,
                'image_path': path,
                'caption': caption
            }
        except Exception as e:
            print(f"Failed processing idx {idx} ({path}): {e}")
            
    # Save index and metadata
    hnsw_file = os.path.join(args.index_path, f"index_alpha_{args.alpha}.bin")
    meta_file = os.path.join(args.index_path, f"metadata_alpha_{args.alpha}.pkl")
    
    index.save_index(hnsw_file)
    with open(meta_file, 'wb') as f:
        pickle.dump(metadata, f)
        
    print(f"Index successfully built and saved to {args.index_path}!")

if __name__ == "__main__":
    build_index()
