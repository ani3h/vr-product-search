import os
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from retrieval import RetrievalPipeline
from metrics import compute_all_metrics
from utils import load_deepfashion_metadata

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Evaluation Script")
    parser.add_argument("--query_dir", type=str, required=True, help="Directory containing query images")
    parser.add_argument("--gallery_dir", type=str, required=False, help="Ignored since index is prebuilt")
    parser.add_argument("--index_path", type=str, required=True, help="Path to HNSW directory")
    parser.add_argument("--model_path", type=str, default=None, help="Path to Finetuned CLIP")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha used during indexing")
    parser.add_argument("--output", type=str, default="results.csv", help="CSV Output file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Initializing pipeline...")
    pipeline = RetrievalPipeline(
        index_path=args.index_path, 
        alpha=args.alpha, 
        k=15, 
        clip_model_path=args.model_path
    )
    
    # Load query ground truth
    df_queries = load_deepfashion_metadata(args.query_dir)
    if 'item_id' not in df_queries.columns:
        print("Error: Could not determine ground truth item_ids for queries.")
        return
        
    results = []
    
    print(f"Running evaluation on {len(df_queries)} queries...")
    for _, row in tqdm(df_queries.iterrows(), total=len(df_queries)):
        image_path = row['image_path']
        gt_id = row['item_id']
        
        try:
            img = Image.open(image_path).convert('RGB')
            # Run pipeline
            retrieved = pipeline.retrieve(img)
            retrieved_ids = [cand['item_id'] for cand in retrieved]
            
            metrics = compute_all_metrics(gt_id, retrieved_ids, k_list=[5, 10, 15])
            results.append(metrics)
        except Exception as e:
            print(f"Failed query {image_path}: {e}")
            
    if not results:
        print("No successful queries processed.")
        return
        
    # Aggregate and save
    df_res = pd.DataFrame(results)
    mean_metrics = df_res.mean().to_dict()
    
    print("\n--- Evaluation Results ---")
    for k, v in mean_metrics.items():
        print(f"{k}: {v:.4f}")
        
    df_res.to_csv(args.output, index=False)
    print(f"\nSaved raw query results to {args.output}")

if __name__ == "__main__":
    main()
