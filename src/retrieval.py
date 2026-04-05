import hnswlib
import pickle
import os
import torch
import numpy as np
from PIL import Image
from typing import List, Dict

from detection import YOLODetector
from embedding import CLIPEmbedder
from reranking import BLIP2Reranker

class RetrievalPipeline:
    def __init__(self, index_path: str, alpha: float, k: int = 15, clip_model_path=None):
        self.detector = YOLODetector()
        self.embedder = CLIPEmbedder()
        
        if clip_model_path and os.path.exists(clip_model_path):
            self.embedder.model.load_state_dict(torch.load(clip_model_path, map_location=self.embedder.device))
            
        self.reranker = BLIP2Reranker()
        self.k = k
        self.alpha = alpha
        
        # Load index and metadata
        idx_file = os.path.join(index_path, f"index_alpha_{alpha}.bin")
        meta_file = os.path.join(index_path, f"metadata_alpha_{alpha}.pkl")
        
        with open(meta_file, 'rb') as f:
            self.metadata = pickle.load(f)
            
        dim = self.embedder.model.visual.output_dim
        self.index = hnswlib.Index(space='ip', dim=dim)
        self.index.load_index(idx_file, max_elements=len(self.metadata))

    def retrieve(self, query_image: Image.Image) -> List[Dict]:
        """
        Online pipeline:
        1. Crop query
        2. Visual Encode with CLIP
        3. HNSW Search
        4. Re-rank with BLIP-2 ITM
        """
        # Step 1: Crop
        cropped = self.detector.crop_primary_item(query_image)
        
        # Step 2: Encode with visually ONLY since no caption for query
        # But wait - if offline was fusion alpha * vis + (1-alpha) * text, 
        # online is recommended in prompt "Encode with CLIP visual encoder only"
        # However, it says "alpha" is a parameter for fusion embedding. If we only use visual for query, we do so:
        v_vis = self.embedder.get_visual_embedding(cropped)
        query_vector = v_vis.cpu().numpy()[0]
        
        # Step 3: Search candidates
        # search_knn returns labels and distances
        labels, distances = self.index.knn_query(query_vector, k=self.k)
        
        candidates = []
        for i, label in enumerate(labels[0]):
            cand = self.metadata[label]
            cand['ann_score'] = 1 - distances[0][i] # HNSW IP distance is 1 - inner_product
            candidates.append(cand)
            
        # Step 4: Re-rank
        ranked_candidates = self.reranker.rerank(cropped, candidates)
        
        return ranked_candidates
