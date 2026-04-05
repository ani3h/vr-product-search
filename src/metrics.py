import numpy as np

def score_recall_at_k(gt_id, retrieved_ids, k):
    """Recall@K is 1 if any retrieved item in top K matches gt_id, 0 otherwise."""
    return 1.0 if gt_id in retrieved_ids[:k] else 0.0

def score_ndcg_at_k(gt_id, retrieved_ids, k):
    """Normalized Discounted Cumulative Gain at K."""
    dcg = 0.0
    for i, pred in enumerate(retrieved_ids[:k]):
        rel = 1.0 if pred == gt_id else 0.0
        dcg += rel / np.log2(i + 2) # i=0 -> log2(2)=1
        
    # IDCG is always 1 for a single relevant item. If multiple exist, IDCG could be higher, 
    # but in our setup there is 1 correct 'item_id'.
    idcg = 1.0 
    return dcg / idcg

def score_map_at_k(gt_id, retrieved_ids, k):
    """Mean Average Precision at K."""
    ap = 0.0
    hits = 0.0
    for i, pred in enumerate(retrieved_ids[:k]):
        if pred == gt_id:
            hits += 1.0
            ap += hits / (i + 1.0)
    # Divided by number of relevant items in the dataset, but usually 1 target gt_id considered for AP here.
    return ap

def compute_all_metrics(gt_id, retrieved_ids, k_list=[5, 10, 15]):
    metrics = {}
    for k in k_list:
        metrics[f'Recall@{k}'] = score_recall_at_k(gt_id, retrieved_ids, k)
        metrics[f'NDCG@{k}'] = score_ndcg_at_k(gt_id, retrieved_ids, k)
        metrics[f'mAP@{k}'] = score_map_at_k(gt_id, retrieved_ids, k)
    return metrics
