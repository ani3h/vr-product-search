import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip

from utils import set_seed, load_deepfashion_metadata, DeepFashionDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/clip_finetuned")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    return parser.parse_args()

class SupConLoss(torch.nn.Module):
    # Supervised Contrastive Learning loss.
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # Max for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Mask out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        loss = - mean_log_prob_pos
        return loss.mean()

def train():
    args = parse_args()
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained="openai", device=device)
    
    # Freeze text encoder
    for param in model.transformer.parameters():
        param.requires_grad = False
    model.token_embedding.requires_grad = False
    model.ln_final.requires_grad = False
    model.text_projection.requires_grad = False
    
    # Optionally freeze early blocks of Vision encoder to save memory
    if hasattr(model.visual, 'transformer'):
        for i, block in enumerate(model.visual.transformer.resblocks):
            # Keep last 4 blocks unfrozen
            if i < len(model.visual.transformer.resblocks) - 4:
                for param in block.parameters():
                    param.requires_grad = False
                    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = SupConLoss()
    
    df = load_deepfashion_metadata(args.data_dir, split='train')
    # Convert item_id strings to ints for the loss function
    unique_ids = df['item_id'].unique()
    id_to_int = {v: k for k, v in enumerate(unique_ids)}
    df['label'] = df['item_id'].map(id_to_int)
    
    dataset = DeepFashionDataset(df, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, item_ids, _, _ in pbar:
            images = images.to(device)
            # Map item_ids to torch labels
            labels = torch.tensor([id_to_int[i] for i in item_ids]).to(device)
            
            optimizer.zero_grad()
            features = model.encode_image(images)
            features = F.normalize(features, dim=-1)
            
            loss = loss_fn(features, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader)}")
        
    output_path = os.path.join(args.output_dir, f"clip_finetuned_seed_{args.seed}.pt")
    torch.save(model.state_dict(), output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    train()
