#!/usr/bin/env python3

import os
import json
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import math

DATASET_DIR = Path.home() / "shop-the-look-dataset"
MODEL_DIR = Path.home() / "shop-the-look-model"
EMBEDDINGS_DIR = MODEL_DIR / "embeddings"

class CTLDatasetV2(Dataset):
    def __init__(self, pairs, scene_embeddings, product_embeddings, categories, mode="train"):
        self.scene_embeddings = scene_embeddings
        self.product_embeddings = product_embeddings
        self.categories = categories
        self.mode = mode
        
        # Filter pairs to only include those with valid embeddings
        self.valid_pairs = [
            p for p in pairs 
            if p["scene"] in scene_embeddings and p["product"] in product_embeddings
        ]
        print(f"{mode}: {len(self.valid_pairs)}/{len(pairs)} valid pairs")
        
        # Build category mapping
        self.cat_to_idx = {}
        self.idx_to_cat = {}
        for prod_id, cat in categories.items():
            # Use leaf category (most specific)
            leaf_cat = cat.split("|")[-1]
            if leaf_cat not in self.cat_to_idx:
                idx = len(self.cat_to_idx)
                self.cat_to_idx[leaf_cat] = idx
                self.idx_to_cat[idx] = leaf_cat
        self.num_categories = len(self.cat_to_idx)
        print(f"Found {self.num_categories} categories")
        
        # Build category -> products mapping for hard negative sampling
        self.cat_to_products = defaultdict(list)
        for prod_id in product_embeddings.keys():
            if prod_id in categories:
                leaf_cat = categories[prod_id].split("|")[-1]
                self.cat_to_products[leaf_cat].append(prod_id)
        
        # Build scene -> positive products mapping
        self.scene_to_products = defaultdict(set)
        for p in self.valid_pairs:
            self.scene_to_products[p["scene"]].add(p["product"])
        
        # All product IDs
        self.all_products = list(product_embeddings.keys())
    
    def get_category_idx(self, product_id):
        if product_id in self.categories:
            leaf_cat = self.categories[product_id].split("|")[-1]
            return self.cat_to_idx.get(leaf_cat, 0)
        return 0
    
    def sample_hard_negative(self, scene_id, positive_product_id):
        if positive_product_id in self.categories:
            leaf_cat = self.categories[positive_product_id].split("|")[-1]
            candidates = self.cat_to_products.get(leaf_cat, [])
            
            # Filter out positive products for this scene
            positives = self.scene_to_products[scene_id]
            candidates = [c for c in candidates if c not in positives]
            
            if candidates:
                return random.choice(candidates)
        return random.choice(self.all_products)
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        scene_id = pair["scene"]
        product_id = pair["product"]
        
        scene_emb = torch.tensor(self.scene_embeddings[scene_id], dtype=torch.float32)
        product_emb = torch.tensor(self.product_embeddings[product_id], dtype=torch.float32)
        
        # Get category index
        cat_idx = self.get_category_idx(product_id)
        
        # Sample hard negative for training
        if self.mode == "train":
            neg_product_id = self.sample_hard_negative(scene_id, product_id)
            neg_emb = torch.tensor(self.product_embeddings[neg_product_id], dtype=torch.float32)
        else:
            neg_emb = torch.zeros_like(product_emb)  # Placeholder for val
        
        return {
            "scene_emb": scene_emb,
            "product_emb": product_emb,
            "neg_emb": neg_emb,
            "cat_idx": cat_idx,
            "scene_id": scene_id,
            "product_id": product_id
        }


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.norm(x + self.net(x))


class CTLModelV2(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256, num_categories=50):
        super().__init__()
        
        self.category_embedding = nn.Embedding(num_categories, 64)

        self.scene_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

        self.product_encoder = nn.Sequential(
            nn.Linear(input_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

        self.log_temperature = nn.Parameter(torch.tensor(math.log(0.07)))
    
    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(min=0.01, max=1.0)
    
    def encode_scene(self, x):
        emb = self.scene_encoder(x)
        return F.normalize(emb, dim=-1)
    
    def encode_product(self, x, cat_idx):
        cat_emb = self.category_embedding(cat_idx)
        x = torch.cat([x, cat_emb], dim=-1)
        emb = self.product_encoder(x)
        return F.normalize(emb, dim=-1)
    
    def forward(self, scene_emb, product_emb, cat_idx):
        scene_features = self.encode_scene(scene_emb)
        product_features = self.encode_product(product_emb, cat_idx)
        return scene_features, product_features


def contrastive_loss(scene_features, product_features, temperature):
    logits = torch.matmul(scene_features, product_features.T) / temperature
    batch_size = scene_features.shape[0]
    labels = torch.arange(batch_size, device=scene_features.device)
    
    loss_s2p = F.cross_entropy(logits, labels)
    loss_p2s = F.cross_entropy(logits.T, labels)
    
    return (loss_s2p + loss_p2s) / 2


def triplet_loss(anchor, positive, negative, margin=0.3):
    pos_dist = 1 - (anchor * positive).sum(dim=-1)
    neg_dist = 1 - (anchor * negative).sum(dim=-1)
    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_data():
    print("Loading embeddings...")
    
    with open(EMBEDDINGS_DIR / "scene_embeddings.pkl", "rb") as f:
        scene_embeddings = pickle.load(f)
    with open(EMBEDDINGS_DIR / "product_embeddings.pkl", "rb") as f:
        product_embeddings = pickle.load(f)
    
    print(f"Loaded {len(scene_embeddings)} scene embeddings")
    print(f"Loaded {len(product_embeddings)} product embeddings")
    
    print("Loading dataset...")
    pairs = []
    with open(DATASET_DIR / "fashion.json") as f:
        for line in f:
            pairs.append(json.loads(line))
    
    print("Loading categories...")
    with open(DATASET_DIR / "fashion-cat.json") as f:
        categories = json.load(f)
    
    return pairs, scene_embeddings, product_embeddings, categories


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    total_contrastive = 0
    total_triplet = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        scene_emb = batch["scene_emb"].to(device)
        product_emb = batch["product_emb"].to(device)
        neg_emb = batch["neg_emb"].to(device)
        cat_idx = batch["cat_idx"].to(device)
        
        optimizer.zero_grad()
        
        scene_features, product_features = model(scene_emb, product_emb, cat_idx)
        neg_features = model.encode_product(neg_emb, cat_idx)
        
        # Combined loss
        loss_c = contrastive_loss(scene_features, product_features, model.temperature)
        loss_t = triplet_loss(scene_features, product_features, neg_features, margin=0.2)
        
        loss = loss_c + 0.5 * loss_t
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_contrastive += loss_c.item()
        total_triplet += loss_t.item()
    
    n = len(dataloader)
    return total_loss / n, total_contrastive / n, total_triplet / n


def evaluate(model, dataloader, device):
    model.eval()
    
    all_scene_features = []
    all_product_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            scene_emb = batch["scene_emb"].to(device)
            product_emb = batch["product_emb"].to(device)
            cat_idx = batch["cat_idx"].to(device)
            
            scene_features, product_features = model(scene_emb, product_emb, cat_idx)
            
            all_scene_features.append(scene_features.cpu())
            all_product_features.append(product_features.cpu())
    
    scene_features = torch.cat(all_scene_features)
    product_features = torch.cat(all_product_features)
    
    # Compute similarity matrix
    similarities = torch.matmul(scene_features, product_features.T)
    n = similarities.shape[0]
    
    ranks = []
    for i in range(n):
        sim_row = similarities[i]
        sorted_indices = torch.argsort(sim_row, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    
    metrics = {
        "recall@1": (ranks <= 1).mean(),
        "recall@5": (ranks <= 5).mean(),
        "recall@10": (ranks <= 10).mean(),
        "recall@50": (ranks <= 50).mean(),
        "mrr": (1.0 / ranks).mean(),
        "median_rank": np.median(ranks)
    }
    
    return metrics


def main():
    # Config
    BATCH_SIZE = 256  # Larger batch for better negatives
    EPOCHS = 50
    LR = 3e-4
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    PATIENCE = 10  # Early stopping
    DEVICE = "cpu"
    
    # Load data
    pairs, scene_embeddings, product_embeddings, categories = load_data()
    
    # Get embedding dim
    sample_emb = next(iter(scene_embeddings.values()))
    input_dim = len(sample_emb)
    print(f"Embedding dimension: {input_dim}")
    
    # Train/val split
    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    # Create datasets
    train_dataset = CTLDatasetV2(train_pairs, scene_embeddings, product_embeddings, categories, "train")
    val_dataset = CTLDatasetV2(val_pairs, scene_embeddings, product_embeddings, categories, "val")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Create model
    model = CTLModelV2(
        input_dim=input_dim, 
        hidden_dim=512, 
        output_dim=256, 
        num_categories=train_dataset.num_categories
    ).to(DEVICE)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    num_training_steps = len(train_loader) * EPOCHS
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    scheduler = get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps)
    
    # Training loop
    best_recall = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        
        train_loss, loss_c, loss_t = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        print(f"Train loss: {train_loss:.4f} (contrastive: {loss_c:.4f}, triplet: {loss_t:.4f})")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}, Temp: {model.temperature.item():.4f}")
        
        metrics = evaluate(model, val_loader, DEVICE)
        print(f"Val R@1: {metrics['recall@1']:.4f}, R@5: {metrics['recall@5']:.4f}, "
              f"R@10: {metrics['recall@10']:.4f}, R@50: {metrics['recall@50']:.4f}, MRR: {metrics['mrr']:.4f}")
        
        # Save best model
        if metrics['recall@10'] > best_recall:
            best_recall = metrics['recall@10']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'num_categories': train_dataset.num_categories,
                'cat_to_idx': train_dataset.cat_to_idx,
            }, MODEL_DIR / "best_model_v2.pt")
            print(f"✓ Saved best model (R@10: {best_recall:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{PATIENCE})")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\n" + "="*50)
    print(f"Training complete! Best R@10: {best_recall:.4f}")
    print(f"Model saved to {MODEL_DIR / 'best_model_v2.pt'}")


if __name__ == "__main__":
    main()
