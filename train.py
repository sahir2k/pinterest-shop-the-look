#!/usr/bin/env python3

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

DATASET_DIR = Path.home() / "shop-the-look-dataset"
MODEL_DIR = Path.home() / "shop-the-look-model"
EMBEDDINGS_DIR = MODEL_DIR / "embeddings"

class CTLDataset(Dataset):
    def __init__(self, pairs, scene_embeddings, product_embeddings, categories, mode="train"):
        self.pairs = pairs
        self.scene_embeddings = scene_embeddings
        self.product_embeddings = product_embeddings
        self.categories = categories
        self.mode = mode
        
        self.valid_pairs = [
            p for p in pairs 
            if p["scene"] in scene_embeddings and p["product"] in product_embeddings
        ]
        print(f"{mode}: {len(self.valid_pairs)}/{len(pairs)} valid pairs")

        self.scene_to_products = defaultdict(set)
        for p in self.valid_pairs:
            self.scene_to_products[p["scene"]].add(p["product"])

        self.all_products = list(product_embeddings.keys())
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        scene_id = pair["scene"]
        product_id = pair["product"]
        
        scene_emb = torch.tensor(self.scene_embeddings[scene_id], dtype=torch.float32)
        product_emb = torch.tensor(self.product_embeddings[product_id], dtype=torch.float32)

        cat = self.categories.get(product_id, "Unknown")
        cat_parts = cat.split("|")
        top_cat = cat_parts[-1] if len(cat_parts) > 1 else cat_parts[0]
        
        return {
            "scene_emb": scene_emb,
            "product_emb": product_emb,
            "scene_id": scene_id,
            "product_id": product_id,
            "category": top_cat
        }


class CTLModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()

        self.scene_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

        self.product_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def encode_scene(self, x):
        emb = self.scene_encoder(x)
        return F.normalize(emb, dim=-1)
    
    def encode_product(self, x):
        emb = self.product_encoder(x)
        return F.normalize(emb, dim=-1)
    
    def forward(self, scene_emb, product_emb):
        scene_features = self.encode_scene(scene_emb)
        product_features = self.encode_product(product_emb)
        return scene_features, product_features
    
    def compatibility_score(self, scene_emb, product_emb):
        scene_features = self.encode_scene(scene_emb)
        product_features = self.encode_product(product_emb)
        return (scene_features * product_features).sum(dim=-1)


def contrastive_loss(scene_features, product_features, temperature):
    logits = torch.matmul(scene_features, product_features.T) / temperature
    batch_size = scene_features.shape[0]
    labels = torch.arange(batch_size, device=scene_features.device)

    loss_s2p = F.cross_entropy(logits, labels)
    loss_p2s = F.cross_entropy(logits.T, labels)
    
    return (loss_s2p + loss_p2s) / 2


def load_data():
    print("Loading embeddings...")

    scene_pkl = EMBEDDINGS_DIR / "scene_embeddings.pkl"
    product_pkl = EMBEDDINGS_DIR / "product_embeddings.pkl"
    
    if scene_pkl.exists():
        import pickle
        with open(scene_pkl, "rb") as f:
            scene_embeddings = pickle.load(f)
        with open(product_pkl, "rb") as f:
            product_embeddings = pickle.load(f)
    else:
        scene_embeddings = dict(np.load(EMBEDDINGS_DIR / "scene_embeddings.npz"))
        product_embeddings = dict(np.load(EMBEDDINGS_DIR / "product_embeddings.npz"))
    
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


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        scene_emb = batch["scene_emb"].to(device)
        product_emb = batch["product_emb"].to(device)
        
        optimizer.zero_grad()
        
        scene_features, product_features = model(scene_emb, product_emb)
        loss = contrastive_loss(scene_features, product_features, model.temperature)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    
    all_scene_features = []
    all_product_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            scene_emb = batch["scene_emb"].to(device)
            product_emb = batch["product_emb"].to(device)
            
            scene_features, product_features = model(scene_emb, product_emb)
            
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
    BATCH_SIZE = 128
    EPOCHS = 20
    LR = 1e-3
    DEVICE = "cpu"

    pairs, scene_embeddings, product_embeddings, categories = load_data()
    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    train_dataset = CTLDataset(train_pairs, scene_embeddings, product_embeddings, categories, "train")
    val_dataset = CTLDataset(val_pairs, scene_embeddings, product_embeddings, categories, "val")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    sample_emb = next(iter(scene_embeddings.values()))
    input_dim = len(sample_emb)
    print(f"Embedding dimension: {input_dim}")
    
    model = CTLModel(input_dim=input_dim, hidden_dim=384, output_dim=128).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_recall = 0
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE)
        print(f"Train loss: {train_loss:.4f}")
        
        metrics = evaluate(model, val_loader, DEVICE)
        print(f"Val R@1: {metrics['recall@1']:.4f}, R@5: {metrics['recall@5']:.4f}, "
              f"R@10: {metrics['recall@10']:.4f}, MRR: {metrics['mrr']:.4f}")
        
        scheduler.step()
        
        # Save best model
        if metrics['recall@10'] > best_recall:
            best_recall = metrics['recall@10']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
            }, MODEL_DIR / "best_model.pt")
            print(f"Saved best model (R@10: {best_recall:.4f})")
    
    print(f"\nTraining complete! Best R@10: {best_recall:.4f}")
    print(f"Model saved to {MODEL_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()
