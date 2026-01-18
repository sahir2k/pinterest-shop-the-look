#!/usr/bin/env python3

import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

MODEL_DIR = Path.home() / "shop-the-look-model"
DATASET_DIR = Path.home() / "shop-the-look-dataset"
EMBEDDINGS_DIR = MODEL_DIR / "embeddings"

from train import CTLModel


class Recommender:
    def __init__(self):
        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        
        print("Loading CTL model...")
        checkpoint = torch.load(MODEL_DIR / "best_model.pt", map_location="cpu")
        self.ctl_model = CTLModel(input_dim=512, hidden_dim=256, output_dim=128)
        self.ctl_model.load_state_dict(checkpoint["model_state_dict"])
        self.ctl_model.eval()

        print("Loading product embeddings...")
        self.product_embeddings = dict(np.load(EMBEDDINGS_DIR / "product_embeddings.npz"))
        self.product_ids = list(self.product_embeddings.keys())

        print("Pre-computing product features...")
        product_embs = np.stack([self.product_embeddings[pid] for pid in self.product_ids])
        with torch.no_grad():
            self.product_features = self.ctl_model.encode_product(
                torch.tensor(product_embs, dtype=torch.float32)
            )

        with open(DATASET_DIR / "fashion-cat.json") as f:
            self.categories = json.load(f)
        print(f"Loaded {len(self.product_ids)} products")

    def get_clip_embedding(self, image: Image.Image) -> np.ndarray:
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

    def recommend(self, image_path: str, top_k: int = 10, category: str = None):
        image = Image.open(image_path).convert("RGB")
        scene_emb = self.get_clip_embedding(image)

        with torch.no_grad():
            scene_features = self.ctl_model.encode_scene(
                torch.tensor(scene_emb, dtype=torch.float32).unsqueeze(0)
            )

        similarities = torch.matmul(scene_features, self.product_features.T).squeeze()
        scores, indices = torch.topk(similarities, k=min(top_k * 10, len(self.product_ids)))
        
        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            product_id = self.product_ids[idx]
            cat = self.categories.get(product_id, "Unknown")

            if category and category.lower() not in cat.lower():
                continue
            
            url = f"http://i.pinimg.com/400x/{product_id[0:2]}/{product_id[2:4]}/{product_id[4:6]}/{product_id}.jpg"
            results.append({
                "product_id": product_id,
                "score": score,
                "category": cat,
                "url": url
            })
            
            if len(results) >= top_k:
                break
        
        return results


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python recommend.py <image_path> [top_k] [category]")
        print("Example: python recommend.py scene.jpg 10 Shoes")
        sys.exit(1)
    
    image_path = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    category = sys.argv[3] if len(sys.argv) > 3 else None
    
    recommender = Recommender()
    results = recommender.recommend(image_path, top_k, category)
    
    print(f"\nTop {len(results)} recommendations:")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.3f}] {r['category']}")
        print(f"   {r['url']}")


if __name__ == "__main__":
    main()
