#!/usr/bin/env python3

import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

DATASET_DIR = Path.home() / "shop-the-look-dataset"
MODEL_DIR = Path.home() / "shop-the-look-model"
IMAGES_DIR = MODEL_DIR / "images"
EMBEDDINGS_DIR = MODEL_DIR / "embeddings"

def load_clip_model():
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

def extract_image_embedding(model, processor, image_path: Path) -> np.ndarray:
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        # Normalize embedding
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()
    except:
        return None

def extract_embeddings_batch(model, processor, image_dir: Path, output_file: Path, batch_size: int = 32):
    image_files = list(image_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} images in {image_dir}")
    
    embeddings = {}
    failed = 0
    
    for i in tqdm(range(0, len(image_files), batch_size), desc=f"Processing {image_dir.name}"):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        batch_ids = []
        
        for img_path in batch_files:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                batch_ids.append(img_path.stem)
            except:
                failed += 1
                continue
        
        if not batch_images:
            continue
            
        try:
            inputs = processor(images=batch_images, return_tensors="pt", padding=True)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            
            for idx, img_id in enumerate(batch_ids):
                embeddings[img_id] = features[idx].numpy()
        except Exception as e:
            print(f"Batch error: {e}")
            failed += len(batch_images)
    
    print(f"Extracted {len(embeddings)} embeddings, {failed} failed")
    
    # Save as npz
    np.savez_compressed(output_file, **embeddings)
    print(f"Saved to {output_file}")
    return embeddings

def main():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    model, processor = load_clip_model()
    
    # Extract scene embeddings
    scenes_dir = IMAGES_DIR / "scenes"
    if scenes_dir.exists():
        extract_embeddings_batch(
            model, processor, 
            scenes_dir, 
            EMBEDDINGS_DIR / "scene_embeddings.npz"
        )
    else:
        print(f"Scenes directory not found: {scenes_dir}")
    
    # Extract product embeddings
    products_dir = IMAGES_DIR / "products"
    if products_dir.exists():
        extract_embeddings_batch(
            model, processor,
            products_dir,
            EMBEDDINGS_DIR / "product_embeddings.npz"
        )
    else:
        print(f"Products directory not found: {products_dir}")

if __name__ == "__main__":
    main()
