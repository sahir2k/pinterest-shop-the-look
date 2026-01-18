#!/usr/bin/env python3

import os
import json
import asyncio
import aiohttp
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import pickle

DATASET_DIR = Path.home() / "shop-the-look-dataset"
MODEL_DIR = Path.home() / "shop-the-look-model"
EMBEDDINGS_DIR = MODEL_DIR / "embeddings"

API_BASE = "https://fashion-siglip-embedder-326559897777.us-central1.run.app"

def hash_to_url(signature: str) -> str:
    return f"http://i.pinimg.com/400x/{signature[0:2]}/{signature[2:4]}/{signature[4:6]}/{signature}.jpg"

def load_dataset():
    scenes = set()
    products = set()
    
    with open(DATASET_DIR / "fashion.json") as f:
        for line in f:
            data = json.loads(line)
            scenes.add(data["scene"])
            products.add(data["product"])
    
    return list(scenes), list(products)

async def get_embedding(session: aiohttp.ClientSession, image_id: str, semaphore: asyncio.Semaphore):
    url = hash_to_url(image_id)
    
    async with semaphore:
        try:
            async with session.post(
                f"{API_BASE}/embed",
                json={"image_url": url, "pad_to_square": False, "use_white_background": False},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return image_id, np.array(data["embedding"], dtype=np.float32)
                else:
                    return image_id, None
        except Exception as e:
            return image_id, None

async def extract_batch(image_ids: list, desc: str, max_concurrent: int = 500):
    semaphore = asyncio.Semaphore(max_concurrent)
    embeddings = {}
    failed = []
    
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [get_embedding(session, img_id, semaphore) for img_id in image_ids]
        
        log_file = open('/home/exedev/shop-the-look-model/progress.log', 'w')
        pbar = tqdm(total=len(tasks), desc=desc, file=log_file, mininterval=1)
        for coro in asyncio.as_completed(tasks):
            img_id, emb = await coro
            if emb is not None:
                embeddings[img_id] = emb
            else:
                failed.append(img_id)
            pbar.update(1)
        pbar.close()
        log_file.close()
    
    return embeddings, failed

async def main():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset...")
    scenes, products = load_dataset()
    print(f"Found {len(scenes)} unique scenes, {len(products)} unique products")
    
    # Check API health
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{API_BASE}/health") as resp:
            if resp.status == 200:
                print("API is healthy!")
            else:
                print(f"API health check failed: {resp.status}")
                return
    
    # Extract scene embeddings
    print("\n=== Extracting Scene Embeddings ===")
    start = time.time()
    scene_embeddings, scene_failed = await extract_batch(scenes, "Scenes", max_concurrent=150)
    print(f"Extracted {len(scene_embeddings)} scene embeddings in {time.time()-start:.1f}s")
    print(f"Failed: {len(scene_failed)}")
    
    # Save scene embeddings
    scene_file = EMBEDDINGS_DIR / "scene_embeddings.pkl"
    with open(scene_file, "wb") as f:
        pickle.dump(scene_embeddings, f)
    print(f"Saved to {scene_file}")
    
    # Extract product embeddings
    print("\n=== Extracting Product Embeddings ===")
    start = time.time()
    product_embeddings, product_failed = await extract_batch(products, "Products", max_concurrent=150)
    print(f"Extracted {len(product_embeddings)} product embeddings in {time.time()-start:.1f}s")
    print(f"Failed: {len(product_failed)}")
    
    # Save product embeddings
    product_file = EMBEDDINGS_DIR / "product_embeddings.pkl"
    with open(product_file, "wb") as f:
        pickle.dump(product_embeddings, f)
    print(f"Saved to {product_file}")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Scene embeddings: {len(scene_embeddings)}")
    print(f"Product embeddings: {len(product_embeddings)}")
    
    # Check embedding dimension
    if scene_embeddings:
        sample = next(iter(scene_embeddings.values()))
        print(f"Embedding dimension: {sample.shape}")

if __name__ == "__main__":
    asyncio.run(main())
