#!/usr/bin/env python3

import os
import json
import asyncio
import aiohttp
from pathlib import Path
from tqdm import tqdm
import hashlib
import time

DATASET_DIR = Path.home() / "shop-the-look-dataset"
IMAGES_DIR = Path.home() / "shop-the-look-model" / "images"
SCENES_DIR = IMAGES_DIR / "scenes"
PRODUCTS_DIR = IMAGES_DIR / "products"

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
    
    return scenes, products

async def download_image(session: aiohttp.ClientSession, signature: str, output_path: Path, semaphore: asyncio.Semaphore):
    if output_path.exists():
        return True
    
    url = hash_to_url(signature)
    
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    output_path.write_bytes(content)
                    return True
                else:
                    return False
        except Exception as e:
            return False

async def download_batch(signatures: list, output_dir: Path, desc: str, max_concurrent: int = 20):
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for sig in signatures:
            output_path = output_dir / f"{sig}.jpg"
            tasks.append(download_image(session, sig, output_path, semaphore))
        
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
            result = await coro
            results.append(result)
        
        return sum(results)

async def main():
    print("Loading dataset...")
    scenes, products = load_dataset()
    print(f"Found {len(scenes)} unique scenes, {len(products)} unique products")
    
    # Download scenes
    print("\nDownloading scene images...")
    scenes_success = await download_batch(list(scenes), SCENES_DIR, "Scenes")
    print(f"Successfully downloaded {scenes_success}/{len(scenes)} scenes")
    
    # Download products  
    print("\nDownloading product images...")
    products_success = await download_batch(list(products), PRODUCTS_DIR, "Products")
    print(f"Successfully downloaded {products_success}/{len(products)} products")
    
    print(f"\nImages saved to {IMAGES_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
