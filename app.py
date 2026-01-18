#!/usr/bin/env python3

import io
import json
import pickle
import base64
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import requests
from flask import Flask, request, jsonify, render_template_string

MODEL_DIR = Path.home() / "shop-the-look-model"
DATASET_DIR = Path.home() / "shop-the-look-dataset"
EMBEDDINGS_DIR = MODEL_DIR / "embeddings"

API_BASE = "https://fashion-siglip-embedder-326559897777.us-central1.run.app"

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
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
    
    def encode_scene(self, x):
        emb = self.scene_encoder(x)
        return F.normalize(emb, dim=-1)
    
    def encode_product(self, x, cat_idx):
        cat_emb = self.category_embedding(cat_idx)
        x = torch.cat([x, cat_emb], dim=-1)
        emb = self.product_encoder(x)
        return F.normalize(emb, dim=-1)


class Recommender:
    def __init__(self):
        print("Loading model...")
        checkpoint = torch.load(MODEL_DIR / "best_model_v2.pt", map_location="cpu", weights_only=False)
        
        num_categories = checkpoint.get('num_categories', 10)
        self.cat_to_idx = checkpoint.get('cat_to_idx', {})
        
        self.model = CTLModelV2(
            input_dim=768, hidden_dim=512, output_dim=256, 
            num_categories=num_categories
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("Loading product embeddings...")
        with open(EMBEDDINGS_DIR / "product_embeddings.pkl", "rb") as f:
            self.product_embeddings = pickle.load(f)
        self.product_ids = list(self.product_embeddings.keys())
        
        print("Loading categories...")
        with open(DATASET_DIR / "fashion-cat.json") as f:
            self.categories = json.load(f)
        
        # Pre-compute product features for each category
        print("Pre-computing product features...")
        self.product_features = {}
        self.product_cat_indices = {}
        
        for pid in self.product_ids:
            cat = self.categories.get(pid, "Unknown")
            leaf_cat = cat.split("|")[-1]
            cat_idx = self.cat_to_idx.get(leaf_cat, 0)
            self.product_cat_indices[pid] = cat_idx
        
        # Batch encode all products
        product_embs = torch.tensor(
            np.stack([self.product_embeddings[pid] for pid in self.product_ids]),
            dtype=torch.float32
        )
        cat_indices = torch.tensor(
            [self.product_cat_indices[pid] for pid in self.product_ids],
            dtype=torch.long
        )
        
        with torch.no_grad():
            self.all_product_features = self.model.encode_product(product_embs, cat_indices)
        
        print(f"Loaded {len(self.product_ids)} products")
    
    def get_scene_embedding(self, image_bytes):
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        resp = requests.post(f"{API_BASE}/embed_upload", files=files, timeout=30)
        if resp.status_code == 200:
            return np.array(resp.json()['embedding'], dtype=np.float32)
        else:
            raise Exception(f"Embedding API error: {resp.status_code}")
    
    def recommend(self, image_bytes, top_k=20, category_filter=None):
        scene_emb = self.get_scene_embedding(image_bytes)

        with torch.no_grad():
            scene_features = self.model.encode_scene(
                torch.tensor(scene_emb, dtype=torch.float32).unsqueeze(0)
            )

        similarities = torch.matmul(scene_features, self.all_product_features.T).squeeze()
        fetch_k = top_k * 20 if category_filter else top_k * 3
        scores, indices = torch.topk(similarities, k=min(fetch_k, len(self.product_ids)))

        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            pid = self.product_ids[idx]
            cat = self.categories.get(pid, "Unknown")
            leaf_cat = cat.split("|")[-1]

            if category_filter and category_filter.lower() not in cat.lower():
                continue
            
            results.append({
                "product_id": pid,
                "score": round(score, 4),
                "category": leaf_cat,
                "full_category": cat,
                "image_url": f"http://i.pinimg.com/400x/{pid[0:2]}/{pid[2:4]}/{pid[4:6]}/{pid}.jpg"
            })
            
            if len(results) >= top_k:
                break
        
        return results


app = Flask(__name__)
recommender = None

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Complete the Look - Fashion Recommender</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; text-align: center; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
        .container { display: flex; gap: 30px; flex-wrap: wrap; }
        .upload-section { 
            flex: 1; 
            min-width: 300px;
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .results-section { 
            flex: 2; 
            min-width: 400px;
        }
        .drop-zone {
            border: 3px dashed #ccc;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: #fafafa;
        }
        .drop-zone:hover, .drop-zone.dragover {
            border-color: #007bff;
            background: #f0f7ff;
        }
        .drop-zone img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 8px;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 15px;
        }
        .product-card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .product-card:hover {
            transform: translateY(-4px);
        }
        .product-card img {
            width: 100%;
            height: 180px;
            object-fit: cover;
        }
        .product-info {
            padding: 12px;
        }
        .product-category {
            font-size: 12px;
            color: #666;
            margin-bottom: 4px;
        }
        .product-score {
            font-size: 14px;
            font-weight: 600;
            color: #007bff;
        }
        .category-section {
            margin: 20px 0;
            padding: 15px;
            background: #f0f7ff;
            border-radius: 8px;
        }
        .category-section label {
            font-weight: 600;
            color: #333;
            display: block;
            margin-bottom: 10px;
        }
        .category-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .cat-btn {
            padding: 8px 16px;
            border: 2px solid #007bff;
            background: white;
            color: #007bff;
            border-radius: 20px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        .cat-btn:hover, .cat-btn.active {
            background: #007bff;
            color: white;
        }
        .cat-btn.active {
            font-weight: 600;
        }
        select, button {
            padding: 10px 16px;
            font-size: 14px;
            border-radius: 8px;
            border: 1px solid #ddd;
            cursor: pointer;
        }
        button.primary {
            background: #007bff;
            color: white;
            border: none;
            width: 100%;
            padding: 12px;
            font-size: 16px;
            margin-top: 15px;
        }
        button.primary:hover { background: #0056b3; }
        button.primary:disabled { background: #ccc; }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007bff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .stats {
            background: #e8f4fd;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-size: 14px;
        }
        .how-it-works {
            background: #fff3cd;
            padding: 12px 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-size: 13px;
            color: #856404;
        }
        
        /* Mobile responsive */
        @media (max-width: 768px) {
            body {
                padding: 10px;
            }
            h1 {
                font-size: 24px;
            }
            .subtitle {
                font-size: 14px;
                margin-bottom: 20px;
            }
            .container {
                flex-direction: column;
                gap: 20px;
            }
            .upload-section, .results-section {
                min-width: unset;
                width: 100%;
            }
            .drop-zone {
                padding: 25px;
            }
            .drop-zone img {
                max-height: 200px;
            }
            .category-buttons {
                gap: 6px;
            }
            .cat-btn {
                padding: 6px 12px;
                font-size: 12px;
            }
            .products-grid {
                grid-template-columns: repeat(2, 1fr);
                gap: 10px;
            }
            .product-card img {
                height: 140px;
            }
            .product-info {
                padding: 8px;
            }
            .product-category {
                font-size: 11px;
            }
            .product-score {
                font-size: 12px;
            }
            .how-it-works {
                font-size: 12px;
                padding: 10px 12px;
            }
            .stats {
                font-size: 13px;
                padding: 8px 12px;
            }
        }
        
        @media (max-width: 400px) {
            .products-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            .cat-btn {
                padding: 5px 10px;
                font-size: 11px;
            }
        }
    </style>
</head>
<body>
    <h1>👗 Complete the Look</h1>
    <p class="subtitle">Upload your outfit photo, then choose what you want to shop for</p>
    
    <div class="container">
        <div class="upload-section">
            <div class="how-it-works">
                <strong>💡 How it works:</strong><br>
                1. Upload a photo of your outfit/style<br>
                2. Select what you want to find (shoes, bag, etc.)<br>
                3. Get products that match your style!
            </div>
            
            <h3>1. Upload Your Look</h3>
            <div class="drop-zone" id="dropZone">
                <p>📷 Drop image here or click to upload</p>
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
            </div>
            
            <div class="category-section">
                <label>2. What do you want to find?</label>
                <div class="category-buttons">
                    <button class="cat-btn active" data-cat="">All</button>
                    <button class="cat-btn" data-cat="Shoes">👟 Shoes</button>
                    <button class="cat-btn" data-cat="Pants">👖 Pants</button>
                    <button class="cat-btn" data-cat="Shirts">👕 Tops</button>
                    <button class="cat-btn" data-cat="Dresses">👗 Dresses</button>
                    <button class="cat-btn" data-cat="Outerwear">🧥 Jackets</button>
                    <button class="cat-btn" data-cat="Shorts">🩳 Shorts</button>
                    <button class="cat-btn" data-cat="Skirts">Skirts</button>
                </div>
            </div>
            
            <button id="recommendBtn" class="primary" disabled>🔍 Find Matching Products</button>
        </div>
        
        <div class="results-section">
            <h3>Matching Products</h3>
            <div id="results">
                <p style="color: #999; text-align: center; padding: 40px;">Upload an image and select a category to find matching products</p>
            </div>
        </div>
    </div>
    
    <script>
        const dropZone = document.getElementById("dropZone");
        const fileInput = document.getElementById("fileInput");
        const recommendBtn = document.getElementById("recommendBtn");
        const results = document.getElementById("results");
        const catButtons = document.querySelectorAll(".cat-btn");
        
        let currentFile = null;
        let selectedCategory = "";
        
        // Category button handling
        catButtons.forEach(btn => {
            btn.onclick = () => {
                catButtons.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                selectedCategory = btn.dataset.cat;
            };
        });
        
        dropZone.onclick = () => fileInput.click();
        
        dropZone.ondragover = (e) => {
            e.preventDefault();
            dropZone.classList.add("dragover");
        };
        
        dropZone.ondragleave = () => dropZone.classList.remove("dragover");
        
        dropZone.ondrop = (e) => {
            e.preventDefault();
            dropZone.classList.remove("dragover");
            handleFile(e.dataTransfer.files[0]);
        };
        
        fileInput.onchange = () => handleFile(fileInput.files[0]);
        
        function handleFile(file) {
            if (!file || !file.type.startsWith("image/")) return;
            currentFile = file;
            
            const reader = new FileReader();
            reader.onload = (e) => {
                dropZone.innerHTML = `<img src="${e.target.result}">`;
                recommendBtn.disabled = false;
            };
            reader.readAsDataURL(file);
        }
        
        recommendBtn.onclick = async () => {
            if (!currentFile) return;
            
            recommendBtn.disabled = true;
            recommendBtn.textContent = "Processing...";
            results.innerHTML = `<div class="loading"><div class="spinner"></div>Getting recommendations...</div>`;
            
            const formData = new FormData();
            formData.append("image", currentFile);
            formData.append("category", selectedCategory);
            
            try {
                const resp = await fetch("/recommend", { method: "POST", body: formData });
                const data = await resp.json();
                
                if (data.error) {
                    results.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
                } else {
                    displayResults(data);
                }
            } catch (err) {
                results.innerHTML = `<p style="color: red;">Error: ${err.message}</p>`;
            }
            
            recommendBtn.disabled = false;
            recommendBtn.textContent = "Get Recommendations";
        };
        
        function displayResults(data) {
            let html = '';
            const catLabel = data.category_filter === 'All' ? 'all categories' : data.category_filter;
            html += `<div class="stats">Found ${data.results.length} matching products in <strong>${catLabel}</strong> (${data.time_ms}ms)</div>`;
            html += `<div class="products-grid">`;
            
            for (const item of data.results) {
                html += `
                    <div class="product-card">
                        <img src="${item.image_url}" onerror="this.src=\'https://via.placeholder.com/180?text=No+Image\'">
                        <div class="product-info">
                            <div class="product-category">${item.category}</div>
                            <div class="product-score">Score: ${item.score}</div>
                        </div>
                    </div>
                `;
            }
            
            html += `</div>`;
            results.innerHTML = html;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/recommend', methods=['POST'])
def recommend():
    global recommender
    
    import time
    start = time.time()
    
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"})
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        category = request.form.get('category', '')
        
        results = recommender.recommend(
            image_bytes, 
            top_k=20,
            category_filter=category if category else None
        )
        
        elapsed = int((time.time() - start) * 1000)
        
        return jsonify({
            "results": results,
            "category_filter": category if category else "All",
            "time_ms": elapsed
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/health')
def health():
    return jsonify({"status": "ok"})


if __name__ == '__main__':
    print("Initializing recommender...")
    recommender = Recommender()
    print("Starting server on port 8000...")
    app.run(host='0.0.0.0', port=8000, debug=False)
