# complete the look - fashion recommender

scene-based complementary product recommendation model trained on the [shop the look dataset](https://github.com/kang205/STL-Dataset).

## what it does

given an outfit/scene photo, find products (shoes, tops, pants, etc.) that match the style.

## model architecture

- **embeddings**: fashion SigLIP (768-dim)
- **scene encoder**: MLP with residual blocks (768 → 512 → 256)
- **product encoder**: MLP with category embeddings (768+64 → 512 → 256)
- **training**: contrastive + triplet loss with hard negative mining

## results

| metric | v1 (baseline) | v2 (improved) |
|--------|---------------|---------------|
| R@1    | 6.1%          | 12.3%         |
| R@5    | 17.9%         | 28.6%         |
| R@10   | 25.4%         | **36.6%**     |
| MRR    | 0.126         | 0.204         |

## files

### code (this repo)
- `train.py` - baseline training script
- `train_v2.py` - improved training with hard negatives & category embeddings
- `app.py` - flask web demo
- `extract_embeddings_gpu.py` - extract fashion SigLIP embeddings via API
- `download_images.py` - download images from pinterest

### model & embeddings (hugging face)
see: [huggingface.co/[USER]/complete-the-look](https://huggingface.co/)

- `best_model_v2.pt` - trained model weights (37MB)
- `product_embeddings.pkl` - product embeddings (110MB, 36K products)
- `scene_embeddings.pkl` - scene embeddings (87MB, 29K scenes)

## quick start

```bash
git clone https://github.com/[USER]/complete-the-look
cd complete-the-look

# download model & embeddings from huggingface

pip install torch flask numpy requests

python app.py
# open http://localhost:8000
```

## training

```bash
# 1. download dataset from https://github.com/kang205/STL-Dataset
# 2. download images
python download_images.py

# 3. extract embeddings (requires fashion SigLIP API or local model)
python extract_embeddings_gpu.py

# 4. train
python train_v2.py
```

## dataset

- **scenes**: 29K fashion scene images
- **products**: 37K product images  
- **pairs**: 72K scene-product compatibility pairs
- **categories**: shoes, pants, shirts, dresses, outerwear, etc.

## paper

based on "complete the look: scene-based complementary product recommendation" (CVPR 2019)

```bibtex
@inproceedings{kang2019complete,
  title={Complete the Look: Scene-based Complementary Product Recommendation},
  author={Kang, Wang-Cheng and Kim, Eric and Leskovec, Jure and Rosenberg, Charles and McAuley, Julian},
  booktitle={CVPR},
  year={2019}
}
```

## license

MIT
