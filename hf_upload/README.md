---
license: mit
tags:
- fashion
- recommendation
- pytorch
- image-to-image
datasets:
- shop-the-look
language:
- en
---

# complete the look - fashion compatibility model

scene-based complementary product recommendation model. given an outfit photo, find products that match the style.

## model description

trained on the [shop the look dataset](https://github.com/kang205/STL-Dataset) with fashion SigLIP embeddings.

### architecture
- **input**: fashion SigLIP embeddings (768-dim)
- **scene encoder**: MLP with residual blocks → 256-dim
- **product encoder**: MLP with category embeddings → 256-dim
- **training**: contrastive + triplet loss with hard negative mining

### performance

| metric | score |
|--------|-------|
| R@1    | 12.3% |
| R@5    | 28.6% |
| R@10   | 36.6% |
| MRR    | 0.204 |

## files

- `best_model_v2.pt` - model weights (37MB)
- `product_embeddings.pkl` - pre-computed product embeddings (110MB, 36K products)
- `scene_embeddings.pkl` - pre-computed scene embeddings (87MB, 29K scenes)

## usage

```python
import torch
import pickle

checkpoint = torch.load("best_model_v2.pt", map_location="cpu", weights_only=False)
# see github repo for full code

with open("product_embeddings.pkl", "rb") as f:
    product_embeddings = pickle.load(f)
```

## code

full training code and web demo: [github repo](https://github.com/USER/complete-the-look)

## citation

based on:
```bibtex
@inproceedings{kang2019complete,
  title={Complete the Look: Scene-based Complementary Product Recommendation},
  author={Kang, Wang-Cheng and Kim, Eric and Leskovec, Jure and Rosenberg, Charles and McAuley, Julian},
  booktitle={CVPR},
  year={2019}
}
```
