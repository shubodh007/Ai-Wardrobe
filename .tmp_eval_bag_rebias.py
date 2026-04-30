import pickle
from pathlib import Path
import numpy as np
import torch
import cv2
from PIL import Image
import io

import api.main as api
from src.train_classifier import FashionMNISTCNN


def fg_stats(rgb_28):
    u8 = np.clip(rgb_28 * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    border = np.concatenate([gray[0,:], gray[-1,:], gray[:,0], gray[:,-1]])
    bg = float(np.median(border))
    diff = np.abs(gray.astype(np.float32) - bg)
    thr = max(8.0, float(np.percentile(diff, 75) * 0.55))
    mask = (diff > thr).astype(np.uint8)
    k = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if mask.sum() < 8:
        return None

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        m = mask.astype(bool)
    else:
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = labels == idx

    ratio = float(np.mean(m))
    if ratio < 0.01 or ratio > 0.85:
        return None

    ys, xs = np.where(m)
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    asp = float(h / max(w, 1))
    return {"fg_ratio": ratio, "aspect_ratio": asp}


def rebias_bag(probs, rgb_28):
    vec = probs.astype(np.float32).copy()
    bag_idx = api.CATEGORIES.index("Bag")
    top = int(np.argmax(vec))
    bag_prob = float(vec[bag_idx])
    if top != bag_idx or bag_prob >= 0.62:
        return vec

    s = fg_stats(rgb_28)
    if s is None:
        return vec

    if s["aspect_ratio"] < 1.18:
        return vec

    targets = [api.CATEGORIES.index(n) for n in ["Shirt", "T-shirt", "Pullover", "Coat", "Dress"]]
    base = vec[targets]
    weights = base + 1e-3
    weights = weights / np.sum(weights)

    transfer = min(0.24, 0.09 + (s["aspect_ratio"] - 1.18) * 0.16 + (0.62 - bag_prob) * 0.25)
    transfer = max(0.0, min(transfer, bag_prob - 0.02))
    vec[bag_idx] -= transfer
    vec[targets] += transfer * weights
    vec = vec / max(float(np.sum(vec)), 1e-8)
    return vec


root=Path("d:/ml pbl/ai-wardrobe-ml")
with (root/"data/processed/fashion_mnist_processed.pkl").open("rb") as f:
    d=pickle.load(f)

x=np.asarray(d["x_test"], dtype=np.float32)
y=np.asarray(d["y_test"], dtype=np.int64)
if x.ndim == 4 and x.shape[1] in (1,3) and x.shape[-1] not in (1,3):
    x=np.transpose(x,(0,2,3,1))
if np.max(x)>1.0:
    x=x/255.0

ckpt=torch.load(root/"models/saved_models/best_model.pt", map_location="cpu")
state=ckpt["model_state_dict"] if isinstance(ckpt,dict) and "model_state_dict" in ckpt else ckpt
inp=tuple(int(v) for v in ckpt.get("input_shape",(28,28,3))) if isinstance(ckpt,dict) else (28,28,3)
num_classes=int(ckpt.get("num_classes",10)) if isinstance(ckpt,dict) else 10
model=FashionMNISTCNN(input_shape=inp, num_classes=num_classes)
model.load_state_dict(state); model.eval()
api.model_holder["model"] = model
api.model_holder["feature_extractor"] = None
api.model_holder["color_extractor"] = None
api.model_holder["class_prototypes"] = None

rng=np.random.default_rng(42)
n=2000
for mode in ["std", "padded"]:
    corr0=corr1=0
    c0=np.zeros(10,dtype=np.int64)
    c1=np.zeros(10,dtype=np.int64)

    for i in range(n):
        img=np.asarray(x[i], dtype=np.float32)
        if img.ndim==2:
            img=np.stack([img,img,img],axis=-1)
        elif img.ndim==3 and img.shape[-1]==1:
            img=np.repeat(img,3,axis=-1)
        u8=np.clip(img*255.0,0,255).astype(np.uint8)

        rgb=u8
        if mode=="padded":
            canvas=np.full((256,256,3),228,dtype=np.uint8)
            scale=float(rng.uniform(3.0,5.8))
            nh=max(56,min(180,int(round(u8.shape[0]*scale))))
            nw=max(56,min(180,int(round(u8.shape[1]*scale))))
            rs=cv2.resize(u8,(nw,nh),interpolation=cv2.INTER_CUBIC)
            y0=(256-nh)//2 + int(rng.integers(-10,11))
            x0=(256-nw)//2 + int(rng.integers(-10,11))
            y0=max(0,min(256-nh,y0)); x0=max(0,min(256-nw,x0))
            canvas[y0:y0+nh, x0:x0+nw]=rs
            rgb=canvas

        buf=io.BytesIO(); Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
        rgb_28, rgb_img, gm = api._decode_image_arrays(buf.getvalue())
        res = api._classify_image(rgb_28, rgb_img, gm)
        probs = np.array([res["all_scores"][c] for c in api.CATEGORIES], dtype=np.float32)

        p0=int(np.argmax(probs))
        c0[p0]+=1
        corr0 += int(p0==int(y[i]))

        p_adj = rebias_bag(probs, rgb_28)
        p1=int(np.argmax(p_adj))
        c1[p1]+=1
        corr1 += int(p1==int(y[i]))

    print(mode, "orig", round(corr0/n,4), c0.tolist())
    print(mode, "adj ", round(corr1/n,4), c1.tolist())
