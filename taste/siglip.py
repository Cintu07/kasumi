"""kasumi taste sidecar: image-model embeddings + logistic regression.

The built-in taste model (`kasumi taste`) uses hand-built image features.
This one embeds each thumbnail with a pretrained image model (SigLIP or CLIP,
run locally on the gpu) and fits a logistic regression on the embeddings.

    python taste/siglip.py compare            # siglip vs clip on your clicks, held-out AUC
    python taste/siglip.py train --model siglip
    python taste/siglip.py serve              # used by `kasumi evolve --taste siglip`

`serve` reads one png path per line on stdin and answers one score per line.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

MODELS = {
    "siglip": ("ViT-B-16-SigLIP", "webli"),
    "clip": ("ViT-B-32", "openai"),
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPLORE = os.path.join(ROOT, "explore")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


class Embedder:
    def __init__(self, name):
        import open_clip
        import torch

        arch, tag = MODELS[name]
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=tag)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.name = name

    def __call__(self, paths, batch=64):
        from PIL import Image

        out = []
        torch = self.torch
        for k in range(0, len(paths), batch):
            ims = [self.preprocess(Image.open(p).convert("RGB")) for p in paths[k : k + batch]]
            with torch.no_grad(), torch.autocast(self.device, enabled=self.device == "cuda"):
                e = self.model.encode_image(torch.stack(ims).to(self.device))
            e = e.float()
            e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def read_clicks(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            thumb = c["thumb"] if os.path.isabs(c["thumb"]) else os.path.join(ROOT, c["thumb"])
            if os.path.exists(thumb):
                rows.append((thumb, bool(c["liked"])))
    return rows


def embed_cached(emb, paths):
    cache_path = os.path.join(EXPLORE, f"emb_{emb.name}.pkl")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    todo = [p for p in paths if p not in cache]
    if todo:
        log(f"embedding {len(todo)} thumbnails with {emb.name}")
        for p, v in zip(todo, emb(todo)):
            cache[p] = v
        os.makedirs(EXPLORE, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
    return np.stack([cache[p] for p in paths])


def heldout_auc(x, y, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(xte)[:, 1])


def load_xy(name, clicks):
    rows = read_clicks(clicks)
    y = np.array([r[1] for r in rows], dtype=int)
    if y.sum() < 4 or (len(y) - y.sum()) < 4:
        sys.exit(f"need at least 4 likes and 4 skips in {clicks} (have {y.sum()} / {len(y) - y.sum()})")
    x = embed_cached(Embedder(name), [r[0] for r in rows])
    return x, y


def cmd_compare(a):
    for name in MODELS:
        x, y = load_xy(name, a.clicks)
        aucs = [heldout_auc(x, y, seed) for seed in range(5)]
        print(f"{name:<7} held-out auc {np.mean(aucs) * 100:5.1f}%  (5 splits, {y.sum()} likes / {len(y) - y.sum()} skips)")


def cmd_train(a):
    from sklearn.linear_model import LogisticRegression

    x, y = load_xy(a.model, a.clicks)
    auc = heldout_auc(x, y)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(x, y)
    out = os.path.join(EXPLORE, "taste_embed.pkl")
    with open(out, "wb") as f:
        pickle.dump({"model": a.model, "clf": clf, "auc": auc, "n": len(y)}, f)
    verdict = "meets the 75% bar" if auc >= 0.75 else "below the 75% bar"
    print(f"{a.model}: held-out auc {auc * 100:.1f}% ({verdict}); saved {out}")


def cmd_serve(a):
    path = os.path.join(EXPLORE, "taste_embed.pkl")
    with open(path, "rb") as f:
        saved = pickle.load(f)
    emb = Embedder(saved["model"])
    clf = saved["clf"]
    print("ready", flush=True)
    batch = []
    for line in sys.stdin:
        line = line.strip()
        if line == "":
            # blank line ends a batch: score everything received so far
            if batch:
                s = clf.predict_proba(emb(batch))[:, 1]
                sys.stdout.write("".join(f"{v:.6f}\n" for v in s))
                sys.stdout.flush()
                batch = []
            continue
        batch.append(line)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clicks", default=os.path.join(EXPLORE, "clicks.jsonl"))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("compare")
    t = sub.add_parser("train")
    t.add_argument("--model", choices=list(MODELS), default="siglip")
    sub.add_parser("serve")
    a = p.parse_args()
    {"compare": cmd_compare, "train": cmd_train, "serve": cmd_serve}[a.cmd](a)


if __name__ == "__main__":
    main()
