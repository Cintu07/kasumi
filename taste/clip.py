"""The taste model: CLIP image embeddings and a logistic regression on your clicks.

    python taste/clip.py train    # learn from explore/clicks.jsonl
    python taste/clip.py serve    # used by `kasumi evolve --taste clip`

`serve` reads one png path per line and answers one score per line. A blank
line ends a batch. With --embed, each score is followed by the image embedding.
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPLORE = os.path.join(ROOT, "explore")
MODEL = os.path.join(EXPLORE, "taste_embed.pkl")
CACHE = os.path.join(EXPLORE, "emb_clip.pkl")


class Embedder:
    def __init__(self):
        import open_clip
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.model = model.to(self.device).eval()

    def __call__(self, paths, batch=64):
        from PIL import Image

        torch = self.torch
        out = []
        for k in range(0, len(paths), batch):
            images = [self.preprocess(Image.open(p).convert("RGB")) for p in paths[k : k + batch]]
            with torch.no_grad(), torch.autocast(self.device, enabled=self.device == "cuda"):
                e = self.model.encode_image(torch.stack(images).to(self.device)).float()
            out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out)


def read_clicks(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in filter(str.strip, f):
            click = json.loads(line)
            thumb = os.path.join(ROOT, click["thumb"])
            if os.path.exists(thumb):
                rows.append((thumb, bool(click["liked"])))
    return rows


def embed_cached(paths):
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            cache = pickle.load(f)
    todo = [p for p in paths if p not in cache]
    if todo:
        cache.update(zip(todo, Embedder()(todo)))
        with open(CACHE, "wb") as f:
            pickle.dump(cache, f)
    return np.stack([cache[p] for p in paths])


def train(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    rows = read_clicks(args.clicks)
    y = np.array([liked for _, liked in rows], dtype=int)
    if y.sum() < 4 or len(y) - y.sum() < 4:
        sys.exit(f"need at least 4 likes and 4 skips, have {y.sum()} and {len(y) - y.sum()}")
    x = embed_cached([path for path, _ in rows])
    fit = lambda a, b: LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(a, b)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
    auc = roc_auc_score(yte, fit(xtr, ytr).predict_proba(xte)[:, 1])
    with open(MODEL, "wb") as f:
        pickle.dump({"model": "clip", "clf": fit(x, y), "auc": auc, "n": len(y)}, f)
    print(f"held-out auc {auc * 100:.1f}% on {len(y)} judgements, saved {MODEL}")


def serve(args):
    clf = None
    if os.path.exists(MODEL):
        with open(MODEL, "rb") as f:
            clf = pickle.load(f)["clf"]
    embed = Embedder()
    print("ready", flush=True)
    batch = []
    for line in sys.stdin:
        if line.strip():
            batch.append(line.strip())
        elif batch:
            e = embed(batch)
            scores = clf.predict_proba(e)[:, 1] if clf is not None else np.full(len(batch), 0.5)
            for score, vector in zip(scores, e):
                extra = " " + " ".join(f"{v:.5f}" for v in vector) if args.embed else ""
                sys.stdout.write(f"{score:.6f}{extra}\n")
            sys.stdout.flush()
            batch = []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clicks", default=os.path.join(EXPLORE, "clicks.jsonl"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train").set_defaults(run=train)
    server = commands.add_parser("serve")
    server.add_argument("--embed", action="store_true")
    server.set_defaults(run=serve)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
