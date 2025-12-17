# scripts/infer_topk.py
import json, numpy as np, torch
from collections import defaultdict
import argparse

ART = "artifacts"
meta = json.load(open(f"{ART}/meta.json"))
train = torch.from_numpy(np.load(f"{ART}/train_edge_ur.npy")).long()
u_repr = torch.load(f"{ART}/user_repr.pt")
r_repr = torch.load(f"{ART}/recipe_repr.pt")

seen = defaultdict(set)
for u,i in train.tolist():
    seen[int(u)].add(int(i))

def cosine_scores(mat_a, vec_b, eps=1e-8):
    a = mat_a / (mat_a.norm(dim=1, keepdim=True) + eps)
    b = vec_b / (vec_b.norm(dim=0, keepdim=False) + eps)
    return (a @ b)

def recommend_for_user(u, K=20):
    # 既視アイテムを除外して全件スコア → 上位K
    mask = torch.ones(len(r_repr), dtype=torch.bool)
    for j in seen[u]:
        mask[j] = False
    cand_idx = torch.arange(len(r_repr))[mask]
    scores = cosine_scores(r_repr[cand_idx], u_repr[u]).numpy()
    order = np.argsort(-scores)[:K]
    top_items = cand_idx[order].tolist()
    top_scores = scores[order].tolist()
    return list(zip(top_items, top_scores))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--limit_users", type=int, default=2000, help="出力ユーザー数の上限（大きいと時間がかかります）")
    ap.add_argument("--out", type=str, default=f"{ART}/topk.csv")
    args = ap.parse_args()

    import csv
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "rank", "recipe_id", "score"])
        U = len(u_repr)
        for u in range(min(U, args.limit_users)):
            recs = recommend_for_user(u, K=args.k)
            for r, (item, s) in enumerate(recs, 1):
                w.writerow([u, r, int(item), float(s)])
    print(f"wrote: {args.out}")
