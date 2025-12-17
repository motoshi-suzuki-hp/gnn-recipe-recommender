# scripts/eval_metrics.py  （dev/testどちらでも評価可能）
import json, numpy as np, torch, random, os
from collections import defaultdict

seed = int(os.getenv("SEED", "42"))
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

ART = "artifacts"
meta = json.load(open(f"{ART}/meta.json"))

split = os.getenv("SPLIT", "test")  # "dev" or "test"
train = torch.from_numpy(np.load(f"{ART}/train_edge_ur.npy")).long()
edges = torch.from_numpy(np.load(f"{ART}/{split}_edge_ur.npy")).long()
u_repr = torch.load(f"{ART}/user_repr.pt")
r_repr = torch.load(f"{ART}/recipe_repr.pt")

U, I = meta["num_users"], meta["num_recipes"]

seen = defaultdict(set)
for u, i in train.tolist():
    seen[int(u)].add(int(i))

def cosine_scores(mat_a, vec_b, eps=1e-8):
    a = mat_a / (mat_a.norm(dim=1, keepdim=True) + eps)
    b = vec_b / (vec_b.norm(dim=0, keepdim=False) + eps)
    return (a @ b)

def recall_ndcg_at_k(K=10, neg_samples=1000):
    recalls, ndcgs, n = [], [], 0
    for (u, i_pos) in edges.tolist():
        u = int(u); i_pos = int(i_pos)
        if u >= len(u_repr):
            continue
        n += 1
        cand = [i_pos]
        while len(cand) < neg_samples + 1:
            j = np.random.randint(0, I)
            if j == i_pos or j in seen[u]:
                continue
            cand.append(j)
        scores = cosine_scores(r_repr[cand], u_repr[u]).numpy()
        rank = np.argsort(-scores)
        topk = rank[:K]
        hit = 1 if 0 in topk else 0
        pos_idx = int(np.where(rank == 0)[0][0])
        dcg = 1.0/np.log2(2 + pos_idx) if pos_idx < K else 0.0
        recalls.append(hit); ndcgs.append(dcg)
    return float(np.mean(recalls)), float(np.mean(ndcgs)), n

neg_samples = int(os.getenv("NEG", "1000"))
for K in [5, 10, 20, 50]:
    r, n, users = recall_ndcg_at_k(K=K, neg_samples=neg_samples)
    print(f"[{split}] K={K:>2} | Recall@K={r:.4f} | NDCG@K={n:.4f} | users={users}")
