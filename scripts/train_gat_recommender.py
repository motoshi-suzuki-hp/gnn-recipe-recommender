# scripts/train_gat_recommender.py
import os, json, numpy as np, torch, random
from torch import nn
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv

# ===== Reproducibility =====
seed = int(os.getenv("SEED", "42"))
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# ===== Device =====
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Device] Using: {device} (MPS available? {torch.backends.mps.is_available()})")

# ===== Load artifacts =====
ART = "artifacts"
meta = json.load(open(f"{ART}/meta.json"))
E_RI_NP  = np.load(f"{ART}/edge_ri.npy")
E_TR     = torch.from_numpy(np.load(f"{ART}/train_edge_ur.npy")).long()
E_DV     = torch.from_numpy(np.load(f"{ART}/dev_edge_ur.npy")).long()
E_TE     = torch.from_numpy(np.load(f"{ART}/test_edge_ur.npy")).long()

# recipe -> ingredients set（Hard Negative用）
rec2ing = defaultdict(set)
for r, g in E_RI_NP:
    rec2ing[int(r)].add(int(g))

# Popularity proxy
pop = np.bincount(E_RI_NP[:, 0], minlength=meta['num_recipes']).astype(float)
pop = pop / (pop.sum() if pop.sum()>0 else 1.0)

# ===== Graph (bi-directional) =====
def add_rev(edge_index):
    if isinstance(edge_index, torch.Tensor):
        e = edge_index
    else:
        e = torch.as_tensor(edge_index, dtype=torch.long)
    rev = torch.stack([e[:,1], e[:,0]], dim=1)
    return e, rev

data = HeteroData()
data['user'].num_nodes   = meta['num_users']
data['recipe'].num_nodes = meta['num_recipes']
data['ing'].num_nodes    = meta['num_ings']

ur, ru = add_rev(E_TR)
data['user','rates','recipe'].edge_index     = ur.t().contiguous()
data['recipe','rev_rates','user'].edge_index = ru.t().contiguous()

ri = torch.from_numpy(E_RI_NP).long()
ri, ir = add_rev(ri)
data['recipe','has','ing'].edge_index     = ri.t().contiguous()
data['ing','rev_has','recipe'].edge_index = ir.t().contiguous()

data = data.to(device)

# ===== Embeddings =====
dim_id = 64
emb_user   = nn.Embedding(meta['num_users'], dim_id).to(device)
emb_recipe = nn.Embedding(meta['num_recipes'], dim_id).to(device)
emb_ing    = nn.Embedding(meta['num_ings'], dim_id).to(device)

# ===== HeteroGAT =====
def make_conv(in_channels_dict, out_channels, heads=2, dropout=0.1):
    cdim = out_channels // heads
    convs = {
        ('user','rates','recipe'):     GATConv((in_channels_dict['user'],   in_channels_dict['recipe']),
                                               cdim, heads=heads, add_self_loops=False, concat=True,
                                               dropout=dropout),
        ('recipe','rev_rates','user'): GATConv((in_channels_dict['recipe'], in_channels_dict['user']),
                                               cdim, heads=heads, add_self_loops=False, concat=True,
                                               dropout=dropout),
        ('recipe','has','ing'):        GATConv((in_channels_dict['recipe'], in_channels_dict['ing']),
                                               cdim, heads=heads, add_self_loops=False, concat=True,
                                               dropout=dropout),
        ('ing','rev_has','recipe'):    GATConv((in_channels_dict['ing'],    in_channels_dict['recipe']),
                                               cdim, heads=heads, add_self_loops=False, concat=True,
                                               dropout=dropout),
    }
    return HeteroConv(convs, aggr='mean')

class HeteroGAT(nn.Module):
    def __init__(self, hidden=128, heads=2, dropout=0.1):
        super().__init__()
        in1 = {'user': dim_id, 'recipe': dim_id, 'ing': dim_id}
        in2 = {'user': hidden, 'recipe': hidden, 'ing': hidden}
        self.conv1 = make_conv(in1, hidden, heads=heads, dropout=dropout)
        self.conv2 = make_conv(in2, hidden, heads=heads, dropout=dropout)
        # relation-wise gains
        self.w_rates     = nn.Parameter(torch.tensor(1.0))
        self.w_rev_rates = nn.Parameter(torch.tensor(1.0))
        self.w_has       = nn.Parameter(torch.tensor(1.0))
        self.w_rev_has   = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_dict, edge_index_dict):
        h1 = self.conv1(x_dict, edge_index_dict)
        h1 = {
            'user':   h1['user']   * self.w_rev_rates,
            'recipe': h1['recipe'] * (self.w_rates + self.w_has) / 2.0,
            'ing':    h1['ing']    * self.w_rev_has,
        }
        h1 = {k: v.relu() for k, v in h1.items()}

        h2 = self.conv2(h1, edge_index_dict)
        h2 = {
            'user':   h2['user']   * self.w_rev_rates,
            'recipe': h2['recipe'] * (self.w_rates + self.w_has) / 2.0,
            'ing':    h2['ing']    * self.w_rev_has,
        }
        out = {k: (h1[k] + h2[k]) / 2.0 for k in h2.keys()}
        return out

model = HeteroGAT(
    hidden=128,
    heads=int(os.getenv("HEADS","4")),
    dropout=float(os.getenv("DROPOUT","0.1")),
).to(device)

# ===== Cosine BPR =====
def cosine(a, b, eps=1e-8):
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)

def bpr_loss(u_emb, i_pos, i_neg):
    s_pos = cosine(u_emb, i_pos)
    s_neg = cosine(u_emb, i_neg)
    return -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-9).mean()

# ===== Negatives =====
def sample_hard_neg(i_pos_idx, tries=20):
    neg = []
    for ip in i_pos_idx.tolist():
        base = rec2ing[ip]
        j = None
        for _ in range(tries):
            cand = int(torch.randint(0, meta['num_recipes'], (1,), device=device))
            if cand != ip and len(base & rec2ing[cand]) > 0:
                j = cand
                break
        if j is None:
            j = int(torch.randint(0, meta['num_recipes'], (1,), device=device))
        neg.append(j)
    return torch.tensor(neg, device=device)

def sample_popular_neg(batch_size):
    idx = np.random.choice(meta['num_recipes'], size=batch_size, p=pop)
    return torch.tensor(idx, device=device, dtype=torch.long)

def sample_mixed_neg(i_pos_idx, p_hard=0.7):
    hard = sample_hard_neg(i_pos_idx)
    popu = sample_popular_neg(len(i_pos_idx))
    mask = (torch.rand(len(i_pos_idx), device=device) < p_hard)
    return torch.where(mask, hard, popu)

# ===== Train config =====
opt = torch.optim.Adam(
    [
        {"params": model.parameters(), "lr": float(os.getenv("LR", "0.002"))},
        {"params": emb_user.parameters(), "lr": float(os.getenv("LR", "0.002"))},
        {"params": emb_recipe.parameters(), "lr": float(os.getenv("LR", "0.002"))},
        {"params": emb_ing.parameters(), "lr": float(os.getenv("LR", "0.002"))},
    ],
    weight_decay=float(os.getenv("WD", "1e-4"))
)

def get_repr():
    data['user'].x   = emb_user.weight.to(device)
    data['recipe'].x = emb_recipe.weight.to(device)
    data['ing'].x    = emb_ing.weight.to(device)
    out = model(data.x_dict, data.edge_index_dict)
    return out['user'], out['recipe'], out['ing']

edge_ui = data['user','rates','recipe'].edge_index.t()
num_epochs = int(os.getenv("EPOCHS", "120"))
bs         = int(os.getenv("BATCH",  "12288"))
p_hard     = float(os.getenv("PHARD", "0.8"))

# ===== Dev eval (Recall@10 / NDCG@10) =====
NEG_DEV = int(os.getenv("NEG_DEV","1000"))

# seen mask from TRAIN
seen = defaultdict(set)
for u, i in E_TR.tolist():
    seen[int(u)].add(int(i))

@torch.no_grad()
def eval_split(u_repr, r_repr, split_edges, K=10, neg_samples=1000):
    if len(split_edges) == 0:
        return 0.0, 0.0, 0
    recalls, ndcgs, n = [], [], 0
    U, I = meta["num_users"], meta["num_recipes"]
    for (u, i_pos) in split_edges.tolist():
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
        a = r_repr[cand]; b = u_repr[u].unsqueeze(0)
        a = a / (a.norm(dim=1, keepdim=True) + 1e-8)
        b = b / (b.norm(dim=1, keepdim=True) + 1e-8)
        scores = (a @ b.T).squeeze(1).cpu().numpy()
        rank = np.argsort(-scores)
        topk = rank[:K]
        hit = 1 if 0 in topk else 0
        pos_idx = int(np.where(rank==0)[0][0])
        dcg = 1.0/np.log2(2+pos_idx) if pos_idx < K else 0.0
        recalls.append(hit); ndcgs.append(dcg)
    return float(np.mean(recalls)), float(np.mean(ndcgs)), n

# ===== Training loop with early stopping =====
EVAL_EVERY = int(os.getenv("EVAL_EVERY", "4"))
PATIENCE   = int(os.getenv("PATIENCE", "6"))
BEST_METRIC = -1.0
best_state = None
no_improve = 0

for epoch in range(1, num_epochs+1):
    model.train(); opt.zero_grad()
    u_repr, r_repr, _ = get_repr()

    idx = torch.randint(0, edge_ui.size(0), (bs,), device=device)
    u_idx     = edge_ui[idx, 0]
    i_pos_idx = edge_ui[idx, 1]
    i_neg_idx = sample_mixed_neg(i_pos_idx, p_hard=p_hard)

    loss = bpr_loss(u_repr[u_idx], r_repr[i_pos_idx], r_repr[i_neg_idx])
    loss.backward(); opt.step()

    if epoch % 2 == 0:
        print(f"[{epoch:02d}/{num_epochs}] bpr_loss={loss.item():.4f}")

    if epoch % EVAL_EVERY == 0 or epoch == num_epochs:
        model.eval()
        u_eval, r_eval, _ = get_repr()
        r_dev, n_dev, _ = eval_split(u_eval, r_eval, E_DV, K=10, neg_samples=NEG_DEV)
        ndcg_dev = n_dev
        metric = (r_dev + ndcg_dev) / 2.0  # 併用：Recall@10 と NDCG@10 の平均
        print(f"[dev@{epoch}] R@10={r_dev:.4f}  NDCG@10={ndcg_dev:.4f}  metric={metric:.4f}")
        if metric > BEST_METRIC:
            BEST_METRIC = metric
            best_state = {
                "emb_u": u_eval.detach().cpu(),
                "emb_r": r_eval.detach().cpu(),
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[early stop] no improvement for {PATIENCE} evals; best metric={BEST_METRIC:.4f}")
                break

# Save best (or last if none)
if best_state is None:
    u_final, r_final, _ = get_repr()
    best_state = {"emb_u": u_final.detach().cpu(), "emb_r": r_final.detach().cpu()}

torch.save(best_state["emb_u"], f"{ART}/user_repr.pt")
torch.save(best_state["emb_r"], f"{ART}/recipe_repr.pt")
print("Saved best embeddings to artifacts/.")
