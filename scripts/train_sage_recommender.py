# scripts/train_sage_recommender.py
import os, json, numpy as np, torch
from torch import nn
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv

# ======================
# Device (MPS → CPU fallback)
# ======================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Device] Using: {device} (MPS available? {torch.backends.mps.is_available()})")

# ======================
# Load artifacts
# ======================
ART = "artifacts"
meta = json.load(open(f"{ART}/meta.json"))
E_UR_ALL = torch.from_numpy(np.load(f"{ART}/edge_ur.npy")).long()
E_RI_NP  = np.load(f"{ART}/edge_ri.npy")               # numpy for preproc
E_TR     = torch.from_numpy(np.load(f"{ART}/train_edge_ur.npy")).long()
E_TE     = torch.from_numpy(np.load(f"{ART}/test_edge_ur.npy")).long()

# recipe -> ingredients set（Hard Negative用）
rec2ing = defaultdict(set)
for r, g in E_RI_NP:
    rec2ing[int(r)].add(int(g))

# 人気分布（レシピの“材料辺”回数を proxy に）
pop = np.bincount(E_RI_NP[:, 0], minlength=meta['num_recipes']).astype(float)
if pop.sum() == 0:
    pop = np.ones_like(pop)
pop = pop / pop.sum()

# ======================
# Hetero graph（双方向エッジも追加）
# ======================
def add_rev(edge_index):
    """Return edge_index and reversed edge_index as (N,2) int64 tensors on device."""
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

# user <-> recipe (train only)
ur, ru = add_rev(E_TR)
data['user','rates','recipe'].edge_index     = ur.t().contiguous()
data['recipe','rev_rates','user'].edge_index = ru.t().contiguous()

# recipe <-> ing
ri = torch.from_numpy(E_RI_NP).long()
ri, ir = add_rev(ri)
data['recipe','has','ing'].edge_index     = ri.t().contiguous()
data['ing','rev_has','recipe'].edge_index = ir.t().contiguous()

data = data.to(device)

# ======================
# Learnable ID embeddings
# ======================
dim_id = 64
emb_user   = nn.Embedding(meta['num_users'], dim_id).to(device)
emb_recipe = nn.Embedding(meta['num_recipes'], dim_id).to(device)
emb_ing    = nn.Embedding(meta['num_ings'], dim_id).to(device)

# ======================
# Hetero GraphSAGE（to_hetero を使わない版）
# 層1で recipe↔ing を強め、層2で user↔recipe を強めたい場合は
# relation weight を学習させても良い（ここではシンプルに平均）。
# LightGCN 風に中間層と最終層の平均を出力に採用。
# ======================
def make_conv(in_channels_dict, out_channels):
    convs = {
        ('user','rates','recipe'):      SAGEConv((in_channels_dict['user'],   in_channels_dict['recipe']), out_channels),
        ('recipe','rev_rates','user'):  SAGEConv((in_channels_dict['recipe'], in_channels_dict['user']),   out_channels),
        ('recipe','has','ing'):         SAGEConv((in_channels_dict['recipe'], in_channels_dict['ing']),    out_channels),
        ('ing','rev_has','recipe'):     SAGEConv((in_channels_dict['ing'],    in_channels_dict['recipe']), out_channels),
    }
    return HeteroConv(convs, aggr='mean')

class HeteroSAGE(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        in_dims1 = {'user': dim_id, 'recipe': dim_id, 'ing': dim_id}
        in_dims2 = {'user': hidden, 'recipe': hidden, 'ing': hidden}
        self.conv1 = make_conv(in_dims1, hidden)
        self.conv2 = make_conv(in_dims2, hidden)

    def forward(self, x_dict, edge_index_dict):
        h1 = self.conv1(x_dict, edge_index_dict)
        h1 = {k: v.relu() for k, v in h1.items()}
        h2 = self.conv2(h1, edge_index_dict)    # final（活性なし）
        out = {k: (h1[k] + h2[k]) / 2.0 for k in h2.keys()}  # LightGCN 風平均
        return out

model = HeteroSAGE(hidden=128).to(device)

# ======================
# Cosine BPR loss（ノルム安定）
# ======================
def cosine(a, b, eps=1e-8):
    a = a / (a.norm(dim=-1, keepdim=True) + eps)
    b = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a * b).sum(dim=-1)

def bpr_loss(u_emb, i_pos, i_neg):
    s_pos = cosine(u_emb, i_pos)
    s_neg = cosine(u_emb, i_neg)
    return -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-9).mean()

# ======================
# Negatives: Hard（材料が1つ以上被る）と Popular のミックス
# ======================
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
        if j is None:  # fallback: random
            j = int(torch.randint(0, meta['num_recipes'], (1,), device=device))
        neg.append(j)
    return torch.tensor(neg, device=device)

def sample_popular_neg(batch_size):
    # numpy choice → torch.tensor
    idx = np.random.choice(meta['num_recipes'], size=batch_size, p=pop)
    return torch.tensor(idx, device=device, dtype=torch.long)

def sample_mixed_neg(i_pos_idx, p_hard=0.5):
    hard = sample_hard_neg(i_pos_idx)
    popu = sample_popular_neg(len(i_pos_idx))
    mask = (torch.rand(len(i_pos_idx), device=device) < p_hard)
    return torch.where(mask, hard, popu)

# ======================
# Training loop
# ======================
opt = torch.optim.Adam(
    [
        {"params": model.parameters(), "lr": float(os.getenv("LR", "0.0015"))},
        {"params": emb_user.parameters(), "lr": float(os.getenv("LR", "0.0015"))},
        {"params": emb_recipe.parameters(), "lr": float(os.getenv("LR", "0.0015"))},
        {"params": emb_ing.parameters(), "lr": float(os.getenv("LR", "0.0015"))},
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

num_epochs = int(os.getenv("EPOCHS", "80"))
bs         = int(os.getenv("BATCH",  "6144"))
p_hard     = float(os.getenv("PHARD", "0.5"))

for epoch in range(1, num_epochs+1):
    model.train(); opt.zero_grad()
    u_repr, r_repr, _ = get_repr()

    # Mini-batch of observed (u,i+)
    idx = torch.randint(0, edge_ui.size(0), (bs,), device=device)
    u_idx    = edge_ui[idx, 0]
    i_pos_idx= edge_ui[idx, 1]

    # Hard/Popular mixed negatives
    i_neg_idx = sample_mixed_neg(i_pos_idx, p_hard=p_hard)

    loss = bpr_loss(u_repr[u_idx], r_repr[i_pos_idx], r_repr[i_neg_idx])
    loss.backward(); opt.step()

    if epoch % 2 == 0:
        print(f"[{epoch:02d}/{num_epochs}] bpr_loss={loss.item():.4f}")

# Save embeddings for evaluation
u_repr, r_repr, _ = get_repr()
torch.save(u_repr.detach().cpu(), f"{ART}/user_repr.pt")
torch.save(r_repr.detach().cpu(), f"{ART}/recipe_repr.pt")
print("Saved embeddings to artifacts/.")
