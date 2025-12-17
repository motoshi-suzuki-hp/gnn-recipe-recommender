# scripts/train_gat_warm_multi.py
import os, json, numpy as np, torch, random
from torch import nn
from collections import defaultdict
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv

# ===== Repro =====
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

# rec->ing
rec2ing = defaultdict(set)
for r,g in E_RI_NP: rec2ing[int(r)].add(int(g))
# popularity
pop = np.bincount(E_RI_NP[:,0], minlength=meta['num_recipes']).astype(float)
pop = pop/(pop.sum() if pop.sum()>0 else 1.0)

# ===== Graph =====
def add_rev(edge_index):
    e = edge_index if isinstance(edge_index, torch.Tensor) else torch.as_tensor(edge_index, dtype=torch.long)
    rev = torch.stack([e[:,1], e[:,0]], dim=1)
    return e, rev

data = HeteroData()
data['user'].num_nodes   = meta['num_users']
data['recipe'].num_nodes = meta['num_recipes']
data['ing'].num_nodes    = meta['num_ings']
ur, ru = add_rev(E_TR); data['user','rates','recipe'].edge_index=ur.t(); data['recipe','rev_rates','user'].edge_index=ru.t()
ri = torch.from_numpy(E_RI_NP).long()
ri, ir = add_rev(ri); data['recipe','has','ing'].edge_index=ri.t(); data['ing','rev_has','recipe'].edge_index=ir.t()
data = data.to(device)

# ===== Embeds (warm start if available) =====
dim_id = 64
emb_user   = nn.Embedding(meta['num_users'], dim_id).to(device)
emb_recipe = nn.Embedding(meta['num_recipes'], dim_id).to(device)
emb_ing    = nn.Embedding(meta['num_ings'], dim_id).to(device)

warm_u = f"{ART}/emb_user_init.pt"
warm_r = f"{ART}/emb_recipe_init.pt"
if os.path.exists(warm_u) and os.path.exists(warm_r):
    print("[WarmStart] Loading emb_user_init.pt & emb_recipe_init.pt")
    emb_user.weight.data.copy_(torch.load(warm_u))
    emb_recipe.weight.data.copy_(torch.load(warm_r))
else:
    print("[WarmStart] Not found; using random init.")

# ===== HeteroGAT =====
def make_conv(in_channels_dict, out_channels, heads=4, dropout=0.1):
    cdim = out_channels//heads
    convs = {
        ('user','rates','recipe'):     GATConv((in_channels_dict['user'],   in_channels_dict['recipe']), cdim, heads=heads, add_self_loops=False, concat=True, dropout=dropout),
        ('recipe','rev_rates','user'): GATConv((in_channels_dict['recipe'], in_channels_dict['user']),   cdim, heads=heads, add_self_loops=False, concat=True, dropout=dropout),
        ('recipe','has','ing'):        GATConv((in_channels_dict['recipe'], in_channels_dict['ing']),    cdim, heads=heads, add_self_loops=False, concat=True, dropout=dropout),
        ('ing','rev_has','recipe'):    GATConv((in_channels_dict['ing'],    in_channels_dict['recipe']), cdim, heads=heads, add_self_loops=False, concat=True, dropout=dropout),
    }
    return HeteroConv(convs, aggr='mean')

class HeteroGAT(nn.Module):
    def __init__(self, hidden=128, heads=4, dropout=0.1):
        super().__init__()
        self.conv1 = make_conv({'user':64,'recipe':64,'ing':64}, hidden, heads=heads, dropout=dropout)
        self.conv2 = make_conv({'user':hidden,'recipe':hidden,'ing':hidden}, hidden, heads=heads, dropout=dropout)
        # relation-wise gains
        self.w_rates     = nn.Parameter(torch.tensor(1.0))
        self.w_rev_rates = nn.Parameter(torch.tensor(1.0))
        self.w_has       = nn.Parameter(torch.tensor(1.0))
        self.w_rev_has   = nn.Parameter(torch.tensor(1.0))
    def forward(self, x_dict, edge_index_dict):
        h1 = self.conv1(x_dict, edge_index_dict)
        h1 = {'user':h1['user']*self.w_rev_rates, 'recipe':h1['recipe']*(self.w_rates+self.w_has)/2.0, 'ing':h1['ing']*self.w_rev_has}
        h1 = {k:v.relu() for k,v in h1.items()}
        h2 = self.conv2(h1, edge_index_dict)
        h2 = {'user':h2['user']*self.w_rev_rates, 'recipe':h2['recipe']*(self.w_rates+self.w_has)/2.0, 'ing':h2['ing']*self.w_rev_has}
        return {k:(h1[k]+h2[k])/2.0 for k in h2.keys()}

model = HeteroGAT(
    hidden=128,
    heads=int(os.getenv("HEADS","4")),
    dropout=float(os.getenv("DROPOUT","0.1")),
).to(device)

# ===== Loss / utils =====
def cosine(a,b,eps=1e-8):
    a = a/(a.norm(dim=-1,keepdim=True)+eps)
    b = b/(b.norm(dim=-1,keepdim=True)+eps)
    return (a*b).sum(dim=-1)

def bpr_multi(u_emb, pos_emb, neg_emb):  # neg_emb: [B, K, D]
    # s_pos: [B,1], s_neg: [B,K]
    s_pos = cosine(u_emb, pos_emb).unsqueeze(1)
    B,K,D = neg_emb.shape
    u_rep = u_emb.unsqueeze(1).expand(B,K,u_emb.size(-1))
    s_neg = cosine(u_rep.reshape(B*K,-1), neg_emb.reshape(B*K,-1)).reshape(B,K)
    loss = -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-9).mean()
    return loss

# ===== Negatives (multi-k) =====
PHARD = float(os.getenv("PHARD","0.8"))
KNEG  = int(os.getenv("KNEG","5"))

def sample_hard_neg_vec(i_pos_idx, tries=20):
    # return [B,] tensor
    neg = []
    for ip in i_pos_idx.tolist():
        base = rec2ing[ip]; j=None
        for _ in range(tries):
            cand = int(torch.randint(0, meta['num_recipes'], (1,), device=device))
            if cand!=ip and len(base & rec2ing[cand])>0: j=cand; break
        if j is None: j = int(torch.randint(0, meta['num_recipes'], (1,), device=device))
        neg.append(j)
    return torch.tensor(neg, device=device)

def sample_popular_neg_vec(bs):
    idx = np.random.choice(meta['num_recipes'], size=bs, p=pop)
    return torch.tensor(idx, device=device, dtype=torch.long)

def sample_mixed_multi(i_pos_idx, k=5, p_hard=0.8):
    # returns [B, K]
    B = len(i_pos_idx)
    outs = []
    for _ in range(k):
        hard = sample_hard_neg_vec(i_pos_idx)
        popu = sample_popular_neg_vec(B)
        mask = (torch.rand(B, device=device) < p_hard)
        outs.append(torch.where(mask, hard, popu))
    return torch.stack(outs, dim=1)

# ===== Train config =====
lr = float(os.getenv("LR","0.002"))
wd = float(os.getenv("WD","1e-4"))
bs = int(os.getenv("BATCH","12288"))
epochs = int(os.getenv("EPOCHS","120"))
EVAL_EVERY = int(os.getenv("EVAL_EVERY","4"))
PATIENCE   = int(os.getenv("PATIENCE","6"))
NEG_DEV    = int(os.getenv("NEG_DEV","1000"))

opt = torch.optim.Adam([
    {"params": model.parameters(), "lr": lr},
    {"params": emb_user.parameters(), "lr": lr},
    {"params": emb_recipe.parameters(), "lr": lr},
    {"params": emb_ing.parameters(), "lr": lr},
], weight_decay=wd)

def get_repr():
    data['user'].x   = emb_user.weight.to(device)
    data['recipe'].x = emb_recipe.weight.to(device)
    data['ing'].x    = emb_ing.weight.to(device)
    out = model(data.x_dict, data.edge_index_dict)
    return out['user'], out['recipe'], out['ing']

edge_ui = data['user','rates','recipe'].edge_index.t()

# seen mask for dev eval
seen = defaultdict(set)
for u,i in E_TR.tolist(): seen[int(u)].add(int(i))

@torch.no_grad()
def eval_split(u_repr, r_repr, split_edges, K=10, neg_samples=1000):
    if len(split_edges)==0: return 0.0,0.0,0
    recalls, ndcgs, n = [], [], 0
    I = meta["num_recipes"]
    for (u,i_pos) in split_edges.tolist():
        u = int(u); i_pos = int(i_pos)
        if u >= len(u_repr): continue
        n += 1
        cand = [i_pos]
        while len(cand) < neg_samples+1:
            j = np.random.randint(0, I)
            if j==i_pos or j in seen[u]: continue
            cand.append(j)
        a = r_repr[cand]; b = u_repr[u].unsqueeze(0)
        a = a/(a.norm(dim=1,keepdim=True)+1e-8)
        b = b/(b.norm(dim=1,keepdim=True)+1e-8)
        scores = (a @ b.T).squeeze(1).cpu().numpy()
        rank = np.argsort(-scores); topk = rank[:K]
        hit = 1 if 0 in topk else 0
        pos_idx = int(np.where(rank==0)[0][0])
        dcg = 1.0/np.log2(2+pos_idx) if pos_idx < K else 0.0
        recalls.append(hit); ndcgs.append(dcg)
    return float(np.mean(recalls)), float(np.mean(ndcgs)), n

# ===== Train with early stopping =====
BEST, best_state, no_imp = -1.0, None, 0
for ep in range(1, epochs+1):
    model.train(); opt.zero_grad()
    u_repr, r_repr, _ = get_repr()

    idx = torch.randint(0, edge_ui.size(0), (bs,), device=device)
    u_idx = edge_ui[idx,0]
    i_pos = edge_ui[idx,1]
    # multi-negative indices [B, K]
    i_negs = sample_mixed_multi(i_pos, k=KNEG, p_hard=PHARD)
    # gather embeddings
    loss = bpr_multi(u_repr[u_idx], r_repr[i_pos], r_repr[i_negs])  # i_negs uses advanced indexing broadcasting
    loss.backward(); opt.step()

    if ep%2==0: print(f"[{ep:02d}/{epochs}] loss={loss.item():.4f}")

    if ep%EVAL_EVERY==0 or ep==epochs:
        model.eval()
        u_eval, r_eval, _ = get_repr()
        r_dev, n_dev, _ = eval_split(u_eval, r_eval, E_DV, K=10, neg_samples=NEG_DEV)
        metric = (r_dev + n_dev)/2.0
        print(f"[dev@{ep}] R@10={r_dev:.4f} NDCG@10={n_dev:.4f} metric={metric:.4f}")
        if metric > BEST:
            BEST = metric; no_imp = 0
            best_state = {"emb_u": u_eval.detach().cpu(), "emb_r": r_eval.detach().cpu()}
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"[early stop] metric no improve for {PATIENCE} evals; best={BEST:.4f}")
                break

# ===== Save best =====
if best_state is None:
    u_eval, r_eval, _ = get_repr()
    best_state = {"emb_u": u_eval.detach().cpu(), "emb_r": r_eval.detach().cpu()}
torch.save(best_state["emb_u"], f"{ART}/user_repr.pt")
torch.save(best_state["emb_r"], f"{ART}/recipe_repr.pt")
print("Saved best embeddings to artifacts/.")
