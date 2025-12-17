# scripts/train_sage_warmup.py
import os, json, numpy as np, torch, random
from torch import nn
from collections import defaultdict
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv

# ===== Repro =====
seed = int(os.getenv("SEED", "42"))
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# ===== Device =====
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Device] Using: {device} (MPS available? {torch.backends.mps.is_available()})")

# ===== Data =====
ART = "artifacts"
meta = json.load(open(f"{ART}/meta.json"))
E_RI_NP  = np.load(f"{ART}/edge_ri.npy")
E_TR     = torch.from_numpy(np.load(f"{ART}/train_edge_ur.npy")).long()

def add_rev(edge_index):
    e = edge_index if isinstance(edge_index, torch.Tensor) else torch.as_tensor(edge_index, dtype=torch.long)
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

# ===== Embeds =====
dim_id = 64
emb_user   = nn.Embedding(meta['num_users'], dim_id).to(device)
emb_recipe = nn.Embedding(meta['num_recipes'], dim_id).to(device)
emb_ing    = nn.Embedding(meta['num_ings'], dim_id).to(device)

# ===== Model (Hetero SAGE) =====
def make_conv(in_channels_dict, out_channels):
    return HeteroConv({
        ('user','rates','recipe'):     SAGEConv((in_channels_dict['user'],   in_channels_dict['recipe']), out_channels),
        ('recipe','rev_rates','user'): SAGEConv((in_channels_dict['recipe'], in_channels_dict['user']),   out_channels),
        ('recipe','has','ing'):        SAGEConv((in_channels_dict['recipe'], in_channels_dict['ing']),    out_channels),
        ('ing','rev_has','recipe'):    SAGEConv((in_channels_dict['ing'],    in_channels_dict['recipe']), out_channels),
    }, aggr='mean')

class HeteroSAGE(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.conv1 = make_conv({'user':64,'recipe':64,'ing':64}, hidden)
        self.conv2 = make_conv({'user':hidden,'recipe':hidden,'ing':hidden}, hidden)
    def forward(self, x_dict, edge_index_dict):
        h1 = self.conv1(x_dict, edge_index_dict)
        h1 = {k: v.relu() for k,v in h1.items()}
        h2 = self.conv2(h1, edge_index_dict)
        out = {k: (h1[k]+h2[k])/2.0 for k in h2.keys()}
        return out

model = HeteroSAGE(hidden=128).to(device)

def cosine(a,b,eps=1e-8):
    a = a/(a.norm(dim=-1,keepdim=True)+eps)
    b = b/(b.norm(dim=-1,keepdim=True)+eps)
    return (a*b).sum(dim=-1)

def bpr_loss(u_emb, i_pos, i_neg):
    return -torch.log(torch.sigmoid(cosine(u_emb,i_pos)-cosine(u_emb,i_neg))+1e-9).mean()

# Popularity（optional for neg mix）
pop = np.bincount(E_RI_NP[:,0], minlength=meta['num_recipes']).astype(float)
pop = pop/(pop.sum() if pop.sum()>0 else 1.0)

def sample_popular_neg(bs):
    idx = np.random.choice(meta['num_recipes'], size=bs, p=pop)
    return torch.tensor(idx, device=device, dtype=torch.long)

edge_ui = data['user','rates','recipe'].edge_index.t()
def get_repr():
    data['user'].x   = emb_user.weight.to(device)
    data['recipe'].x = emb_recipe.weight.to(device)
    data['ing'].x    = emb_ing.weight.to(device)
    out = model(data.x_dict, data.edge_index_dict)
    return out['user'], out['recipe'], out['ing']

# ===== Train (短期) =====
epochs = int(os.getenv("EPOCHS","30"))
bs     = int(os.getenv("BATCH","12288"))
lr     = float(os.getenv("LR","0.002"))
wd     = float(os.getenv("WD","1e-4"))
opt = torch.optim.Adam([
    {"params": model.parameters(), "lr": lr},
    {"params": emb_user.parameters(), "lr": lr},
    {"params": emb_recipe.parameters(), "lr": lr},
    {"params": emb_ing.parameters(), "lr": lr},
], weight_decay=wd)

for ep in range(1, epochs+1):
    model.train(); opt.zero_grad()
    u_repr, r_repr, _ = get_repr()
    idx = torch.randint(0, edge_ui.size(0), (bs,), device=device)
    u_idx = edge_ui[idx,0]; i_pos = edge_ui[idx,1]
    i_neg = sample_popular_neg(bs)  # シンプルにpopular負例
    loss = bpr_loss(u_repr[u_idx], r_repr[i_pos], r_repr[i_neg])
    loss.backward(); opt.step()
    if ep%2==0: print(f"[warmup {ep:02d}/{epochs}] loss={loss.item():.4f}")

# ===== Save warm-start weights（Embeddingの初期値として使う） =====
torch.save(emb_user.weight.detach().cpu(),   f"{ART}/emb_user_init.pt")
torch.save(emb_recipe.weight.detach().cpu(), f"{ART}/emb_recipe_init.pt")
print("Saved warm-start embeddings: artifacts/emb_user_init.pt, emb_recipe_init.pt")
