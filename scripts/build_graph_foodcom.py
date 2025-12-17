# scripts/build_graph_foodcom.py
import os, json, numpy as np, pandas as pd, ast

DATA_DIR = os.getenv("DATA_DIR", "data")
OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

REC_PATH = f"{DATA_DIR}/RAW_recipes.csv"
INT_PATH = f"{DATA_DIR}/RAW_interactions.csv"

assert os.path.exists(REC_PATH), f"Not found: {REC_PATH}"
assert os.path.exists(INT_PATH), f"Not found: {INT_PATH}"

# === Load RAW CSVs ===
recipes = pd.read_csv(REC_PATH)           # expects 'id','ingredients' (stringified list)
inter   = pd.read_csv(INT_PATH)           # expects 'user_id','recipe_id','rating','date'

# === Build ingredients.csv (on the fly) ===
rows = []
for rid, ings in zip(recipes['id'], recipes['ingredients']):
    try:
        lst = ast.literal_eval(ings) if isinstance(ings, str) else []
        for ing in lst:
            rows.append({"recipe_id": rid, "ingredient": str(ing)})
    except Exception:
        continue
ings_df = pd.DataFrame(rows)
ings_df['ingredient'] = ings_df['ingredient'].str.lower().str.strip()
ings_df = ings_df[ings_df['ingredient'].notnull() & (ings_df['ingredient']!="")]
print(f"[info] built ingredients table: {len(ings_df)} rows")

# === Minimal cleaning for interactions ===
if 'rating' in inter.columns:
    inter = inter[inter['rating'] >= 4].copy()   # implicit positives

# Filter active users & popular recipes
u_cnt = inter.groupby('user_id').size()
active_users = set(u_cnt[u_cnt>=5].index)
i_cnt = inter.groupby('recipe_id').size()
popular_recipes = set(i_cnt[i_cnt>=5].index)

inter = inter[inter.user_id.isin(active_users) & inter.recipe_id.isin(popular_recipes)].copy()
sub_recipes = recipes[recipes['id'].isin(popular_recipes)].copy()
sub_ings = ings_df[ings_df['recipe_id'].isin(popular_recipes)].copy()

# Remap ids to contiguous
def remap(series):
    uniq = {k:i for i,k in enumerate(series.dropna().unique())}
    return series.map(uniq), uniq

inter['u'], u_map = remap(inter['user_id'])
inter['i'], i_map = remap(inter['recipe_id'])
sub_ings['i'] = sub_ings['recipe_id'].map(i_map)
sub_ings = sub_ings.dropna(subset=['i']).astype({'i':int})
sub_ings['g'], g_map = remap(sub_ings['ingredient'])

# Save static edges
edge_ur = inter[['u','i']].drop_duplicates().values
edge_ri = sub_ings[['i','g']].drop_duplicates().values
np.save(f"{OUT_DIR}/edge_ur.npy", edge_ur)
np.save(f"{OUT_DIR}/edge_ri.npy", edge_ri)

# === Time-aware split: per-user 80/10/10 (train/dev/test) ===
if 'date' in inter.columns:
    inter['date'] = pd.to_datetime(inter['date'], errors='coerce')
else:
    inter['date'] = pd.Timestamp.now()

inter = inter.sort_values(['u','date'])
g = inter.groupby('u')

train_rows, dev_rows, test_rows = [], [], []
for u, df in g:
    n = len(df)
    if n < 3:
        # fallback to LOO
        if n == 1:
            test_rows.append(df[['u','i']])
        elif n == 2:
            train_rows.append(df.iloc[:1][['u','i']])
            test_rows.append(df.iloc[1:][['u','i']])
        else:
            train_rows.append(df.iloc[:-1][['u','i']])
            test_rows.append(df.iloc[-1:][['u','i']])
        continue
    t1 = int(n*0.8)
    t2 = int(n*0.9)
    train_rows.append(df.iloc[:t1][['u','i']])
    dev_rows.append(df.iloc[t1:t2][['u','i']])
    test_rows.append(df.iloc[t2:][['u','i']])

train_edges = (pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame(columns=['u','i'])).values
dev_edges   = (pd.concat(dev_rows,   ignore_index=True) if dev_rows   else pd.DataFrame(columns=['u','i'])).values
test_edges  = (pd.concat(test_rows,  ignore_index=True) if test_rows  else pd.DataFrame(columns=['u','i'])).values

np.save(f"{OUT_DIR}/train_edge_ur.npy", train_edges)
np.save(f"{OUT_DIR}/dev_edge_ur.npy",   dev_edges)
np.save(f"{OUT_DIR}/test_edge_ur.npy",  test_edges)

meta = {
    "num_users": int(len(u_map)),
    "num_recipes": int(len(i_map)),
    "num_ings": int(len(g_map)),
    "stats": {
        "interactions_pos": int(len(edge_ur)),
        "ur_train": int(len(train_edges)),
        "ur_dev": int(len(dev_edges)),
        "ur_test": int(len(test_edges)),
        "thresholds": {"user_min_inter": 5, "recipe_min_inter": 5}
    }
}
with open(f"{OUT_DIR}/meta.json","w") as f:
    json.dump(meta, f, indent=2)

print("=== Build done (RAW mode, time-split 80/10/10) ===")
print(json.dumps(meta, indent=2))

