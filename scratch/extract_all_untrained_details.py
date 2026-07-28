import os
import ast
import random
import glob
import pandas as pd

# 1. 경로 탐색
working_dir = "C:/Users/user/Desktop/서울대"
data_dir = os.path.join(working_dir, "snuaichallenge_data")
if not os.path.exists(data_dir):
    hits = glob.glob(os.path.join(working_dir, "**/train.csv"), recursive=True)
    if hits:
        data_dir = os.path.dirname(hits[0])

aug_weights_path = find_csv = lambda name: (glob.glob(os.path.join(working_dir, f"**/{name}"), recursive=True) or [None])[0]
aug_weights_path = find_csv("aug_weights_exp16.csv")
clip_feats_path  = find_csv("snu_clip_features.csv")
holdout_path     = find_csv("holdout_300.csv")

CFG = {
    "aug_mult": 1,
    "hard_shuffle": False,
    "seed": 42,
    "grad_accum": 16,
    "step_stopped": 977,
}

# 2. 데이터 준비
random.seed(CFG["seed"])
rng = random.Random(CFG["seed"])

def chrono(ans):
    c = [0] * 4
    for i, p in enumerate(ans):
        c[p - 1] = i + 1
    return c

PAIR_COLS = {
    (1,2): "dist_12", (1,3): "dist_13", (1,4): "dist_14",
    (2,3): "dist_23", (2,4): "dist_24", (3,4): "dist_34"
}
def load_pairs(path):
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    out = {}
    for r in df.itertuples():
        out[r.Id] = [p for p, c in PAIR_COLS.items() if getattr(r, c) < 0.20]
    return out

def hard_perm(seen, pairs, files, tfiles):
    best, bs = None, -1
    for _ in range(16):
        cand = list(range(4))
        rng.shuffle(cand)
        if tuple(cand) in seen or [files[j] for j in cand] == tfiles:
            continue
        pos = {o: s for s, o in enumerate(cand)}
        sc = sum(1 for a, b in pairs if pos[a-1] > pos[b-1]) * 10 + sum(abs(pos[i]-i) for i in range(4))
        if sc > bs:
            best, bs = cand, sc
    return best

train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
if holdout_path and os.path.exists(holdout_path):
    hold = set(pd.read_csv(holdout_path)["Id"])
    train_df = train_df[~train_df["Id"].isin(hold)].reset_index(drop=True)

augw = {}
if aug_weights_path and os.path.exists(aug_weights_path):
    w = pd.read_csv(aug_weights_path)
    augw = dict(zip(w["Id"], w["aug_mult"].astype(int)))

pairs = load_pairs(clip_feats_path)

items = []
for _, row in train_df.iterrows():
    mult = augw.get(row["Id"], CFG["aug_mult"])
    files = [row["Input_1"], row["Input_2"], row["Input_3"], row["Input_4"]]
    ans = ast.literal_eval(row["Answer"])
    ch = chrono(ans)
    tfiles = [files[n-1] for n in ch]
    sp = pairs.get(row["Id"], [])
    seen = set()
    for v in range(mult):
        if v == 0:
            perm = list(range(4))
        else:
            perm = hard_perm(seen, sp, files, tfiles) if CFG["hard_shuffle"] else None
            if perm is None:
                perm = list(range(4))
                for _ in range(10):
                    rng.shuffle(perm)
                    if tuple(perm) not in seen:
                        break
        seen.add(tuple(perm))
        shown = [files[j] for j in perm]
        target = [shown.index(f) + 1 for f in tfiles]
        items.append(dict(id=row["Id"], sentence=row["Sentence"],
                          shown_images=str(shown), target=str(target)))

# 셔플
rng.shuffle(items)

# 분할
split_idx = CFG["step_stopped"] * CFG["grad_accum"]
untrained_items = items[split_idx:]

# 1. raw untrained items 저장 (8,182행)
df_raw = pd.DataFrame(untrained_items)
raw_path = os.path.join(working_dir, "all_untrained_items_raw.csv")
df_raw.to_csv(raw_path, index=False)
print(f"Saved {len(df_raw)} raw untrained items to: {raw_path}")

# 2. unique untrained IDs 저장 (5,898행)
unique_ids = sorted(list(set(it["id"] for it in untrained_items)))
df_unique = pd.DataFrame({"Id": unique_ids})
unique_path = os.path.join(working_dir, "all_untrained_unique_ids.csv")
df_unique.to_csv(unique_path, index=False)
print(f"Saved {len(df_unique)} unique untrained IDs to: {unique_path}")
