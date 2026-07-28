import os
import ast
import random
import glob
import pandas as pd

# 1. 경로 탐색
working_dir = "C:/Users/user/Desktop/서울대"
data_dir = os.path.join(working_dir, "snuaichallenge_data")

# 데이터셋 경로가 없으면 kaggle/aux_upload 등 다른 곳도 탐색
if not os.path.exists(data_dir):
    hits = glob.glob(os.path.join(working_dir, "**/train.csv"), recursive=True)
    if hits:
        data_dir = os.path.dirname(hits[0])

print(f"Data Dir: {data_dir}")

def find_csv(name):
    hits = glob.glob(os.path.join(working_dir, f"**/{name}"), recursive=True)
    return hits[0] if hits else None

# 1488 완주 데이터셋은 _half 가 아닌 원본 aug_weights_exp16.csv 를 사용함!
aug_weights_path = find_csv("aug_weights_exp16.csv")
clip_feats_path  = find_csv("snu_clip_features.csv")
holdout_path     = find_csv("holdout_300.csv")

print(f"Forced aug_weights (1488 step dataset): {aug_weights_path}")
print(f"clip: {clip_feats_path}")
print(f"holdout: {holdout_path}")

# 설정값 (FINAL_8B_v2.py 와 동일)
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

if os.path.exists(os.path.join(data_dir, "train.csv")):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    print(f"Loaded train.csv with {len(train_df)} rows.")
else:
    print("Error: train.csv not found!")
    exit(1)

if holdout_path and os.path.exists(holdout_path):
    hold = set(pd.read_csv(holdout_path)["Id"])
    train_df = train_df[~train_df["Id"].isin(hold)].reset_index(drop=True)
    print(f"holdout {len(hold)}개 제외 -> train {len(train_df)}")

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
                          paths=[os.path.join(data_dir, "train", row["Id"], f) for f in shown],
                          target=str(target)))

# 셔플 수행
rng.shuffle(items)
total_items = len(items)
print(f"Total training items: {total_items}")

# 3. 인덱스 분할 계산
split_idx = CFG["step_stopped"] * CFG["grad_accum"]
print(f"Split index for step {CFG['step_stopped']}: {split_idx}")

trained_items = items[:split_idx]
untrained_items = items[split_idx:]

print(f"Trained items: {len(trained_items)}")
print(f"Untrained items: {len(untrained_items)}")

# 4. 통계 추출
trained_ids = set(it["id"] for it in trained_items)
untrained_ids = set(it["id"] for it in untrained_items)

completely_unseen = untrained_ids - trained_ids
partially_seen = untrained_ids & trained_ids

print(f"\nUnique IDs in train dataset (after holdout): {len(train_df)}")
print(f"Unique IDs trained on at least once: {len(trained_ids)}")
print(f"Unique IDs completely skipped (unseen): {len(completely_unseen)}")
print(f"Unique IDs partially trained (some augmentations skipped): {len(partially_seen)}")

# 5. CSV 파일로 저장
# completely_unseen 에 해당하는 train_df의 행들을 추출하여 저장
df_unseen = train_df[train_df["Id"].isin(completely_unseen)].copy()
unseen_csv_path = os.path.join(working_dir, "untrained_samples_report.csv")
df_unseen.to_csv(unseen_csv_path, index=False)
print(f"Saved {len(df_unseen)} completely untrained samples to: {unseen_csv_path}")

# 6. 미학습 샘플 중 일부 예시 출력
if len(df_unseen) > 0:
    print("\n--- Example of completely skipped questions ---")
    for idx, row in df_unseen.head(5).iterrows():
        print(f"ID: {row['Id']}")
        print(f"Sentence: {row['Sentence']}")
        print(f"Answer: {row['Answer']}")
        print("-" * 50)
