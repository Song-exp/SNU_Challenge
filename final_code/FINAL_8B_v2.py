# ================================================================================
# SNU AI Challenge — Qwen3-VL-8B QLoRA Training Pipeline
# ================================================================================

import ast
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from transformers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# Initialize library installations
def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs])

pip("transformers==5.13.0", "peft", "bitsandbytes", "accelerate", "qwen-vl-utils")

# Set environment variables for memory management
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---- Configuration -----------------------------------------------------------
CFG = dict(
    model_id="Qwen/Qwen3-VL-8B-Instruct",
    prompt_v5=(
        "Look at the 4 images above labeled Image 1 to Image 4. Determine the "
        "correct chronological order of these images to match the sentence below.\n"
        'Sentence: "{s}"\nProvide the answer ONLY as a Python list of integers. '
        "Example: [1, 2, 3, 4]"
    ),
    aug_mult=2,
    hard_shuffle=True,
    lr=1e-4,
    lora_r=16,
    lora_alpha=32,
    lora_targets="q_proj,k_proj,v_proj,o_proj",
    grad_accum=16,
    max_pixels=512 * 384,
    warmup_ratio=0.03,
    seed=42,
    out="/kaggle/working/adapter",
    ckpt="/kaggle/working/ckpt",
    save_every=100,
    max_seconds=11.3 * 3600,
)

os.makedirs(CFG["ckpt"], exist_ok=True)
random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
rng = random.Random(CFG["seed"])

# ---- Data Directory Detection ------------------------------------------------
DATA_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train.csv" in files and "test.csv" in files and "train" in dirs:
        DATA_DIR = root
        break
assert DATA_DIR, "Error: Challenge dataset not found."
print(f"Data Directory: {DATA_DIR}")

def find_csv(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None

AUG_WEIGHTS = find_csv("aug_weights_exp16.csv")
CLIP_FEATS = find_csv("snu_clip_features.csv")
HOLDOUT = find_csv("holdout_300.csv")
print(f"aug_weights: {AUG_WEIGHTS}\nclip: {CLIP_FEATS}\nholdout: {HOLDOUT}")

# ---- Helper Functions --------------------------------------------------------
def chrono(ans):
    c = [0] * 4
    for i, p in enumerate(ans):
        c[p - 1] = i + 1
    return c

PAIR_COLS = {
    (1, 2): "dist_12",
    (1, 3): "dist_13",
    (1, 4): "dist_14",
    (2, 3): "dist_23",
    (2, 4): "dist_24",
    (3, 4): "dist_34",
}

def load_pairs(path):
    if not path:
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
        sc = sum(1 for a, b in pairs if pos[a - 1] > pos[b - 1]) * 10 + sum(
            abs(pos[i] - i) for i in range(4)
        )
        if sc > bs:
            best, bs = cand, sc
    return best

# ---- Dataset Construction ----------------------------------------------------
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
if HOLDOUT:
    hold = set(pd.read_csv(HOLDOUT)["Id"])
    train_df = train_df[~train_df["Id"].isin(hold)].reset_index(drop=True)
    print(f"Excluded holdout: {len(hold)} samples. Remaining: {len(train_df)}")

augw = {}
if AUG_WEIGHTS:
    w = pd.read_csv(AUG_WEIGHTS)
    augw = dict(zip(w["Id"], w["aug_mult"].astype(int)))
pairs = load_pairs(CLIP_FEATS)

items = []
for _, row in train_df.iterrows():
    mult = augw.get(row["Id"], CFG["aug_mult"])
    files = [row["Input_1"], row["Input_2"], row["Input_3"], row["Input_4"]]
    ans = ast.literal_eval(row["Answer"])
    ch = chrono(ans)
    tfiles = [files[n - 1] for n in ch]
    sp = pairs.get(row["Id"], [])
    seen = set()
    for v in range(mult):
        if v == 0:
            perm = list(range(4))
        else:
            perm = (
                hard_perm(seen, sp, files, tfiles) if CFG["hard_shuffle"] else None
            )
            if perm is None:
                perm = list(range(4))
                for _ in range(10):
                    rng.shuffle(perm)
                    if tuple(perm) not in seen:
                        break
        seen.add(tuple(perm))
        shown = [files[j] for j in perm]
        target = [shown.index(f) + 1 for f in tfiles]
        items.append(
            dict(
                id=row["Id"],
                sentence=row["Sentence"],
                paths=[os.path.join(DATA_DIR, "train", row["Id"], f) for f in shown],
                target=str(target),
            )
        )
rng.shuffle(items)
print(f"Total training items: {len(items)}")

# ---- Checkpoint Restore Logic ------------------------------------------------
def restore_ckpt_from_input():
    if os.path.exists(os.path.join(CFG["ckpt"], "adapter_model.safetensors")):
        return "working"

    adapter_paths = glob.glob("/kaggle/input/**/adapter", recursive=True)
    if adapter_paths:
        adapter_dir = adapter_paths[0]
        for f in os.listdir(adapter_dir):
            shutil.copy(os.path.join(adapter_dir, f), os.path.join(CFG["ckpt"], f))
        json.dump({"step": 977}, open(os.path.join(CFG["ckpt"], "meta.json"), "w"))
        print("Restored step 977 weights from input adapter directory.")
        return "input_adapter"

    for meta in glob.glob("/kaggle/input/**/meta.json", recursive=True):
        d = os.path.dirname(meta)
        if os.path.exists(os.path.join(d, "adapter_model.safetensors")):
            for f in os.listdir(d):
                shutil.copy(os.path.join(d, f), os.path.join(CFG["ckpt"], f))
            step = json.load(open(meta)).get("step", 0)
            print(f"Restored checkpoint from: {d} at step {step}")
            return "input_ckpt"
    return None

restore_ckpt_from_input()

# ---- Model Initialization (4bit Quantization & LoRA) ------------------------
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForImageTextToText.from_pretrained(
    CFG["model_id"],
    dtype=torch.float16,
    device_map="auto",
    quantization_config=quant,
)
proc = AutoProcessor.from_pretrained(CFG["model_id"], max_pixels=CFG["max_pixels"])
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.config.use_cache = False

resume = 0
meta = os.path.join(CFG["ckpt"], "meta.json")
if os.path.exists(os.path.join(CFG["ckpt"], "adapter_model.safetensors")):
    model = PeftModel.from_pretrained(model, CFG["ckpt"], is_trainable=True)
    resume = json.load(open(meta))["step"]
    print(f"Resumed from step: {resume}")
else:
    model = get_peft_model(
        model,
        LoraConfig(
            r=CFG["lora_r"],
            lora_alpha=CFG["lora_alpha"],
            lora_dropout=0.05,
            target_modules=CFG["lora_targets"].split(","),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
model.print_trainable_parameters()
model.train()

# ---- Optimizer & LR Scheduler ------------------------------------------------
total = (len(items)) // CFG["grad_accum"]
trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=CFG["lr"], weight_decay=0.01)
sched = get_cosine_schedule_with_warmup(
    opt, int(total * CFG["warmup_ratio"]), total
)
optpt = os.path.join(CFG["ckpt"], "optim.pt")
if resume and os.path.exists(optpt):
    st = torch.load(optpt, map_location="cpu")
    opt.load_state_dict(st["o"])
    sched.load_state_dict(st["s"])
dev = next(model.parameters()).device

def encode(it):
    content = []
    for i, p in enumerate(it["paths"]):
        content += [{"type": "image", "image": p}, {"type": "text", "text": f"\nImage {i+1}\n"}]
    content.append({"type": "text", "text": CFG["prompt_v5"].format(s=it["sentence"])})
    msgs = [{"role": "user", "content": content}]
    pt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    fm = msgs + [{"role": "assistant", "content": [{"type": "text", "text": it["target"]}]}]
    ft = proc.apply_chat_template(fm, tokenize=False)
    img, vid = process_vision_info(msgs)
    full = proc(text=[ft], images=img, videos=vid, padding=True, return_tensors="pt")
    pr = proc(text=[pt], images=img, videos=vid, padding=True, return_tensors="pt")
    lab = full.input_ids.clone()
    lab[:, : pr.input_ids.shape[1]] = -100
    full["labels"] = lab
    return full.to(dev)

def save_ckpt(step):
    model.save_pretrained(CFG["ckpt"])
    torch.save({"o": opt.state_dict(), "s": sched.state_dict()}, optpt)
    json.dump({"step": step}, open(meta, "w"))
    print(f"Checkpoint saved at step {step}", flush=True)

# ---- Training Loop -----------------------------------------------------------
t0 = time.time()
step = resume
micro = 0
lacc = 0.0
skip = 0
start = resume * CFG["grad_accum"]
pbar = tqdm(total=len(items), initial=start)

try:
    for idx, it in enumerate(items):
        if idx < start:
            continue
        try:
            loss = model(**encode(it)).loss / CFG["grad_accum"]
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            skip += 1
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        lacc += loss.item()
        micro += 1
        pbar.update(1)

        if micro % CFG["grad_accum"] == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % 10 == 0:
                pbar.set_postfix(loss=round(lacc, 4), step=step)
                lacc = 0.0

            if step % CFG["save_every"] == 0:
                save_ckpt(step)

            if time.time() - t0 > CFG["max_seconds"]:
                print("Time limit reached. Saving checkpoint and exiting.")
                save_ckpt(step)
                raise KeyboardInterrupt
except KeyboardInterrupt:
    pass

model.save_pretrained(CFG["out"])
save_ckpt(step)
pct = step / total * 100
print(f"\nTraining session ended at step {step}/{total} ({pct:.0f}%) | Time: {(time.time()-t0)/3600:.1f}h | OOM skips: {skip}")
