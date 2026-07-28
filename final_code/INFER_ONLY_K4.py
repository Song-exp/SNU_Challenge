# ================================================================================
# SNU AI Challenge — Qwen3-VL-8B Inference Pipeline (K4 Likelihood TTA)
# ================================================================================

import ast
import copy
import glob
from itertools import permutations
import os
import subprocess
import sys
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from transformers.cache_utils import DynamicCache
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# Initialize library installations
def pip(*p):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *p])

pip("transformers==5.13.0", "peft", "bitsandbytes", "accelerate", "qwen-vl-utils")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---- Configuration -----------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MAX_PIXELS = 384 * 512
USE_LIKELIHOOD = True
CHUNK = 6
SAVE_EVERY = 50
PROMPT_V5 = (
    "Look at the 4 images above labeled Image 1 to Image 4. Determine the "
    "correct chronological order of these images to match the sentence below.\n"
    'Sentence: "{s}"\nProvide the answer ONLY as a Python list of integers. '
    "Example: [1, 2, 3, 4]"
)
OUT_PATH = "/kaggle/working/submission.csv"

# ---- Data & Adapter Directory Detection --------------------------------------
DATA_DIR = None
for r, d, f in os.walk("/kaggle/input"):
    if "test.csv" in f and "test" in d:
        DATA_DIR = r
        break
assert DATA_DIR, "Error: Test dataset directory not found."
print(f"Data Directory: {DATA_DIR}")

ADAPTER = None
for p in glob.glob("/kaggle/input/**/adapter_model.safetensors", recursive=True):
    ADAPTER = os.path.dirname(p)
    break
if not ADAPTER:
    for p in glob.glob("/kaggle/working/**/adapter_model.safetensors", recursive=True):
        ADAPTER = os.path.dirname(p)
        break
assert ADAPTER, "Error: Adapter checkpoint not found."
print(f"Adapter Directory: {ADAPTER}")

# ---- Model Initialization ----------------------------------------------------
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
n_gpu = torch.cuda.device_count()
max_mem = {i: "14GiB" for i in range(n_gpu)}

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    device_map="auto",
    max_memory=max_mem,
    quantization_config=quant,
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
proc = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)
dev = next(model.parameters()).device

inner = model
for _ in range(5):
    if hasattr(inner, "rope_deltas"):
        break
    inner = getattr(inner, "model", None) or getattr(inner, "base_model", None)

test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
CANDS = [list(p) for p in permutations([1, 2, 3, 4])]
PERMS = (
    [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]
    if USE_LIKELIHOOD
    else [[0, 1, 2, 3]]
)
eos = proc.tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]

# ---- Helper Functions --------------------------------------------------------
def tgt_str(answer, perm):
    c = [0] * 4
    for i, pos in enumerate(answer):
        c[pos - 1] = i + 1
    files = [1, 2, 3, 4]
    tf = [files[n - 1] for n in c]
    shown = [files[j] for j in perm]
    return str([shown.index(f) + 1 for f in tf])

def parse_greedy(txt):
    s = txt.rfind("[")
    e = txt.find("]", s)
    try:
        r = ast.literal_eval(txt[s : e + 1])
        if isinstance(r, list) and sorted(r) == [1, 2, 3, 4]:
            sub = [0] * 4
            for i, n in enumerate(r):
                sub[n - 1] = i + 1
            return sub
    except Exception:
        pass
    return [1, 2, 3, 4]

def _clone_cache(cache):
    """
    Safely clones DynamicCache object to prevent state pollution.
    """
    try:
        return copy.deepcopy(cache)
    except Exception:
        new = DynamicCache()
        if hasattr(cache, "layers") and cache.layers:
            for i, lyr in enumerate(cache.layers):
                new.update(lyr.keys.clone(), lyr.values.clone(), i)
        else:
            for i, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
                new.update(k.clone(), v.clone(), i)
        return new

# ---- Sequence Scoring with OOM Defense ---------------------------------------
def score_perm(enc, plen, amat, L, ch):
    with torch.no_grad():
        o1 = model(**enc, use_cache=True)
        base = o1.past_key_values
        rd = inner.rope_deltas.item()
        flp = torch.log_softmax(o1.logits[:, -1, :].float(), -1)
        tots = []
        for cs in range(0, amat.shape[0], ch):
            amc = amat[cs : cs + ch]
            b = amc.shape[0]

            pkv = _clone_cache(base)
            pkv.batch_repeat_interleave(b)

            pos = (
                (torch.arange(plen, plen + L, device=dev) + rd)
                .view(1, 1, -1)
                .expand(3, b, -1)
                .contiguous()
            )
            o2 = model(
                input_ids=amc,
                position_ids=pos,
                past_key_values=pkv,
                attention_mask=torch.ones(
                    b, plen + L, device=dev, dtype=torch.long
                ),
                use_cache=True,
            )
            lp0 = flp[0, amc[:, 0]]
            rest = torch.log_softmax(o2.logits[:, :-1].float(), -1)
            tc = lp0 + rest[
                torch.arange(b)[:, None],
                torch.arange(L - 1)[None, :],
                amc[:, 1:],
            ].sum(1)
            tots.append(tc.detach().cpu())
            del pkv, o2, rest
        del o1, base, flp
    return torch.cat(tots)

# ---- Recovery Logic (Skip already processed records) --------------------------
recs = []
done = set()
if os.path.exists(OUT_PATH):
    try:
        prev = pd.read_csv(OUT_PATH)
        recs = prev.to_dict("records")
        done = set(prev["Id"].tolist())
        print(f"Resuming: {len(done)} records already completed.")
    except Exception:
        print("Failed to load existing submission file. Restarting from scratch.")

# ---- Inference Loop ----------------------------------------------------------
for ridx, (_, row) in enumerate(tqdm(test.iterrows(), total=len(test))):
    if row["Id"] in done:
        continue
    files = [row["Input_1"], row["Input_2"], row["Input_3"], row["Input_4"]]
    if USE_LIKELIHOOD:
        score = {tuple(a): 0.0 for a in CANDS}
        for perm in PERMS:
            shown = [files[j] for j in perm]
            content = []
            for i, f in enumerate(shown):
                content += [
                    {
                        "type": "image",
                        "image": os.path.join(DATA_DIR, "test", row["Id"], f),
                    },
                    {"type": "text", "text": f"\nImage {i+1}\n"},
                ]
            content.append(
                {"type": "text", "text": PROMPT_V5.format(s=row["Sentence"])}
            )
            m = [{"role": "user", "content": content}]
            pt = proc.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            ii, vi = process_vision_info(m)
            enc = proc(
                text=[pt], images=ii, videos=vi, return_tensors="pt"
            ).to(dev)
            plen = enc["input_ids"].shape[1]
            atok = [
                proc.tokenizer(tgt_str(a, perm), add_special_tokens=False)[
                    "input_ids"
                ]
                + eos
                for a in CANDS
            ]
            L = len(atok[0])
            amat = torch.tensor(atok, device=dev)

            ch = CHUNK
            while True:
                try:
                    tot = score_perm(enc, plen, amat, L, ch)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if ch <= 2:
                        raise
                    ch = max(2, ch // 2)
                    print(f"OOM detected. Reducing chunk size to {ch} and retrying.")
            for a, s in zip(CANDS, tot.tolist()):
                score[tuple(a)] += s
        best = max(CANDS, key=lambda a: score[tuple(a)])
        sub = best
    else:
        content = []
        for i, f in enumerate(files):
            content += [
                {
                    "type": "image",
                    "image": os.path.join(DATA_DIR, "test", row["Id"], f),
                },
                {"type": "text", "text": f"\nImage {i+1}\n"},
            ]
        content.append(
            {"type": "text", "text": PROMPT_V5.format(s=row["Sentence"])}
        )
        m = [{"role": "user", "content": content}]
        pt = proc.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        ii, vi = process_vision_info(m)
        enc = proc(text=[pt], images=ii, videos=vi, return_tensors="pt").to(
            dev
        )
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=32, do_sample=False)
        txt = proc.batch_decode(
            out[:, enc.input_ids.shape[1] :], skip_special_tokens=True
        )[0]
        sub = parse_greedy(txt)

    recs.append({"Id": row["Id"], "Answer": str(sub)})
    if len(recs) % SAVE_EVERY == 0:
        pd.DataFrame(recs).to_csv(OUT_PATH, index=False)

out = pd.DataFrame(recs)
out.to_csv(OUT_PATH, index=False)
print("Inference completed successfully and submission.csv saved.")
