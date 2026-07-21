# =====================================================================
# SNU AI Challenge — Kaggle 8B 추론·제출 생성 (학습 노트북 완주 후 실행)
# =====================================================================
# 학습 노트북이 만든 어댑터(/kaggle/working/exp_8b/adapter 또는 Dataset 업로드본)로
# test 819개를 추론해 submission.csv 생성. 우도 스코어링(선택)까지 포함.
#
# 규정: 추론도 8B라도 3090(24GB) 채점환경에서 24h 내 완료 (test 819 → 수십 분).
# =====================================================================
import os, ast, re, time
import pandas as pd, torch
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

CONFIG = {
    "data_dir": "/kaggle/input/snu-ai-challenge-data/snuaichallenge_data",
    "model_id": "Qwen/Qwen3-VL-8B-Instruct",
    "adapter": "/kaggle/working/exp_8b/adapter",   # 학습 산출물 (또는 Dataset 경로)
    "prompt": "v5_reorder",
    "max_pixels": 512 * 384,
    "use_likelihood": True,     # True=24후보 우도채점(권장, +정확도), False=greedy 생성
    "out": "/kaggle/working/submission.csv",
}

V5 = ("Look at the 4 images above labeled Image 1 to Image 4. Determine the correct "
      "chronological order of these images to match the sentence below.\n"
      'Sentence: "{sentence}"\n'
      "Provide the answer ONLY as a Python list of integers. Example: [1, 2, 3, 4]")

def msg(row, image_dir, files):
    content = []
    for i, f in enumerate(files):
        content.append({"type": "image", "image": os.path.join(image_dir, row["Id"], f)})
        content.append({"type": "text", "text": f"\nImage {i+1}\n"})
    content.append({"type": "text", "text": V5.format(sentence=row["Sentence"])})
    return [{"role": "user", "content": content}]

def parse(txt):
    s = txt.rfind("["); e = txt.find("]", s)
    try:
        r = ast.literal_eval(txt[s:e+1])
        if isinstance(r, list) and sorted(r) == [1, 2, 3, 4]:
            sub = [0]*4
            for i, n in enumerate(r):
                sub[n-1] = i+1
            return sub, True
    except Exception:
        pass
    return [1, 2, 3, 4], False

# ---- 우도 스코어링용 좌표 변환 (test_perm_coords 검증본) ----
from itertools import permutations
CANDS = [list(p) for p in permutations([1, 2, 3, 4])]
def target_string(answer):   # perm=identity (제시=원본)
    c = [0]*4
    for i, pos in enumerate(answer): c[pos-1] = i+1
    files = [1,2,3,4]; time_files = [files[n-1] for n in c]
    return str([files.index(f)+1 for f in time_files])

def main():
    test = pd.read_csv(os.path.join(CONFIG["data_dir"], "test.csv"))
    image_dir = os.path.join(CONFIG["data_dir"], "test")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForImageTextToText.from_pretrained(
        CONFIG["model_id"], dtype=torch.float16, device_map="auto", quantization_config=quant)
    model = PeftModel.from_pretrained(model, CONFIG["adapter"]); model.eval()
    proc = AutoProcessor.from_pretrained(CONFIG["model_id"], max_pixels=CONFIG["max_pixels"])
    inner = model
    for _ in range(4):
        if hasattr(inner, "rope_deltas"): break
        inner = getattr(inner, "model", None) or getattr(inner, "base_model", None)
    dev = next(model.parameters()).device
    eos = proc.tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]

    recs = []
    for _, row in tqdm(test.iterrows(), total=len(test)):
        files = [row["Input_1"], row["Input_2"], row["Input_3"], row["Input_4"]]
        m = msg(row, image_dir, files)
        if CONFIG["use_likelihood"]:
            pt = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            ii, vi = process_vision_info(m)
            enc = proc(text=[pt], images=ii, videos=vi, return_tensors="pt").to(dev)
            plen = enc["input_ids"].shape[1]
            atok = [proc.tokenizer(target_string(a), add_special_tokens=False)["input_ids"]+eos
                    for a in CANDS]
            L = len(atok[0]); amat = torch.tensor(atok, device=dev)
            with torch.no_grad():
                o1 = model(**enc, use_cache=True); pkv = o1.past_key_values
                rd = inner.rope_deltas.item()
                flp = torch.log_softmax(o1.logits[:, -1, :].float(), -1)
                pos = (torch.arange(plen, plen+L, device=dev)+rd).view(1,1,-1).expand(3,24,-1).contiguous()
                pkv.batch_repeat_interleave(24)
                o2 = model(input_ids=amat, position_ids=pos, past_key_values=pkv,
                           attention_mask=torch.ones(24, plen+L, device=dev, dtype=torch.long), use_cache=True)
                lp0 = flp[0, amat[:, 0]]
                rest = torch.log_softmax(o2.logits[:, :-1].float(), -1)
                lpr = rest[torch.arange(24)[:,None], torch.arange(L-1)[None,:], amat[:,1:]].sum(1)
                best = CANDS[(lp0+lpr).argmax()]
            sub = [0]*4
            for i, n in enumerate(best): sub[n-1] = i+1
            recs.append({"Id": row["Id"], "Answer": str(sub)})
        else:
            text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            ii, vi = process_vision_info(m)
            enc = proc(text=[text], images=ii, videos=vi, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=32, do_sample=False)
            txt = proc.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)[0]
            sub, _ = parse(txt)
            recs.append({"Id": row["Id"], "Answer": str(sub)})

    df = pd.DataFrame(recs)
    sample = pd.read_csv(os.path.join(CONFIG["data_dir"], "sample_submission.csv"))
    assert list(df.Id) == list(sample.Id), "Id 순서 불일치"
    df.to_csv(CONFIG["out"], index=False)
    print(f"저장: {CONFIG['out']} ({len(df)}행)")


if __name__ == "__main__":
    main()
