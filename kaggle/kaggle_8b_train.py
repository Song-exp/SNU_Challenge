# =====================================================================
# SNU AI Challenge — Kaggle 8B QLoRA 학습 노트북 (자립형)
# =====================================================================
# 목적: 로컬 8GB에서 불가한 Qwen3-VL-8B를 Kaggle GPU(P100/T4x2 16GB)에서 파인튜닝.
# 레시피 = 로컬 최고 exp17(Public 0.857) + 검증된 재료 최대 투입:
#   - 모델: Qwen3-VL-8B-Instruct (4bit QLoRA)  ← 4B 대비 체급 도약
#   - 프롬프트: v5_reorder (문장 후치, Public +0.9%p 실측)
#   - 증강: 타깃 가중 (sparse_camX x4 / 나머지 x2) + aug 최대
#   - 어려운 셔플: CLIP 유사쌍 순서 뒤집기 (쌍교환 오답 공략, exp20 트랙)
# 배제(실측 기각): CoT(4연패), scene_cuts/OWL-ViT 힌트 주입(v10무효),
#                  gemma 힌트(미니 2연패), 역할부여 프롬프트(노이즈)
#
# ⚠️ 12시간 세션 대응: 체크포인트 저장 + 재개 내장. 세션 끊기면 재실행하면 이어감.
#    출력은 /kaggle/working/adapter 에 저장 → 노트북 Output에서 다운로드.
# =====================================================================

# ---- [셀 1] 설치 ----------------------------------------------------
# !pip install -q -U transformers==5.13.0 peft bitsandbytes accelerate qwen-vl-utils

import os, ast, json, math, random, time
from datetime import datetime

# ---- [셀 2] 설정 ----------------------------------------------------
CONFIG = {
    # 데이터 경로 (Kaggle Dataset — 실제 경로에 맞게 확인/수정)
    "data_dir": "/kaggle/input/snu-ai-challenge-data/snuaichallenge_data",
    "aug_weights": "/kaggle/input/snu-ai-aux/aug_weights_exp16.csv",   # 업로드 필요
    "clip_features": "/kaggle/input/snu-ai-aux/snu_clip_features.csv",  # 어려운셔플용, 업로드 필요
    "holdout": "/kaggle/input/snu-ai-aux/holdout_300.csv",             # 학습 제외용, 업로드 필요
    "model_id": "Qwen/Qwen3-VL-8B-Instruct",   # 규정 공개일 2025-10 OK. 인터넷 켜고 다운 or 데이터셋 첨부

    # 학습 하이퍼 (exp17 레시피 계승)
    "prompt": "v5_reorder",
    "aug_mult": 2,                    # 기본 증강 (가중 CSV가 sparse_camX만 4로 올림)
    "hard_shuffle": True,            # CLIP 유사쌍 순서 뒤집기
    "lr": 1e-4,
    "epochs": 1,
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "lora_targets": "q_proj,k_proj,v_proj,o_proj",
    "grad_accum": 16,
    "max_pixels": 512 * 384,         # 16GB 여유로 로컬(307200)보다 상향 → 시각 정보 ↑
    "warmup_ratio": 0.03,
    "seed": 42,

    # 체크포인트 (12시간 세션 대응)
    "out_dir": "/kaggle/working/exp_8b",
    "ckpt_dir": "/kaggle/working/exp_8b/ckpt",
    "save_every_steps": 100,         # 어댑터+옵티마이저 저장 주기
    "max_seconds": 11.3 * 3600,      # 11.3h 지나면 안전 저장 후 종료 (12h 컷 전 여유)
}

os.makedirs(CONFIG["out_dir"], exist_ok=True)
os.makedirs(CONFIG["ckpt_dir"], exist_ok=True)

# ---- [셀 3] 프롬프트 (prompts.py에서 v5_reorder만 이식) ---------------
PROMPTS = {
    "v5_reorder": (
        "Look at the 4 images above labeled Image 1 to Image 4. Determine the correct "
        "chronological order of these images to match the sentence below.\n"
        'Sentence: "{sentence}"\n'
        "Provide the answer ONLY as a Python list of integers. Example: [1, 2, 3, 4]"
    ),
}

def build_user_text(sentence):
    return PROMPTS[CONFIG["prompt"]].format(sentence=sentence)

def build_messages(sentence, image_paths):
    content = []
    for i, p in enumerate(image_paths):
        content.append({"type": "image", "image": p})
        content.append({"type": "text", "text": f"\nImage {i + 1}\n"})
    content.append({"type": "text", "text": build_user_text(sentence)})
    return [{"role": "user", "content": content}]

# ---- [셀 4] 데이터 구성 (train.py 로직 이식) --------------------------
def chrono_image_numbers(answer):
    c = [0] * 4
    for i, pos in enumerate(answer):
        c[pos - 1] = i + 1
    return c

# CLIP 유사쌍 로드 (어려운 셔플용) — dist_ij < 0.20 인 쌍
PAIR_COLS = {(1, 2): "dist_12", (1, 3): "dist_13", (1, 4): "dist_14",
             (2, 3): "dist_23", (2, 4): "dist_24", (3, 4): "dist_34"}

def load_clip_pairs(path, thr=0.20):
    import pandas as pd
    if not os.path.exists(path):
        print("CLIP 피처 없음 — 어려운 셔플 비활성 (무작위 셔플로 진행)")
        return {}
    df = pd.read_csv(path)
    out = {}
    for r in df.itertuples():
        out[r.Id] = [p for p, c in PAIR_COLS.items() if getattr(r, c) < thr]
    return out

def hard_perm(rng, seen, sim_pairs, files, time_files, n_try=16):
    best, best_score = None, -1
    for _ in range(n_try):
        cand = list(range(4)); rng.shuffle(cand)
        if tuple(cand) in seen:
            continue
        if [files[j] for j in cand] == time_files:      # 타깃 identity 배제
            continue
        pos = {orig: slot for slot, orig in enumerate(cand)}
        score = sum(1 for a, b in sim_pairs if pos[a - 1] > pos[b - 1]) * 10
        score += sum(abs(pos[i] - i) for i in range(4))
        if score > best_score:
            best, best_score = cand, score
    return best

def build_items(df, image_dir, aug_weights, clip_pairs, rng):
    items = []
    for _, row in df.iterrows():
        mult = aug_weights.get(row["Id"], CONFIG["aug_mult"])
        files = [row["Input_1"], row["Input_2"], row["Input_3"], row["Input_4"]]
        answer = ast.literal_eval(row["Answer"])
        chrono = chrono_image_numbers(answer)
        time_files = [files[n - 1] for n in chrono]
        sim_pairs = clip_pairs.get(row["Id"], [])
        seen = set()
        for v in range(mult):
            if v == 0:
                perm = list(range(4))
            else:
                perm = None
                if CONFIG["hard_shuffle"]:
                    perm = hard_perm(rng, seen, sim_pairs, files, time_files)
                if perm is None:
                    perm = list(range(4))
                    for _ in range(10):
                        rng.shuffle(perm)
                        if tuple(perm) not in seen:
                            break
            seen.add(tuple(perm))
            shown = [files[j] for j in perm]
            target = [shown.index(f) + 1 for f in time_files]
            items.append({
                "id": row["Id"], "sentence": row["Sentence"],
                "paths": [os.path.join(image_dir, row["Id"], f) for f in shown],
                "target_text": str(target),
            })
    return items

# ---- [셀 5] 메인 학습 (체크포인트 재개 내장) --------------------------
def main():
    import pandas as pd, torch
    from tqdm.auto import tqdm
    from transformers import (AutoModelForImageTextToText, AutoProcessor,
                              BitsAndBytesConfig)
    from transformers.optimization import get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from qwen_vl_utils import process_vision_info

    random.seed(CONFIG["seed"]); torch.manual_seed(CONFIG["seed"])
    rng = random.Random(CONFIG["seed"])

    # 데이터
    train_df = pd.read_csv(os.path.join(CONFIG["data_dir"], "train.csv"))
    if os.path.exists(CONFIG["holdout"]):
        hold = set(pd.read_csv(CONFIG["holdout"])["Id"])
        train_df = train_df[~train_df["Id"].isin(hold)].reset_index(drop=True)
        print(f"holdout {len(hold)}개 제외 → train {len(train_df)}")
    aug_weights = {}
    if os.path.exists(CONFIG["aug_weights"]):
        w = pd.read_csv(CONFIG["aug_weights"])
        aug_weights = dict(zip(w["Id"], w["aug_mult"].astype(int)))
    clip_pairs = load_clip_pairs(CONFIG["clip_features"])

    items = build_items(train_df, os.path.join(CONFIG["data_dir"], "train"),
                        aug_weights, clip_pairs, rng)
    rng.shuffle(items)
    print(f"학습 항목 {len(items)}개")

    # 모델 (4bit QLoRA)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(
        CONFIG["model_id"], dtype=torch.bfloat16, device_map="auto",
        quantization_config=quant)
    processor = AutoProcessor.from_pretrained(CONFIG["model_id"], max_pixels=CONFIG["max_pixels"])
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    adapter_dir = os.path.join(CONFIG["out_dir"], "adapter")
    resume_step = 0
    ckpt_meta = os.path.join(CONFIG["ckpt_dir"], "meta.json")
    if os.path.exists(os.path.join(CONFIG["ckpt_dir"], "adapter_model.safetensors")):
        # 체크포인트 재개
        model = PeftModel.from_pretrained(model, CONFIG["ckpt_dir"], is_trainable=True)
        resume_step = json.load(open(ckpt_meta))["opt_step"]
        print(f"⏩ 체크포인트 재개: step {resume_step}")
    else:
        lora = LoraConfig(r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"],
                          lora_dropout=CONFIG["lora_dropout"],
                          target_modules=CONFIG["lora_targets"].split(","),
                          bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
    model.print_trainable_parameters(); model.train()

    total_opt = (len(items) * CONFIG["epochs"]) // CONFIG["grad_accum"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=CONFIG["lr"], weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_opt * CONFIG["warmup_ratio"]), total_opt)
    # 옵티마이저/스케줄러 상태 재개
    opt_pt = os.path.join(CONFIG["ckpt_dir"], "optim.pt")
    if resume_step and os.path.exists(opt_pt):
        st = torch.load(opt_pt, map_location="cpu")
        optimizer.load_state_dict(st["optim"]); scheduler.load_state_dict(st["sched"])
        print("옵티마이저 상태 재개됨")

    dev = next(model.parameters()).device

    def encode(item):
        messages = build_messages(item["sentence"], item["paths"])
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_msgs = messages + [{"role": "assistant",
                                 "content": [{"type": "text", "text": item["target_text"]}]}]
        full_text = processor.apply_chat_template(full_msgs, tokenize=False)
        img, vid = process_vision_info(messages)
        full = processor(text=[full_text], images=img, videos=vid, padding=True, return_tensors="pt")
        prompt = processor(text=[prompt_text], images=img, videos=vid, padding=True, return_tensors="pt")
        labels = full.input_ids.clone()
        labels[:, :prompt.input_ids.shape[1]] = -100
        full["labels"] = labels
        return full.to(dev)

    def save_ckpt(step):
        model.save_pretrained(CONFIG["ckpt_dir"])
        torch.save({"optim": optimizer.state_dict(), "sched": scheduler.state_dict()}, opt_pt)
        json.dump({"opt_step": step}, open(ckpt_meta, "w"))
        print(f"💾 체크포인트 저장 step {step}", flush=True)

    t0 = time.time()
    opt_step, micro, loss_acc, n_skip = resume_step, 0, 0.0, 0
    start_item = resume_step * CONFIG["grad_accum"]      # 재개 지점까지 스킵
    pbar = tqdm(items, initial=start_item, total=len(items))
    try:
        for idx, item in enumerate(items):
            if idx < start_item:
                continue
            try:
                loss = model(**encode(item)).loss / CONFIG["grad_accum"]
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                n_skip += 1; optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache(); continue
            loss_acc += loss.item(); micro += 1; pbar.update(1)
            if micro % CONFIG["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % 10 == 0:
                    pbar.set_postfix(loss=round(loss_acc, 4), step=opt_step)
                loss_acc = 0.0
                if opt_step % CONFIG["save_every_steps"] == 0:
                    save_ckpt(opt_step)
                if time.time() - t0 > CONFIG["max_seconds"]:
                    print("⏰ 세션 시간 한도 — 안전 저장 후 종료 (재실행하면 이어감)")
                    save_ckpt(opt_step); raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    model.save_pretrained(adapter_dir)
    save_ckpt(opt_step)
    print(f"\n종료: {opt_step}/{total_opt} 스텝, {(time.time()-t0)/3600:.1f}h, OOM 스킵 {n_skip}")
    print(f"어댑터: {adapter_dir}  (완주 시 이걸 다운로드 → 로컬 제출 생성)")
    print(f"진행률 {opt_step/total_opt*100:.0f}% — 100% 아니면 노트북 재실행으로 이어서 학습")


if __name__ == "__main__":
    main()
