# -*- coding: utf-8 -*-
"""Qwen3-VL QLoRA 파인튜닝 스크립트 — 프레임 순서 예측 태스크.

핵심 설계:
- 프롬프트/출력 형식은 eval_zero_shot.py와 100% 동일 (개선분 = 순수 학습 효과로 해석 가능)
- 재셔플 증강: 샘플당 --aug-mult개 제시 순서 변형 (변형 0 = 원본 제시 순서)
- splits/holdout_300.csv + eda/stratified_valid.csv 의 Id는 학습에서 제외 (평가 오염 방지)
- 8GB VRAM 대응: LoRA + gradient checkpointing + batch 1 + accumulation, bf16
- 밤샘 안전장치: --max-hours 초과 시 저장 후 종료, --save-steps 주기 저장, 절전 차단

사용 예:
    # 스모크 테스트 (소량, 사이클 검증용)
    python scripts/train.py --run-name smoke --max-samples 30 --max-steps 20

    # 밤 배치 (10시간 상한)
    python scripts/train.py --run-name qwen3vl2b_aug2_lr1e4 --aug-mult 2 --max-hours 10

    # 4B 스케일업 (4bit 필수)
    python scripts/train.py --model ./models/Qwen3-VL-4B-Instruct --load-4bit --run-name qwen3vl4b_aug2
"""
import argparse
import ast
import ctypes
import json
import os

# VRAM 위생 (7/19 OOM + 7/20 스필오버 크롤 대응):
# - expandable_segments는 Windows 미지원(로드 시 경고, 무시됨)이라 제거
# - garbage_collection_threshold: 예약(reserve)을 키우기 전에 캐시 블록을 회수 —
#   예약이 공유(시스템 RAM) GPU 메모리까지 부풀어 PCIe 스래싱(클럭 352MHz)을 내던 사인
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "garbage_collection_threshold:0.8,max_split_size_mb:256")
import random
import time
from datetime import datetime

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # 프로젝트 루트 기준


# ---------------------------------------------------------------- 프롬프트 (prompts.py 레지스트리 공유)

import prompts as prompt_registry


def build_messages(sentence, image_paths, prompt_name="v1_list", hint=""):
    content = []
    for i, p in enumerate(image_paths):
        content.append({"type": "image", "image": p})
        content.append({"type": "text", "text": f"\nImage {i + 1}\n"})
    content.append({"type": "text", "text": prompt_registry.build_user_text(prompt_name, sentence, hint)})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------- 데이터 구성

def chrono_image_numbers(answer):
    """제출 형식 Answer([i]=이미지 i+1의 시간상 위치) -> 시간순 이미지 번호 리스트."""
    c = [0] * 4
    for i, pos in enumerate(answer):
        c[pos - 1] = i + 1
    return c


def _hard_perm_candidates(rng, seen, sim_pairs, files, time_files, n_try=16):
    """CLIP 유사쌍의 제시 순서를 변형마다 뒤집는 순열을 우선 고른다 (--hard-shuffle).

    근거(7/20 오답 해부): 전 유형의 주 오답 모드가 '쌍교환'(비슷한 두 프레임의 선후 혼동).
    유사쌍이 변형마다 다른 순서로 제시되면, 제시 위치를 외워서는 맞힐 수 없고
    프레임 내용으로만 판별해야 한다 -> 그 판별 능력에 그래디언트가 집중된다.

    ⚠️ identity 배제는 '순열'이 아니라 '타깃' 기준이어야 한다 — 타깃 [1,2,3,4]는
    제시 순서가 시간순과 일치할 때 나오고, 이는 원본 정답에 따라 달라진다
    (순열만 보고 걸렀다가 identity 타깃이 그대로 남는 버그를 7/21 실측으로 확인).
    """
    best, best_score = None, -1
    for _ in range(n_try):
        cand = list(range(4))
        rng.shuffle(cand)
        if tuple(cand) in seen:
            continue
        shown = [files[j] for j in cand]
        if shown == time_files:                  # 타깃이 [1,2,3,4] = identity 지름길 강화
            continue
        pos = {orig: slot for slot, orig in enumerate(cand)}   # 원본 슬롯 -> 제시 슬롯
        # 유사쌍이 제시 순서에서 뒤집힌 개수 + 전체 변위 (동점 시 더 많이 섞인 쪽)
        score = sum(1 for a, b in sim_pairs if pos[a - 1] > pos[b - 1])
        score = score * 10 + sum(abs(pos[i] - i) for i in range(4))
        if score > best_score:
            best, best_score = cand, score
    return best


def build_training_items(df, image_dir, aug_mult, rng, aug_weights=None, clip_pairs=None,
                         loss_weights=None, hard_shuffle=False, inject_hint=True,
                         owlvit_frames=None, scene_cuts=None):
    """각 샘플을 aug_mult개의 (제시 순서 변형, 시간순 라벨) 학습 항목으로 확장한다.

    aug_weights: {Id: 배수} — 지정된 Id는 해당 배수로, 나머지는 aug_mult로 (타깃 증강용).
    clip_pairs: {Id: [(a,b),...]} — CLIP 유사쌍 (원본 제시 순서 기준 1-based).
                힌트 프롬프트용이면 변형마다 재매핑되어 item["hint"]에 저장되고,
                hard_shuffle이면 순열 선택 기준으로도 쓰인다.
    loss_weights: {Id: 가중} — item["loss_weight"]로 실려 학습 루프에서 손실에 곱해진다.
    hard_shuffle: 무작위 순열 대신 유사쌍을 뒤집는 순열 우선 (_hard_perm_candidates).
    """
    from structure_features import (hint_text, remap_pairs, build_owlvit_hint_text,
                                     build_scene_cut_hint_text)
    items = []
    for _, row in df.iterrows():
        mult = aug_weights.get(row["Id"], aug_mult) if aug_weights else aug_mult
        files = [row["Input_1"], row["Input_2"], row["Input_3"], row["Input_4"]]
        answer = ast.literal_eval(row["Answer"])
        chrono = chrono_image_numbers(answer)             # 시간순 이미지 번호 (원본 제시 기준)
        time_files = [files[n - 1] for n in chrono]       # 시간순 파일 목록 (변형 불변)
        sim_pairs = (clip_pairs or {}).get(row["Id"], [])

        seen = set()
        for v in range(mult):
            if v == 0:
                perm = list(range(4))                      # 변형 0 = 원본 제시 순서
            else:
                perm = None
                if hard_shuffle:
                    perm = _hard_perm_candidates(rng, seen, sim_pairs, files, time_files)
                if perm is None:
                    perm = list(range(4))
                    for _ in range(10):                    # 이미 만든 변형과 중복 회피 (최선 노력)
                        rng.shuffle(perm)
                        if tuple(perm) not in seen:
                            break
            seen.add(tuple(perm))

            shown_files = [files[j] for j in perm]         # 이번 변형에서 Image 1~4로 제시되는 파일
            target = [shown_files.index(f) + 1 for f in time_files]  # 시간순 -> 제시 라벨
            hint = ""
            if inject_hint and scene_cuts is not None:      # scene_cuts 힌트 (전체 속성, 불변)
                hint = build_scene_cut_hint_text(scene_cuts.get(row["Id"]))
            elif inject_hint and owlvit_frames is not None:  # OWL-ViT 좌표 힌트 (제시 순서 재매핑)
                hint = build_owlvit_hint_text(owlvit_frames.get(row["Id"]), perm)
            elif inject_hint and clip_pairs is not None:    # CLIP 유사쌍 힌트
                hint = hint_text(remap_pairs(clip_pairs.get(row["Id"], []), perm))
            items.append({
                "id": row["Id"],
                "sentence": row["Sentence"],
                "paths": [os.path.join(image_dir, row["Id"], f) for f in shown_files],
                "target_text": str(target),                # 예: "[3, 1, 4, 2]" (eval 파서와 동일 형식)
                "hint": hint,
                "loss_weight": (loss_weights or {}).get(row["Id"], 1.0),
            })
    return items


def load_excluded_ids():
    import pandas as pd
    excluded = set()
    for path in ["./splits/holdout_300.csv", "./eda/stratified_valid.csv"]:
        if os.path.exists(path):
            excluded |= set(pd.read_csv(path)["Id"])
            print(f"학습 제외: {path} ({len(excluded)}개 누적)", flush=True)
    return excluded


# ---------------------------------------------------------------- 유틸

def keep_system_awake():
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def wait_for_free_vram(required_gb, timeout_hours=8.0):
    import torch
    deadline = time.time() + timeout_hours * 3600
    while time.time() < deadline:
        free, _ = torch.cuda.mem_get_info()
        if free / 1e9 >= required_gb:
            return
        print(f"VRAM 대기: 여유 {free / 1e9:.1f}GB < 필요 {required_gb:.1f}GB", flush=True)
        time.sleep(60)
    raise RuntimeError("VRAM 확보 대기 시간 초과")


# ---------------------------------------------------------------- 메인

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--load-4bit", action="store_true", help="QLoRA (4B 이상은 필수)")
    parser.add_argument("--run-name", required=True, help="출력 폴더명 (outputs/runs/<run-name>)")
    parser.add_argument("--aug-mult", type=int, default=2, help="샘플당 제시 순서 변형 수 (원본 포함)")
    parser.add_argument("--aug-weights", default="",
                        help="Id별 증강 배수 CSV (열: Id,aug_mult) — 명시된 Id는 그 배수, 나머지는 --aug-mult")
    parser.add_argument("--loss-weights", default="",
                        help="Id별 손실 가중 CSV (열: Id,loss_weight) — 증강 복제 없이 그래디언트 기여만 조절")
    parser.add_argument("--hard-shuffle", action="store_true",
                        help="증강 순열을 CLIP 유사쌍 기준으로 선택 (유사쌍 순서를 뒤집는 변형 우선, identity 제외)")
    parser.add_argument("--owlvit-hints", default="",
                        help="OWL-ViT 좌표 힌트 jsonl 경로 (v9류 프롬프트와 함께 — 없으면 CLIP 유사쌍 힌트)")
    parser.add_argument("--scene-cut-hints", action="store_true",
                        help="scene_cuts 힌트 주입 (v10류 프롬프트와 함께 — 커버리지 100 퍼센트, 최우선 소스)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--prompt", default="v1_list", choices=list(prompt_registry.PROMPTS),
                        help="프롬프트 이름 (평가 시에도 같은 이름 필수)")
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=640 * 480, help="eval과 동일 기본값")
    parser.add_argument("--mem-fraction", type=float, default=0.85,
                        help="GPU 메모리 할당자 캡 (4B=0.85, 8B는 0.95+ 필요 — 스필 위험 감수)")
    parser.add_argument("--skip-kbit-upcast", action="store_true",
                        help="8B용: kbit 준비의 fp32 업캐스트(+2.3GB) 생략 (VRAM 절약, 안정성 스모크 확인)")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="기반 샘플 수 제한 (0=전체, 스모크용)")
    parser.add_argument("--max-steps", type=int, default=0, help="옵티마이저 스텝 상한 (0=제한 없음)")
    parser.add_argument("--max-hours", type=float, default=0, help="시간 상한, 초과 시 저장 후 종료 (0=없음)")
    parser.add_argument("--save-steps", type=int, default=100, help="어댑터 주기 저장 (옵티마이저 스텝 단위)")
    parser.add_argument("--snapshot-steps", type=int, default=0,
                        help="N스텝마다 checkpoints/step_N/ 에 별도 스냅샷 저장 (학습 곡선용, 0=끔)")
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--empty-cache-steps", type=int, default=50,
                        help="N 옵티마이저 스텝마다 CUDA 캐시 반환 (단편화 완화, 0=끔)")
    parser.add_argument("--data-dir", default="./snuaichallenge_data/")
    args = parser.parse_args()

    import pandas as pd
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.optimization import get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info

    assert torch.cuda.is_available(), "GPU 필요"
    # 할당자 하드 캡: 전용 VRAM의 N%까지만 (8.55GB x 0.85 ~= 7.3GB = 4B 실사용 한계).
    # 이 상한을 넘는 요구는 드라이버가 공유 메모리로 스필(→ 전체 크롤)하는 대신
    # OOM 예외가 되고, 학습 루프의 OOM 스킵이 해당 샘플만 건너뛴다 (7/20 진단).
    # 8B는 로드+kbit 준비에 여유가 필요 → --mem-fraction 으로 상향 (스필 위험 감수).
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    keep_system_awake()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    out_dir = os.path.join("./outputs/runs", args.run_name)
    adapter_dir = os.path.join(out_dir, "adapter")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.csv")

    # ---- 데이터: holdout/stratified 제외 -> 증강 항목 생성 -----------------------------------
    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    excluded = load_excluded_ids()
    train_df = train_df[~train_df["Id"].isin(excluded)].reset_index(drop=True)
    if args.max_samples:
        train_df = train_df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
    image_dir = os.path.join(args.data_dir, "train")
    aug_weights = None
    if args.aug_weights:
        wdf = pd.read_csv(args.aug_weights)
        aug_weights = dict(zip(wdf["Id"], wdf["aug_mult"].astype(int)))
        counts = wdf["aug_mult"].value_counts().sort_index()
        dist = ", ".join(f"x{m}: {n}개" for m, n in counts.items())
        print(f"가변 증강: {args.aug_weights} ({len(aug_weights)}개 Id | {dist} | 미지정은 x{args.aug_mult})", flush=True)
    clip_pairs = None
    owlvit_frames = None
    scene_cuts = None
    if prompt_registry.needs_hint(args.prompt):
        if args.scene_cut_hints:
            from structure_features import load_scene_cuts
            scene_cuts = load_scene_cuts()
            print(f"힌트 주입: scene_cuts {len(scene_cuts)}개 Id (커버리지 100%, 프롬프트 {args.prompt})",
                  flush=True)
        elif args.owlvit_hints:
            from structure_features import load_owlvit_frames
            owlvit_frames = load_owlvit_frames(args.owlvit_hints)
            n_ok = sum(1 for v in owlvit_frames.values()
                       if sum(f.get("status") == "ok" for f in v["frames"]) >= 2)
            print(f"힌트 주입: OWL-ViT 좌표 {len(owlvit_frames)}개 Id "
                  f"(유효 힌트 {n_ok}개, 프롬프트 {args.prompt})", flush=True)
        else:
            from structure_features import load_clip_pairs
            clip_pairs = load_clip_pairs()
            print(f"힌트 주입: CLIP 유사쌍 {len(clip_pairs)}개 Id (프롬프트 {args.prompt})", flush=True)
    elif args.hard_shuffle:
        from structure_features import load_clip_pairs
        clip_pairs = load_clip_pairs()          # 힌트 주입 없이 순열 선택에만 사용
        n_with = sum(1 for v in clip_pairs.values() if v)
        print(f"어려운 셔플: CLIP 유사쌍 보유 {n_with}/{len(clip_pairs)}개 Id "
              f"(유사쌍 없는 샘플은 기존 무작위 셔플)", flush=True)

    loss_weights = None
    if args.loss_weights:
        ldf = pd.read_csv(args.loss_weights)
        loss_weights = dict(zip(ldf["Id"], ldf["loss_weight"].astype(float)))
        counts = ldf["loss_weight"].round(2).value_counts().sort_index()
        dist = ", ".join(f"w{w}: {n}개" for w, n in counts.items())
        print(f"손실 가중: {args.loss_weights} ({len(loss_weights)}개 Id | {dist} | 미지정은 w1.0)",
              flush=True)

    items = build_training_items(train_df, image_dir, args.aug_mult, rng, aug_weights, clip_pairs,
                                 loss_weights, args.hard_shuffle,
                                 inject_hint=prompt_registry.needs_hint(args.prompt),
                                 owlvit_frames=owlvit_frames, scene_cuts=scene_cuts)
    rng.shuffle(items)
    print(f"기반 {len(train_df)}개 -> 학습 항목 {len(items)}개 (기본 증강 x{args.aug_mult})", flush=True)

    # ---- 모델 로드 ---------------------------------------------------------------------------
    disk_gb = sum(
        os.path.getsize(os.path.join(args.model, f))
        for f in os.listdir(args.model) if f.endswith(".safetensors")
    ) / 1e9
    need_gb = (disk_gb * 0.4 if args.load_4bit else disk_gb) + 2.0  # 학습은 활성화 여유 +2GB
    wait_for_free_vram(min(need_gb, 7.0))

    quant_cfg = None
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda",
        quantization_config=quant_cfg, local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels, local_files_only=True)

    if args.load_4bit and not args.skip_kbit_upcast:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # 8B가 8GB에 안 들어갈 때: kbit 준비의 fp32 업캐스트(+2.3GB)를 건너뛰고
        # gradient checkpointing만 직접 켠다 (layernorm fp32 안정화 포기, 스모크로 수렴 확인 필요)
        if args.load_4bit:
            print("⚠️ kbit fp32 업캐스트 생략 (VRAM 절약) — 학습 안정성 스모크로 확인", flush=True)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets.split(","), bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    # ---- 옵티마이저/스케줄러 ------------------------------------------------------------------
    total_opt_steps = (len(items) * args.epochs) // args.grad_accum
    if args.max_steps:
        total_opt_steps = min(total_opt_steps, args.max_steps)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_opt_steps * args.warmup_ratio), total_opt_steps
    )

    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "n_items": len(items), "total_opt_steps": total_opt_steps,
                   "started": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)

    def encode(item):
        """프롬프트 토큰은 -100 마스킹, 정답 텍스트 토큰만 지도한다."""
        messages = build_messages(item["sentence"], item["paths"], args.prompt, item.get("hint", ""))
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_msgs = messages + [{"role": "assistant", "content": [{"type": "text", "text": item["target_text"]}]}]
        full_text = processor.apply_chat_template(full_msgs, tokenize=False)
        image_inputs, video_inputs = process_vision_info(messages)

        full = processor(text=[full_text], images=image_inputs, videos=video_inputs,
                         padding=True, return_tensors="pt")
        prompt = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt")
        labels = full.input_ids.clone()
        labels[:, : prompt.input_ids.shape[1]] = -100
        full["labels"] = labels
        return full

    def save_adapter(tag):
        model.save_pretrained(adapter_dir)
        torch.cuda.empty_cache()     # 저장 경로가 잡은 임시 버퍼 반환
        print(f"[{tag}] 어댑터 저장 -> {adapter_dir}", flush=True)

    # ---- 학습 루프 ---------------------------------------------------------------------------
    torch.cuda.empty_cache()          # 모델 로드 과정의 잔여 버퍼 반환 후 시작
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    opt_step, micro_step, loss_acc = 0, 0, 0.0
    n_skipped = 0
    log_rows, stop_reason = [], "완주"

    try:
        for epoch in range(args.epochs):
            if epoch > 0:
                rng.shuffle(items)
            pbar = tqdm(items, desc=f"epoch {epoch + 1}/{args.epochs}")
            for item in pbar:
                # 특정 샘플이 VRAM 스파이크로 OOM을 내는 경우가 있다 (7/19 exp16, 같은 시드로
                # 540스텝에서 2회 재현). 해당 항목만 건너뛰고 학습은 계속한다.
                inputs = loss = None
                try:
                    inputs = encode(item).to(model.device)
                    # 샘플별 손실 가중 (--loss-weights): 증강 복제와 달리 스텝 수를 늘리지 않고
                    # 그래디언트 기여만 조절한다. 미지정 샘플은 1.0.
                    loss = model(**inputs).loss * item.get("loss_weight", 1.0) / args.grad_accum
                    loss.backward()
                except Exception as e:
                    if "out of memory" not in str(e).lower():
                        raise           # OOM이 아닌 예외는 숨기지 않는다
                    n_skipped += 1
                    print(f"\n[OOM 스킵 {n_skipped}] {item['id']} (opt_step {opt_step}) - 계속 진행", flush=True)
                    inputs = loss = None            # encode 단계 실패로 미바인딩일 수 있어 del 대신 재할당
                    optimizer.zero_grad(set_to_none=True)   # 부분 누적 grad 폐기
                    torch.cuda.empty_cache()
                    continue
                loss_acc += loss.item()
                micro_step += 1
                inputs = loss = None    # 활성 텐서 참조 즉시 해제 (다음 항목 전에 반환)

                if micro_step % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1

                    if opt_step % args.log_steps == 0 or opt_step == 1:
                        elapsed = time.time() - t_start
                        row = {
                            "opt_step": opt_step, "epoch": epoch,
                            "loss": round(loss_acc, 4),
                            "lr": scheduler.get_last_lr()[0],
                            "sec_per_item": round(elapsed / micro_step, 2),
                            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                            # 현재 예약량 (최대값이 아님) — allocated와의 격차 = 단편화/캐시 팽창.
                            # 캡(7.3GB) 근처에서 계속 놀면 GC가 일하고 있다는 뜻 (7/20 스필 진단)
                            "reserved_vram_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                            "elapsed_min": round(elapsed / 60, 1),
                        }
                        log_rows.append(row)
                        pd.DataFrame(log_rows).to_csv(log_path, index=False)
                        pbar.set_postfix(loss=row["loss"], vram=row["peak_vram_gb"])
                    loss_acc = 0.0

                    # 주기적 캐시 반환 — 단편화 누적 완화 (동기화 비용 수십 ms, N스텝당 1회)
                    if args.empty_cache_steps and opt_step % args.empty_cache_steps == 0:
                        torch.cuda.empty_cache()

                    if opt_step % args.save_steps == 0:
                        save_adapter(f"step {opt_step}")
                    if args.snapshot_steps and opt_step % args.snapshot_steps == 0:
                        snap_dir = os.path.join(out_dir, "checkpoints", f"step_{opt_step:05d}")
                        model.save_pretrained(snap_dir)
                        torch.cuda.empty_cache()     # 저장 중 잡은 임시 버퍼 반환
                        print(f"[snapshot] 스텝 {opt_step} -> {snap_dir}", flush=True)
                    if args.max_steps and opt_step >= args.max_steps:
                        stop_reason = f"max_steps({args.max_steps}) 도달"
                        raise StopIteration
                    if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
                        stop_reason = f"max_hours({args.max_hours}h) 도달"
                        raise StopIteration
    except StopIteration:
        pass
    except KeyboardInterrupt:
        stop_reason = "수동 중단"

    save_adapter("최종")
    elapsed = time.time() - t_start
    print(f"\n종료({stop_reason}): {opt_step} 스텝, {micro_step} 항목, {elapsed / 3600:.2f}시간, "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
          + (f", OOM 스킵 {n_skipped}개" if n_skipped else ""), flush=True)
    print(f"다음: python scripts/eval_zero_shot.py --model {args.model} --adapter {adapter_dir}"
          + (" --load-4bit" if args.load_4bit else "")
          + (f" --prompt {args.prompt}" if args.prompt != "v1_list" else ""), flush=True)


if __name__ == "__main__":
    main()
