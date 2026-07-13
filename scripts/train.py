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
import random
import time
from datetime import datetime

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # 프로젝트 루트 기준


# ---------------------------------------------------------------- 프롬프트 (prompts.py 레지스트리 공유)

import prompts as prompt_registry


def build_messages(sentence, image_paths, prompt_name="v1_list"):
    content = []
    for i, p in enumerate(image_paths):
        content.append({"type": "image", "image": p})
        content.append({"type": "text", "text": f"\nImage {i + 1}\n"})
    content.append({"type": "text", "text": prompt_registry.build_user_text(prompt_name, sentence)})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------- 데이터 구성

def chrono_image_numbers(answer):
    """제출 형식 Answer([i]=이미지 i+1의 시간상 위치) -> 시간순 이미지 번호 리스트."""
    c = [0] * 4
    for i, pos in enumerate(answer):
        c[pos - 1] = i + 1
    return c


def build_training_items(df, image_dir, aug_mult, rng):
    """각 샘플을 aug_mult개의 (제시 순서 변형, 시간순 라벨) 학습 항목으로 확장한다."""
    items = []
    for _, row in df.iterrows():
        files = [row["Input_1"], row["Input_2"], row["Input_3"], row["Input_4"]]
        answer = ast.literal_eval(row["Answer"])
        chrono = chrono_image_numbers(answer)             # 시간순 이미지 번호 (원본 제시 기준)
        time_files = [files[n - 1] for n in chrono]       # 시간순 파일 목록 (변형 불변)

        seen = set()
        for v in range(aug_mult):
            if v == 0:
                perm = list(range(4))                      # 변형 0 = 원본 제시 순서
            else:
                perm = list(range(4))
                for _ in range(10):                        # 이미 만든 변형과 중복 회피 (최선 노력)
                    rng.shuffle(perm)
                    if tuple(perm) not in seen:
                        break
            seen.add(tuple(perm))

            shown_files = [files[j] for j in perm]         # 이번 변형에서 Image 1~4로 제시되는 파일
            target = [shown_files.index(f) + 1 for f in time_files]  # 시간순 -> 제시 라벨
            items.append({
                "id": row["Id"],
                "sentence": row["Sentence"],
                "paths": [os.path.join(image_dir, row["Id"], f) for f in shown_files],
                "target_text": str(target),                # 예: "[3, 1, 4, 2]" (eval 파서와 동일 형식)
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
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="기반 샘플 수 제한 (0=전체, 스모크용)")
    parser.add_argument("--max-steps", type=int, default=0, help="옵티마이저 스텝 상한 (0=제한 없음)")
    parser.add_argument("--max-hours", type=float, default=0, help="시간 상한, 초과 시 저장 후 종료 (0=없음)")
    parser.add_argument("--save-steps", type=int, default=100, help="어댑터 주기 저장 (옵티마이저 스텝 단위)")
    parser.add_argument("--log-steps", type=int, default=10)
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
    items = build_training_items(train_df, image_dir, args.aug_mult, rng)
    rng.shuffle(items)
    print(f"기반 {len(train_df)}개 x 증강 {args.aug_mult} = 학습 항목 {len(items)}개", flush=True)

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

    if args.load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
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
        messages = build_messages(item["sentence"], item["paths"], args.prompt)
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
        print(f"[{tag}] 어댑터 저장 -> {adapter_dir}", flush=True)

    # ---- 학습 루프 ---------------------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    opt_step, micro_step, loss_acc = 0, 0, 0.0
    log_rows, stop_reason = [], "완주"

    try:
        for epoch in range(args.epochs):
            if epoch > 0:
                rng.shuffle(items)
            pbar = tqdm(items, desc=f"epoch {epoch + 1}/{args.epochs}")
            for item in pbar:
                inputs = encode(item).to(model.device)
                loss = model(**inputs).loss / args.grad_accum
                loss.backward()
                loss_acc += loss.item()
                micro_step += 1

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
                            "elapsed_min": round(elapsed / 60, 1),
                        }
                        log_rows.append(row)
                        pd.DataFrame(log_rows).to_csv(log_path, index=False)
                        pbar.set_postfix(loss=row["loss"], vram=row["peak_vram_gb"])
                    loss_acc = 0.0

                    if opt_step % args.save_steps == 0:
                        save_adapter(f"step {opt_step}")
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
          f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)
    print(f"다음: python scripts/eval_zero_shot.py --model {args.model} --adapter {adapter_dir}"
          + (" --load-4bit" if args.load_4bit else "")
          + (f" --prompt {args.prompt}" if args.prompt != "v1_list" else ""), flush=True)


if __name__ == "__main__":
    main()
