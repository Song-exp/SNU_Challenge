# -*- coding: utf-8 -*-
"""CoT 응답 파인튜닝 스크립트 (train.py와 분리된 실험 버전 — 기존 파이프라인 무변경).

train.py와의 차이 (그 외 학습 루프·LoRA·안전장치는 동일 설계):
- target이 리스트 즉답("[3, 1, 4, 2]")이 아니라 **기계 생성 CoT 응답** (아래 형식)
- 기본 프롬프트 = v4_story_cot (팀 고도화 프롬프트, prompts.py 공유)
- 평가는 반드시: eval_zero_shot.py --prompt v4_story_cot --max-new-tokens 512

CoT target 형식 — "헛소리 없는 기계 생성" 원칙 (문장·정답에서 역산 가능한 사실만):
    [Story Analysis]
    - Event 1: <spacy 절 분해로 얻은 구절>   <- 문장에서만 유도
    - Event 2: ...
    [Chronological Mapping]
    - 1st: Image 3                            <- 정답에서 역산 ("because"는 넣지 않음)
    - 2nd: Image 1
    ...
    [Final Answer]
    <ANSWER>[3, 1, 4, 2]</ANSWER>
[Visual Evidence]는 이미지를 보지 않고 쓸 수 없어 1차 버전에서 제외 (설계: PLAN_cot_finetune.md).

사용 예:
    # target 생성 미리보기 (GPU 불필요)
    python scripts/train_cot.py --run-name preview --preview 5

    # 스모크 (~3분)
    python scripts/train_cot.py --run-name exp12_v4cot_aug1_smoke --max-samples 12 --grad-accum 4 --max-steps 5

    # 본학습 (1배 증강 완주, ~11-13h 예상)
    python scripts/train_cot.py --run-name exp12_v4cot_aug1 --aug-mult 1 --snapshot-steps 150
"""
import argparse
import ast
import json
import os
import random
import time
from datetime import datetime

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # 프로젝트 루트 기준

import prompts as prompt_registry
# 순수 헬퍼는 train.py에서 재사용 (train.py는 수정하지 않음)
from train import build_messages, chrono_image_numbers, load_excluded_ids, \
    keep_system_awake, wait_for_free_vram

ORDINALS = ["1st", "2nd", "3rd", "4th"]
CONNECTOR_PREFIXES = ("then ", "and ", "as ", "while ", "before ", "after ",
                      "followed by ", "next ", "finally ", "later ", "but ")
CONNECTOR_SUFFIXES = (" and", " then", " but", " as", " while", " before", " after")
TEMPORAL_ADVS = {"then", "finally", "next", "later", "afterwards", "afterward",
                 "eventually", "meanwhile", "subsequently"}


# ---------------------------------------------------------------- CoT target 생성

class EventSplitter:
    """spacy 절 경계(ROOT/conj/advcl/ccomp 동사) 기준으로 문장을 이벤트 구절로 분해.

    flag_detector.classify_syntax_spacy와 같은 의존관계 기준을 쓰므로
    문장 유형 분류(Type-1/2/3)와 일관된 절 개수가 나온다.
    """

    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    def split(self, sentence):
        doc = self.nlp(sentence)
        # 자름점 후보: 절 머리동사의 왼쪽 경계 + 시간 부사(then 등) + 전치사 뒤 동명사(before cutting ...)
        edges = []
        for t in doc:
            if t.pos_ in ("VERB", "AUX") and t.dep_ in ("ROOT", "conj", "advcl", "ccomp"):
                edges.append(t.left_edge.i)
            elif t.dep_ == "advmod" and t.lower_ in TEMPORAL_ADVS:
                edges.append(t.i)
            elif t.pos_ == "VERB" and t.dep_ == "pcomp":
                edges.append(t.head.i)  # 전치사(before/after)부터 자름 -> 접두 트림으로 제거
        if len(set(edges)) < 2:
            return [sentence.strip()]

        cuts = []  # 겹침 방지: 단조 증가만 채택
        for edge in sorted(set(edges)):
            if not cuts or edge > cuts[-1]:
                cuts.append(edge)
        if cuts[0] != 0:
            cuts[0] = 0

        events = []
        for a, b in zip(cuts, cuts[1:] + [len(doc)]):
            seg = doc[a:b].text.strip().strip(",;").strip()
            low = seg.lower()
            for pre in CONNECTOR_PREFIXES:   # 선행 접속어 제거 (이벤트 내용만 남김)
                if low.startswith(pre):
                    seg = seg[len(pre):].strip()
                    break
            seg = seg.rstrip(",;").strip()
            low = seg.lower()
            for suf in CONNECTOR_SUFFIXES:   # 꼬리에 매달린 접속어 제거
                if low.endswith(suf):
                    seg = seg[: -len(suf)].rstrip(",; ").strip()
                    break
            if len(seg.split()) >= 2:        # 한 단어짜리 껍데기 이벤트 제거
                events.append(seg)
        return events[:5] if events else [sentence.strip()]


def build_cot_target(events, target):
    """이벤트 구절 + 시간순 이미지 번호(target) -> CoT 응답 텍스트 (사실만 포함)."""
    lines = ["[Story Analysis]"]
    lines += [f"- Event {i + 1}: {e}" for i, e in enumerate(events)]
    lines.append("[Chronological Mapping]")
    lines += [f"- {ORDINALS[i]}: Image {n}" for i, n in enumerate(target)]
    lines.append("[Final Answer]")
    lines.append(f"<ANSWER>{target}</ANSWER>")
    return "\n".join(lines)


def load_gemma_events(df):
    """gemma 라벨의 events를 타깃 소스로 사용 (exp12의 spacy 분해 품질 문제 대체).

    라벨 없는 Id가 있으면 spacy 폴백 없이 중단 — 미니 풀은 전량 라벨돼 있어야 정상.
    """
    from structure_features import load_gemma_labels
    g = load_gemma_labels()
    events = {r.Id: [str(e) for e in r.events][:5] for r in g.itertuples() if r.events}
    missing = [i for i in df["Id"] if i not in events]
    if missing:
        raise SystemExit(f"gemma events 누락 {len(missing)}개 (예: {missing[:5]}) — "
                         f"라벨링 범위를 확인하거나 --events-from spacy로 실행")
    return events


def build_training_items_cot(df, image_dir, aug_mult, rng, splitter, events_from="spacy",
                             clip_pairs=None):
    """train.py의 build_training_items와 동일한 증강 규약 + CoT target.
    이벤트 분해는 샘플당 1회(변형 불변), target 리스트만 변형별로 재계산.
    clip_pairs가 있으면 변형별 재매핑 힌트를 item["hint"]에 저장 (v7_cot_hint용)."""
    from structure_features import hint_text, remap_pairs
    if events_from == "gemma":
        print("이벤트 소스: gemma 라벨", flush=True)
        events_by_id = load_gemma_events(df)
    else:
        print("문장 이벤트 분해 중 (spacy)...", flush=True)
        events_by_id = {row["Id"]: splitter.split(row["Sentence"]) for _, row in df.iterrows()}

    items = []
    for _, row in df.iterrows():
        files = [row["Input_1"], row["Input_2"], row["Input_3"], row["Input_4"]]
        answer = ast.literal_eval(row["Answer"])
        chrono = chrono_image_numbers(answer)
        time_files = [files[n - 1] for n in chrono]
        events = events_by_id[row["Id"]]

        seen = set()
        for v in range(aug_mult):
            if v == 0:
                perm = list(range(4))
            else:
                perm = list(range(4))
                for _ in range(10):
                    rng.shuffle(perm)
                    if tuple(perm) not in seen:
                        break
            seen.add(tuple(perm))

            shown_files = [files[j] for j in perm]
            target = [shown_files.index(f) + 1 for f in time_files]
            hint = ""
            if clip_pairs is not None:
                hint = hint_text(remap_pairs(clip_pairs.get(row["Id"], []), perm))
            items.append({
                "id": row["Id"],
                "sentence": row["Sentence"],
                "paths": [os.path.join(image_dir, row["Id"], f) for f in shown_files],
                "target_text": build_cot_target(events, target),
                "hint": hint,
            })
    return items


# ---------------------------------------------------------------- 메인

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--aug-mult", type=int, default=1, help="1차 검증은 1배 증강 (7/15 결정)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--prompt", default="v4_story_cot", choices=list(prompt_registry.PROMPTS))
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=640 * 480)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--snapshot-steps", type=int, default=0)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--data-dir", default="./snuaichallenge_data/")
    parser.add_argument("--preview", type=int, default=0,
                        help="N개 샘플의 CoT target만 출력하고 종료 (GPU 불필요)")
    parser.add_argument("--events-from", default="spacy", choices=["spacy", "gemma"],
                        help="타깃 이벤트 소스 — gemma = outputs/gemma_labels (v7 트랙)")
    args = parser.parse_args()

    import pandas as pd

    random.seed(args.seed)
    rng = random.Random(args.seed)
    splitter = EventSplitter() if args.events_from == "spacy" else None

    clip_pairs = None
    if prompt_registry.needs_hint(args.prompt):
        from structure_features import load_clip_pairs
        clip_pairs = load_clip_pairs()
        print(f"힌트 주입: CLIP 유사쌍 {len(clip_pairs)}개 Id (프롬프트 {args.prompt})", flush=True)

    # ---- 데이터 ------------------------------------------------------------------------------
    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    excluded = load_excluded_ids()
    train_df = train_df[~train_df["Id"].isin(excluded)].reset_index(drop=True)

    if args.preview:  # target 생성 미리보기 — 학습 없이 형식 눈검사 (미니 풀과 같은 시드로 뽑음)
        sample = train_df.sample(n=1000, random_state=args.seed).head(args.preview)
        items = build_training_items_cot(sample, os.path.join(args.data_dir, "train"),
                                         2, rng, splitter, args.events_from, clip_pairs)
        for it in items:
            print(f"\n===== {it['id']} =====\n문장: {it['sentence']}")
            if it["hint"]:
                print(f"힌트: {it['hint'].strip()}")
            print(f"--- target ---\n{it['target_text']}")
        return

    if args.max_samples:
        train_df = train_df.sample(n=args.max_samples, random_state=args.seed).reset_index(drop=True)
    image_dir = os.path.join(args.data_dir, "train")
    items = build_training_items_cot(train_df, image_dir, args.aug_mult, rng, splitter,
                                     args.events_from, clip_pairs)
    rng.shuffle(items)
    tgt_chars = sum(len(i["target_text"]) for i in items) // max(len(items), 1)
    print(f"기반 {len(train_df)}개 x 증강 {args.aug_mult} = 학습 항목 {len(items)}개 "
          f"| 평균 target {tgt_chars}자", flush=True)

    # ---- 모델 로드 (train.py와 동일 설계) ------------------------------------------------------
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.optimization import get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model
    from qwen_vl_utils import process_vision_info

    assert torch.cuda.is_available(), "GPU 필요"
    keep_system_awake()
    torch.manual_seed(args.seed)

    out_dir = os.path.join("./outputs/runs", args.run_name)
    adapter_dir = os.path.join(out_dir, "adapter")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.csv")

    disk_gb = sum(
        os.path.getsize(os.path.join(args.model, f))
        for f in os.listdir(args.model) if f.endswith(".safetensors")
    ) / 1e9
    need_gb = (disk_gb * 0.4 if args.load_4bit else disk_gb) + 2.0
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

    total_opt_steps = (len(items) * args.epochs) // args.grad_accum
    if args.max_steps:
        total_opt_steps = min(total_opt_steps, args.max_steps)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_opt_steps * args.warmup_ratio), total_opt_steps
    )

    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "script": "train_cot.py", "n_items": len(items),
                   "total_opt_steps": total_opt_steps,
                   "started": datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)

    def encode(item):
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
        print(f"[{tag}] 어댑터 저장 -> {adapter_dir}", flush=True)

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
                    if args.snapshot_steps and opt_step % args.snapshot_steps == 0:
                        snap_dir = os.path.join(out_dir, "checkpoints", f"step_{opt_step:05d}")
                        model.save_pretrained(snap_dir)
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
          f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)
    print(f"다음: python scripts/eval_zero_shot.py --model {args.model} --adapter {adapter_dir} "
          f"--prompt {args.prompt} --max-new-tokens 512"
          + (" --load-4bit" if args.load_4bit else ""), flush=True)


if __name__ == "__main__":
    main()
