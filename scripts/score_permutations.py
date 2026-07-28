# -*- coding: utf-8 -*-
"""우도 스코어링 + 순열 TTA — 추론 레버 (학습 없음, GPU 채점 전용).

생성(greedy) 대신 24개 순열 후보를 teacher-forcing 로그우도로 채점하고,
제시 배치를 순환이동 K개로 바꿔 얻은 점수를 원본 좌표로 합산(-prior)해 argmax.

⚠️ scripts/test_perm_coords.py (0단계) 통과가 전제. 좌표 변환은 그 검증본 재사용.

단계 (계획서 §6):
    --k 1               K=1 (e0만), 우도 채점만
    --k 1 --prior <csv> + prior 차감
    --k 4 --prior <csv> + K=4 TTA 합산 (최종)
    --build-prior <csv> 채점 결과로 prior 테이블 생성·저장 (holdout에서 동결)

사용:
    # holdout K=1 (prior 생성 겸)
    python scripts/score_permutations.py --model ./models/Qwen3-VL-4B-Instruct --load-4bit \
        --adapter ./outputs/runs/exp17_4b_reorder_sparseaug/adapter --prompt v5_reorder \
        --k 1 --build-prior ./outputs/prior_exp17.csv
    # holdout K=4 + prior
    python scripts/score_permutations.py ... --k 4 --prior ./outputs/prior_exp17.csv
    # test 제출 생성
    python scripts/score_permutations.py ... --k 4 --prior ./outputs/prior_exp17.csv \
        --split test --submission exp17_tta
"""
import argparse
import ast
import os
import time
from itertools import permutations

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:256")
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import prompts as prompt_registry
from eval_zero_shot import get_prompt_message

# 순환이동 배치 (라틴 방진 — 슬롯 균형): perm[j] = 새 슬롯 j에 오는 원본 0-based 인덱스
PERMS = {"e0": [0, 1, 2, 3], "r1": [1, 2, 3, 0], "r2": [2, 3, 0, 1], "r3": [3, 0, 1, 2]}
PERM_ORDER = ["e0", "r1", "r2", "r3"]
CANDIDATES = [list(p) for p in permutations([1, 2, 3, 4])]  # 24개 대회 정답 후보


def chrono_from_answer(answer):
    c = [0] * 4
    for i, pos in enumerate(answer):
        c[pos - 1] = i + 1
    return c


def target_string(answer, perm):
    """정답 후보(대회 형식)를 제시 순열 perm 하에서 채점할 문자열 (test_perm_coords 검증본)."""
    files = [1, 2, 3, 4]
    chrono = chrono_from_answer(answer)
    time_files = [files[n - 1] for n in chrono]
    shown = [files[j] for j in perm]
    target = [shown.index(f) + 1 for f in time_files]
    return str(target)  # "[3, 1, 4, 2]"


def submission_from_string(model_list, perm):
    """제시 순열 perm에서의 출력 리스트 -> 대회 제출 형식(원본 좌표)."""
    sub = [0] * 4
    for k, slot in enumerate(model_list):
        orig = perm[slot - 1] + 1
        sub[orig - 1] = k + 1
    return sub


def build_perm_row(row, perm):
    """row의 이미지 순서를 perm으로 재배치한 사본 (get_prompt_message 재사용)."""
    r = dict(row)
    cols = ["Input_1", "Input_2", "Input_3", "Input_4"]
    orig = [row[c] for c in cols]
    for j, c in enumerate(cols):
        r[c] = orig[perm[j]]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--prompt", default="v5_reorder", choices=list(prompt_registry.PROMPTS))
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=640 * 480)
    ap.add_argument("--k", type=int, default=1, choices=[1, 4], help="제시 배치 수 (1=e0, 4=순환TTA)")
    ap.add_argument("--prior", default="", help="prior 테이블 CSV (문자열별 로그우도 평균) 차감")
    ap.add_argument("--build-prior", default="", help="채점 결과로 prior 테이블 생성·저장 경로")
    ap.add_argument("--split", default="./splits/holdout_300.csv",
                    help="holdout CSV 또는 'test'")
    ap.add_argument("--submission", default="", help="test 제출명 (지정 시 제출 CSV 생성)")
    ap.add_argument("--data-dir", default="./snuaichallenge_data/")
    ap.add_argument("--limit", type=int, default=0, help="디버그: 앞 N개만")
    args = ap.parse_args()

    if args.prompt.startswith(("v6", "v7", "v9", "v10")):
        raise SystemExit("힌트 주입 프롬프트는 우도 채점 대상 아님 (§5-6). v1_list/v5_reorder만.")

    import pandas as pd
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    is_test = args.split == "test"
    split_path = f"{args.data_dir}/test.csv" if is_test else args.split
    df = pd.read_csv(split_path)
    if args.limit:
        df = df.head(args.limit)
    image_dir = os.path.join(args.data_dir, "test" if is_test else "train")

    prior = {}
    if args.prior:
        pdf = pd.read_csv(args.prior)
        prior = dict(zip(pdf["string"], pdf["logprob"]))

    # ---- 모델 로드 ----
    quant_cfg = None
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda",
        quantization_config=quant_cfg, local_files_only=True)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels, local_files_only=True)
    print(f"채점: {len(df)}개 x K={args.k} x 24후보 | prompt={args.prompt} | prior={'O' if prior else 'X'}",
          flush=True)

    perms_used = PERM_ORDER[:args.k]

    # rope_deltas 보유 모듈 탐색 (PeftModel이면 2겹 더 안쪽)
    inner = model
    for _ in range(4):
        if hasattr(inner, "rope_deltas"):
            break
        inner = getattr(inner, "model", None) or getattr(inner, "base_model", None)
        if inner is None:
            raise RuntimeError("rope_deltas 모듈을 못 찾음")
    eos_ids = processor.tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def score_candidate_strings(messages, cand_strings):
        """KV 캐시 공유 우도 채점 — 이미지+프롬프트 1회 forward 후 24후보를 배치로 채점.

        Qwen3-VL M-RoPE: 답 토큰은 순수 텍스트라 3D position을 (prefix끝+rope_delta)에서
        연속으로 부여. 캐시 배치 확장은 DynamicCache.batch_repeat_interleave.
        (캐시=비캐시 우도 일치 검증 완료, 오차 <0.02.) 후보 답토큰 길이는 12로 균일."""
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        enc = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt").to(model.device)
        plen = enc["input_ids"].shape[1]
        n = len(cand_strings)

        ans_tok = [processor.tokenizer(cs, add_special_tokens=False)["input_ids"] + eos_ids
                   for cs in cand_strings]
        L = len(ans_tok[0])
        assert all(len(t) == L for t in ans_tok), "후보 답 토큰 길이 불균일 — 패딩 경로 필요"
        ans_mat = torch.tensor(ans_tok, device=model.device)

        o1 = model(**enc, use_cache=True)
        pkv = o1.past_key_values
        rd = inner.rope_deltas.item()
        first_lp = torch.log_softmax(o1.logits[:, -1, :].float(), -1)      # 첫 답토큰 예측
        pos = (torch.arange(plen, plen + L, device=model.device) + rd).view(1, 1, -1).expand(3, n, -1).contiguous()
        pkv.batch_repeat_interleave(n)
        o2 = model(input_ids=ans_mat, position_ids=pos, past_key_values=pkv,
                   attention_mask=torch.ones(n, plen + L, device=model.device, dtype=torch.long),
                   use_cache=True)
        lp0 = first_lp[0, ans_mat[:, 0]]
        rest = torch.log_softmax(o2.logits[:, :-1].float(), -1)
        lp_rest = rest[torch.arange(n)[:, None], torch.arange(L - 1)[None, :], ans_mat[:, 1:]].sum(1)
        total = lp0 + lp_rest
        return [float(x) for x in total]

    # prior 생성용 누적
    prior_acc = {}
    records = []
    t0 = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 후보별 원본좌표 점수 누적
        score = {tuple(a): 0.0 for a in CANDIDATES}
        for pname in perms_used:
            perm = PERMS[pname]
            prow = build_perm_row(row, perm)
            messages = get_prompt_message(prow, image_dir, args.prompt, "")
            cand_strings = [target_string(a, perm) for a in CANDIDATES]
            logps = score_candidate_strings(messages, cand_strings)
            for a, cs, lp in zip(CANDIDATES, cand_strings, logps):
                s = lp - prior.get(cs, 0.0)
                score[tuple(a)] += s
                if args.build_prior:
                    prior_acc.setdefault(cs, []).append(lp)
        best = max(CANDIDATES, key=lambda a: score[tuple(a)])
        rec = {"Id": row["Id"], "pred_answer": str(best)}
        if not is_test:
            gt = ast.literal_eval(row["Answer"])
            rec["correct"] = int(best == gt)
            rec["gt_identity"] = int(gt == [1, 2, 3, 4])
            rec["pred_identity"] = int(best == [1, 2, 3, 4])
        records.append(rec)

    res = pd.DataFrame(records)
    elapsed = time.time() - t0
    print(f"완료 {elapsed/60:.1f}분 ({elapsed/len(df):.1f}초/샘플)", flush=True)

    if args.build_prior:
        rows = [{"string": k, "logprob": sum(v) / len(v)} for k, v in prior_acc.items()]
        pd.DataFrame(rows).to_csv(args.build_prior, index=False)
        print(f"prior 테이블 저장: {args.build_prior} ({len(rows)}개 문자열)", flush=True)

    if not is_test:
        acc = res["correct"].mean()
        sh = res[res.gt_identity == 0]
        idn = res[res.gt_identity == 1]
        print(f"\n=== 결과 (K={args.k}, prior={'O' if prior else 'X'}) ===")
        print(f"acc_total   : {acc:.4f} (n={len(res)})")
        print(f"acc_shuffled: {sh['correct'].mean():.4f} (n={len(sh)})  <- 주 판정 지표")
        print(f"acc_identity: {idn['correct'].mean():.4f} (n={len(idn)})")
        print(f"pred_identity율: {res['pred_identity'].mean():.3f}")

    if args.submission and is_test:
        out_dir = "./outputs/submissions"
        os.makedirs(out_dir, exist_ok=True)
        sub = res[["Id", "pred_answer"]].rename(columns={"pred_answer": "Answer"})
        stamp = time.strftime("%m%d_%H%M")
        path = os.path.join(out_dir, f"submission_{args.submission}_{stamp}.csv")
        sub.to_csv(path, index=False)
        print(f"제출 저장: {path}", flush=True)


if __name__ == "__main__":
    main()
