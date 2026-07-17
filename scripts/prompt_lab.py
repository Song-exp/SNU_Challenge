# -*- coding: utf-8 -*-
"""프롬프트 추론 실험 헬퍼 — Prompt_Experiments.ipynb 전용.

exp07(가중치 고정) 위에서 추론 프롬프트 변형·힌트 주입·취약 세그먼트 라우팅을
반복 실험한다. 파싱·채점은 eval_zero_shot.py와 동일 로직을 재사용하므로
결과가 기존 experiments.csv 수치와 직접 비교 가능하다.

- 모델은 노트북에서 1회 로드해 모든 변형에 재사용 (변형당 ~5분, 라우팅은 ~2분)
- 요약은 outputs/prompt_experiments.csv 누적, 예측 전문은 outputs/preds/prompt_<name>.csv
- 판정: paired 비교(새로 맞춤/틀림) + shuffled ±4%p 노이즈 기준 (EXPERIMENTS.md)
"""
import ast
import os
import sys
import time
from datetime import datetime

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src", "features"))

import pandas as pd

from eval_zero_shot import parse_model_output, wait_for_free_vram  # noqa: E402

BASELINE_PREDS = "./outputs/preds/Qwen3-VL-2B-Instruct_exp07_aug2_full.csv"
RESULTS_CSV = "./outputs/prompt_experiments.csv"
PREDS_DIR = "./outputs/preds"
PAIR_COLS = ["dist_12", "dist_13", "dist_14", "dist_23", "dist_24", "dist_34"]


# ---------------------------------------------------------------- 데이터 준비
def load_holdout(split_path="./splits/holdout_300.csv",
                 clip_csv="./snu_clip_features.csv"):
    """holdout 300 + 문장유형/플래그/ai_score(spacy) + CLIP 거리 피처를 메모리에 준비."""
    df = pd.read_csv(split_path)

    from flag_detector import OrthogonalFlagDetector
    det = OrthogonalFlagDetector()
    feats = pd.DataFrame([det.process_sentence(s) for s in df["Sentence"]])
    df = pd.concat([df.reset_index(drop=True),
                    feats.drop(columns=["Sentence"]).reset_index(drop=True)], axis=1)

    clip = pd.read_csv(clip_csv)
    df = df.merge(clip[["Id", "predicted_scene_cuts"] + PAIR_COLS], on="Id", how="left")

    # 취약 세그먼트 = Round 2 라우팅 대상 (§1 실측: Type-1 17.6%, ai_score>0.5 16~25%)
    df["weak"] = (df["Partition"] == "Type-1") | (df["ai_score"] > 0.5)
    n_weak_sh = int((df["weak"] & ~df["No_ordering"]).sum())
    print(f"holdout {len(df)}개 | 취약 세그먼트 {int(df['weak'].sum())}개 "
          f"(shuffled 기준 {n_weak_sh}개)")
    return df


def clip_pairs_text(row, k=2):
    """CLIP 거리가 가장 가까운 이미지쌍 k개 → 'Image 1 & Image 3, Image 2 & Image 4'."""
    dists = sorted((row[c], c) for c in PAIR_COLS if pd.notna(row[c]))
    pairs = [f"Image {c[5]} & Image {c[6]}" for _, c in dists[:k]]
    return ", ".join(pairs)


def load_baseline(df, path=BASELINE_PREDS):
    """exp07 v1_list 기존 예측(그리디라 재현 동일)을 baseline res_df 형태로 로드."""
    base = pd.read_csv(path)[["Id", "pred", "correct", "parsed", "raw"]]
    base = df[["Id", "No_ordering"]].merge(base, on="Id")
    assert len(base) == len(df), "baseline 예측과 holdout Id 불일치"
    return base


# ---------------------------------------------------------------- 모델/추론
def load_model(model_path="./models/Qwen3-VL-2B-Instruct",
               adapter="./outputs/runs/exp07_aug2_full/adapter",
               max_pixels=640 * 480):
    """eval_zero_shot.py와 동일 설정으로 모델+어댑터 1회 로드 (fp16, ~4.7GB)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    assert torch.cuda.is_available(), "GPU 필요"
    wait_for_free_vram(5.5)  # 학습 프로세스가 GPU 점유 중이면 대기
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.float16, device_map="cuda", local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_path, max_pixels=max_pixels, local_files_only=True)
    print(f"모델 로드 완료: {model_path} + {adapter}")
    return model, processor


def _build_messages(row, user_text, image_dir="./snuaichallenge_data/train"):
    content = []
    for i, col in enumerate(["Input_1", "Input_2", "Input_3", "Input_4"]):
        content.append({"type": "image",
                        "image": os.path.join(image_dir, row["Id"], row[col])})
        content.append({"type": "text", "text": f"\nImage {i + 1}\n"})
    content.append({"type": "text", "text": user_text})
    return [{"role": "user", "content": content}]


def _infer_one(model, processor, row, user_text, max_new_tokens,
               image_dir="./snuaichallenge_data/train"):
    import torch
    from qwen_vl_utils import process_vision_info

    messages = _build_messages(row, user_text, image_dir)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                  skip_special_tokens=True)[0]


# ---------------------------------------------------------------- 실행/채점
def run_variant(name, text_fn, model, processor, df, baseline=None,
                max_new_tokens=32, subset=None, results_csv=RESULTS_CSV):
    """변형 1개 실행. subset(bool Series) 지정 시 해당 행만 추론하고
    나머지는 baseline 결과를 그대로 합쳐 라우팅 성적을 낸다."""
    from tqdm.auto import tqdm

    target = df if subset is None else df[subset]
    if subset is not None:
        assert baseline is not None, "subset 실행에는 baseline이 필요"
    print(f"[{name}] 추론 {len(target)}개 (max_new_tokens={max_new_tokens})")

    records, t0 = [], time.time()
    for _, row in tqdm(target.iterrows(), total=len(target)):
        raw = _infer_one(model, processor, row, text_fn(row), max_new_tokens)
        pred, parsed = parse_model_output(raw)
        gt = ast.literal_eval(row["Answer"])
        records.append({"Id": row["Id"], "No_ordering": row["No_ordering"],
                        "pred": str(pred), "correct": pred == gt,
                        "parsed": parsed, "raw": raw})
    sec = round((time.time() - t0) / max(len(records), 1), 2)
    res = pd.DataFrame(records)

    routed_n = None
    if subset is not None:  # 라우팅: 비대상 행은 baseline 결과 유지
        routed_n = len(res)
        keep = baseline[~baseline["Id"].isin(res["Id"])]
        res = pd.concat([res, keep], ignore_index=True)
    res = df[["Id", "Partition", "ai_score", "weak", "Sentence"]].merge(res, on="Id")

    summary = summarize(name, res, baseline, sec, max_new_tokens, routed_n)
    new_row = pd.DataFrame([summary])
    if os.path.exists(results_csv):
        new_row = pd.concat([pd.read_csv(results_csv), new_row], ignore_index=True)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    new_row.to_csv(results_csv, index=False)
    os.makedirs(PREDS_DIR, exist_ok=True)
    res.to_csv(f"{PREDS_DIR}/prompt_{name}.csv", index=False)

    print("  " + " | ".join(f"{k} {v}" for k, v in summary.items()
                            if k not in ("timestamp", "name")))
    return res


def summarize(name, res, baseline, sec_per_sample, max_new_tokens, routed_n):
    sh = res[~res["No_ordering"]]
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "name": name,
        "eval_n": len(res),
        "routed_n": routed_n,
        "max_new_tokens": max_new_tokens,
        "accuracy": round(res["correct"].mean(), 4),
        "acc_shuffled": round(sh["correct"].mean(), 4),
        "acc_weak": round(res[res["weak"]]["correct"].mean(), 4),
        "acc_type1": round(res[res["Partition"] == "Type-1"]["correct"].mean(), 4),
        "acc_type2": round(res[res["Partition"] == "Type-2"]["correct"].mean(), 4),
        "acc_type3": round(res[res["Partition"] == "Type-3"]["correct"].mean(), 4),
        "parse_fail": int((~res["parsed"]).sum()),
        "sec_per_sample": sec_per_sample,
    }
    if baseline is not None:  # paired 비교 (같은 300개, greedy라 결정론적)
        b = baseline.set_index("Id")["correct"]
        r = res.set_index("Id")["correct"]
        row["fixed"] = int((r & ~b).sum())    # 새로 맞춤
        row["broken"] = int((~r & b).sum())   # 새로 틀림
    return row


def segment_table(res):
    """Partition × weak 분해표 — 모든 변형 판정에 첨부 (풍선 효과 감시)."""
    sh = res[~res["No_ordering"]]
    out = sh.groupby("Partition")["correct"].agg(n="size", acc="mean").round(3)
    w = sh.groupby("weak")["correct"].agg(n="size", acc="mean").round(3)
    w.index = w.index.map({True: "weak(라우팅 대상)", False: "strong"})
    return pd.concat([out, w])


def show_results(results_csv=RESULTS_CSV):
    exp = pd.read_csv(results_csv)
    cols = ["timestamp", "name", "routed_n", "accuracy", "acc_shuffled", "acc_weak",
            "acc_type1", "acc_type2", "acc_type3", "parse_fail", "fixed", "broken",
            "sec_per_sample"]
    return exp[[c for c in cols if c in exp.columns]].sort_values("acc_shuffled")


# ---------------------------------------------------------------- 제출 생성
def load_testset(test_csv="./snuaichallenge_data/test.csv",
                 clip_csv="./snu_clip_features_test.csv"):
    """test 819 + 문장 피처(spacy, ~1분) 준비. 라우팅형 text_fn(row.weak 분기)에 필요.

    CLIP 피처 파일이 아직 없으면 경고만 하고 진행 — 이 경우 clip_hint를 쓰는
    변형은 사용 불가 (전처리 트랙에서 test 819개 CLIP 피처 생성 후 가능).
    """
    df = pd.read_csv(test_csv)

    from flag_detector import OrthogonalFlagDetector
    det = OrthogonalFlagDetector()
    feats = pd.DataFrame([det.process_sentence(s) for s in df["Sentence"]])
    df = pd.concat([df.reset_index(drop=True),
                    feats.drop(columns=["Sentence"]).reset_index(drop=True)], axis=1)
    df["weak"] = (df["Partition"] == "Type-1") | (df["ai_score"] > 0.5)

    if os.path.exists(clip_csv):
        clip = pd.read_csv(clip_csv)
        df = df.merge(clip[["Id", "predicted_scene_cuts"] + PAIR_COLS], on="Id", how="left")
    else:
        print(f"경고: {clip_csv} 없음 - CLIP 힌트 변형은 제출에 사용 불가 (라우팅/문장 피처는 가능)")

    print(f"test {len(df)}개 | 취약 세그먼트 {int(df['weak'].sum())}개")
    return df


def make_submission(name, text_fn, model, processor, test_df,
                    max_new_tokens=32, image_dir="./snuaichallenge_data/test",
                    sample_csv="./snuaichallenge_data/sample_submission.csv",
                    out_dir="./outputs/submissions"):
    """확정 변형으로 test 전체 추론 → 제출 CSV 생성 + 형식 검증.

    - text_fn(row): 노트북 VARIANTS의 함수 그대로. 라우팅 제출은 함수 안에서 분기:
        lambda row: (v1(row) + CAUSAL) if row.weak else v1(row)
    - Answer는 parse_model_output이 제출 형식("각 이미지의 원래 위치")으로 역변환한 값.
      파싱 실패 시 [1, 2, 3, 4] 폴백 (제출 무효화 방지) — parse_fail 수를 반드시 확인할 것.
    - 소요: 819개 × sec/sample (v1 ~0.9초 = 13분, 풀 CoT ~8초 = 2h; 규정 한도 24h)
    """
    from tqdm.auto import tqdm

    records, t0 = [], time.time()
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        raw = _infer_one(model, processor, row, text_fn(row), max_new_tokens,
                         image_dir=image_dir)
        pred, parsed = parse_model_output(raw)
        records.append({"Id": row["Id"], "Answer": str(pred),
                        "parsed": parsed, "raw": raw})
    elapsed = time.time() - t0
    res = pd.DataFrame(records)

    # ---- 형식 검증 (제출 전 필수 통과) ----
    sample = pd.read_csv(sample_csv)
    problems = []
    if len(res) != len(sample):
        problems.append(f"행 수 불일치: {len(res)} vs sample {len(sample)}")
    if set(res["Id"]) != set(sample["Id"]):
        problems.append("Id 집합이 sample_submission과 불일치")
    bad = [r.Id for r in res.itertuples()
           if sorted(ast.literal_eval(r.Answer)) != [1, 2, 3, 4]]
    if bad:
        problems.append(f"순열 무효 {len(bad)}건: {bad[:5]}")
    assert not problems, "제출 형식 오류!\n- " + "\n- ".join(problems)
    sub = sample[["Id"]].merge(res[["Id", "Answer"]], on="Id")  # sample 순서 유지

    os.makedirs(out_dir, exist_ok=True)
    tag = datetime.now().strftime("%m%d_%H%M")
    sub_path = f"{out_dir}/submission_{name}_{tag}.csv"
    sub.to_csv(sub_path, index=False)
    res.to_csv(f"{out_dir}/preds_{name}_{tag}.csv", index=False)  # raw 감사용 보존

    n_fail = int((~res["parsed"]).sum())
    n_identity = int((res["Answer"] == "[1, 2, 3, 4]").sum())
    print(f"완료: {len(sub)}행 | 파싱 실패(폴백) {n_fail} | [1,2,3,4] 예측 {n_identity}개 "
          f"({n_identity / len(sub):.0%}) | {elapsed / len(sub):.2f}초/샘플, 총 {elapsed / 60:.0f}분\n"
          f"제출 파일: {sub_path}")
    if n_fail > 0:
        print(f"⚠️ 파싱 실패 {n_fail}건은 [1,2,3,4] 폴백 — preds_{name}_{tag}.csv의 raw로 원인 확인 권장")
    return sub_path
