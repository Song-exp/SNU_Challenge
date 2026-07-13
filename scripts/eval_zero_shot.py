# -*- coding: utf-8 -*-
"""고정 평가셋(holdout) zero-shot 평가 스크립트 — Model_Experiments.ipynb의 .py 버전.

노트북과 동일한 프롬프트/파싱/채점을 사용하므로 결과가 experiments.csv에서 직접 비교 가능하다.
GPU 메모리가 부족하면 확보될 때까지 대기 후 시작한다 (다른 커널이 GPU를 쓰고 있을 때).

사용 예:
    python scripts/eval_zero_shot.py --model ./models/Qwen2.5-VL-3B-Instruct --load-4bit
"""
import argparse
import ast
import os
import time
from datetime import datetime

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # 프로젝트 루트 기준 상대 경로

import prompts as prompt_registry  # noqa: E402 (scripts/ 가 sys.path[0])


def parse_model_output(output_text):
    """출력에서 순열을 추출해 제출 형식(각 이미지의 원래 위치)으로 역변환."""
    try:
        s, e = output_text.find("["), output_text.rfind("]")
        if s != -1 and e != -1:
            result = ast.literal_eval(output_text[s : e + 1])
            if isinstance(result, list) and sorted(result) == [1, 2, 3, 4]:
                sub = [0] * 4
                for idx, img_num in enumerate(result):
                    sub[img_num - 1] = idx + 1
                return sub, True
    except (ValueError, SyntaxError):
        pass
    return [1, 2, 3, 4], False


def get_prompt_message(row, image_dir, prompt_name="v1_list"):
    img_files = [row["Input_1"], row["Input_2"], row["Input_3"], row["Input_4"]]
    content = []
    for i, img_file in enumerate(img_files):
        content.append({"type": "image", "image": os.path.join(image_dir, row["Id"], img_file)})
        content.append({"type": "text", "text": f"\nImage {i + 1}\n"})
    content.append({"type": "text", "text": prompt_registry.build_user_text(prompt_name, row["Sentence"])})
    return [{"role": "user", "content": content}]


def wait_for_free_vram(required_gb: float, timeout_hours: float = 8.0):
    import torch
    deadline = time.time() + timeout_hours * 3600
    while time.time() < deadline:
        free, _ = torch.cuda.mem_get_info()
        if free / 1e9 >= required_gb:
            return
        print(f"VRAM 대기: 여유 {free / 1e9:.1f}GB < 필요 {required_gb:.1f}GB (다른 커널 종료 필요)", flush=True)
        time.sleep(60)
    raise RuntimeError("VRAM 확보 대기 시간 초과")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="모델 경로 (예: ./models/Qwen2-VL-2B-Instruct)")
    parser.add_argument("--adapter", default=None, help="PEFT LoRA 어댑터 경로 (파인튜닝 결과 평가용)")
    parser.add_argument("--prompt", default="v1_list", choices=list(prompt_registry.PROMPTS),
                        help="프롬프트 이름 (파인튜닝 모델은 학습 때와 같은 이름 필수)")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=640 * 480)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--split", default="./splits/holdout_300.csv")
    parser.add_argument("--results", default="./outputs/experiments.csv")
    parser.add_argument("--data-dir", default="./snuaichallenge_data/")
    args = parser.parse_args()

    import pandas as pd
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    assert torch.cuda.is_available(), "GPU 필요"
    device, dtype = "cuda", torch.float16

    eval_df = pd.read_csv(args.split)
    print(f"평가셋 {len(eval_df)}개, 모델 {args.model}, 4bit={args.load_4bit}", flush=True)

    # 모델 크기 기준으로 필요한 VRAM을 추정하고 확보될 때까지 대기
    disk_gb = sum(
        os.path.getsize(os.path.join(args.model, f))
        for f in os.listdir(args.model) if f.endswith(".safetensors")
    ) / 1e9
    need_gb = (disk_gb * 0.4 if args.load_4bit else disk_gb) + 1.0
    # 이 GPU(8GB)는 시스템이 ~0.8GB를 상시 점유 -> 실질 최대 여유 ~7.3GB, 상한 7.0
    wait_for_free_vram(min(need_gb, 7.0))

    quant_cfg = None
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=dtype, device_map=device,
        quantization_config=quant_cfg, local_files_only=True,
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"어댑터 적용: {args.adapter}", flush=True)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels, local_files_only=True)
    print("모델 로드 완료", flush=True)

    image_dir = os.path.join(args.data_dir, "train")
    torch.cuda.reset_peak_memory_stats()
    records = []
    t_start = time.time()

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        messages = get_prompt_message(row, image_dir, args.prompt)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

        out_text = processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        pred, parsed = parse_model_output(out_text)
        gt = ast.literal_eval(row["Answer"])
        records.append({
            "Id": row["Id"], "pred": str(pred), "gt": str(gt),
            "correct": pred == gt, "parsed": parsed, "raw": out_text[:80],
        })

    elapsed = time.time() - t_start
    res_df = pd.DataFrame(records)

    # 핵심 지표: 섞인 샘플(No_ordering=False) 정확도 — 무작위 기준선 1/24 = 4.2%
    res_df = res_df.merge(eval_df[["Id", "No_ordering"]], on="Id")
    shuffled = res_df[~res_df["No_ordering"]]
    identity = res_df[res_df["No_ordering"]]

    # 어댑터 평가는 model_id에 run 이름을 붙여 experiments.csv에서 구분한다
    model_tag = args.model
    if args.adapter:
        run_name = os.path.basename(os.path.dirname(args.adapter.rstrip("/\\"))) or "adapter"
        model_tag = f"{args.model}+{run_name}"

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_id": model_tag,
        "load_4bit": args.load_4bit,
        "prompt": args.prompt,
        "max_pixels": args.max_pixels,
        "eval_n": len(res_df),
        "accuracy": round(res_df["correct"].mean(), 4),
        "acc_shuffled": round(shuffled["correct"].mean(), 4) if len(shuffled) else None,
        "acc_identity": round(identity["correct"].mean(), 4) if len(identity) else None,
        "parse_fail": int((~res_df["parsed"]).sum()),
        "sec_per_sample": round(elapsed / len(res_df), 2),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    # 컬럼이 추가돼도 기존 기록과 어긋나지 않게 읽어서 합친 뒤 전체를 다시 쓴다
    new_row = pd.DataFrame([summary])
    if os.path.exists(args.results):
        new_row = pd.concat([pd.read_csv(args.results), new_row], ignore_index=True)
    new_row.to_csv(args.results, index=False)

    model_name = model_tag.rstrip("/").split("/")[-1].replace("+", "_")
    wrong = res_df[~res_df["correct"]].merge(eval_df[["Id", "Sentence"]], on="Id")
    wrong.to_csv(f"./outputs/errors_{model_name}.csv", index=False)

    print(f"완료: 전체 {summary['accuracy']:.3f} | 섞인 샘플 {summary['acc_shuffled']:.3f} "
          f"| identity {summary['acc_identity']:.3f} | 샘플당 {summary['sec_per_sample']}초, "
          f"VRAM {summary['peak_vram_gb']}GB -> {args.results}", flush=True)


if __name__ == "__main__":
    main()
