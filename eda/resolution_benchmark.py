"""
SNU AI Challenge - 최적 해상도 탐색 벤치마크 (조장 GPU 실행용)
담당: 이미지 파트 (팀원 1) 설계 / 조장 실행

=== 실행 전 꼭 확인/수정할 것 (아래 3개 경로) ===
  1) STRATIFIED_CSV : 층화추출 검증셋 csv 경로 (Type 컬럼: Fine-grained / Scene Cut)
  2) TRAIN_IMAGE_DIR : train 이미지 폴더 경로
  3) OUTPUT_CSV      : 결과 저장할 경로
전부 하단 argparse 인자로 넘기면 되고, 아무 인자도 안 주면 같은 폴더 기준
상대경로 기본값으로 동작합니다.

=== 지난 실험 대비 바뀐 점 (중요) ===
  - 기존 코드는 Qwen 프로세서의 min_pixels/max_pixels "힌트"만 줬는데, 이 경우
    원본이 640x360(230,400px)인 프레임은 Grid_3(87,808~351,232)와
    Grid_4(109,760~548,800) 양쪽 다 리사이즈가 아예 안 일어나서 두 그리드가
    사실상 "같은 이미지"를 비교하고 있었음.
  - 이번 버전은 PIL로 각 그리드의 target_long_side에 맞춰 실제로 리사이즈한
    이미지를 만들어서 넣기 때문에, 그리드마다 실제로 다른 해상도가 들어가는 게
    보장됨. (min_pixels=max_pixels로 고정해 프로세서가 추가로 손대지 못하게 함)
  - Type(Fine-grained/Scene Cut)별 정확도 + 95% 신뢰구간(정규근사) 같이 출력.
    Fine-grained 표본 수가 10개 미만이면 경고 출력 (숫자 신뢰하지 말 것).
  - 실제로 각 그리드에서 리사이즈된 해상도를 첫 샘플에 한해 로그로 찍어서
    "진짜 그리드별로 다른 해상도가 들어갔는지" 눈으로 확인 가능.

=== 실행 예시 ===
python resolution_benchmark.py \
  --stratified_csv ./stratified_valid.csv \
  --image_dir ./snuaichallenge_data/train \
  --output_csv ./resolution_benchmark_result.csv
"""
import os
import ast
import time
import glob
import math
import argparse

import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    p.add_argument("--stratified_csv", type=str, default="./stratified_valid.csv",
                   help="층화추출 검증셋 csv (Id, Sentence, Answer, Type 컬럼 필요)")
    p.add_argument("--image_dir", type=str, default="./snuaichallenge_data/train",
                   help="train 이미지 폴더 (Id별 하위 폴더 구조)")
    p.add_argument("--output_csv", type=str, default="./resolution_benchmark_result.csv")
    p.add_argument("--tmp_resize_dir", type=str, default="./tmp_resolution_grid_cache")
    p.add_argument("--max_new_tokens", type=int, default=64)
    # 그리드: 실제 장축(long side) 픽셀 기준. 640x360 원본(장축 640) 대비
    # 확실히 낮은 값부터 원본과 비슷한 값, 그 이상까지 포함해야 실제 차이가 보임.
    p.add_argument("--grid_long_sides", type=int, nargs="+",
                   default=[224, 320, 448, 640, 800],
                   help="테스트할 장축 픽셀 후보 목록 (640=원본 그대로, 800=업스케일 비교용)")
    return p.parse_args()


def resize_to_long_side(img_path: str, long_side: int, cache_dir: str) -> str:
    """장축 기준 정확히 리사이즈해서 캐시에 저장하고 경로 반환.
    이미 캐시에 있으면 재사용 (그리드 재실행 시 속도 절약)."""
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.basename(img_path)
    dst_path = os.path.join(cache_dir, fname)
    if os.path.exists(dst_path):
        return dst_path

    with Image.open(img_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        scale = long_side / max(w, h)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img.save(dst_path, "JPEG", quality=95)
    return dst_path


def parse_and_convert_output(output_text: str):
    """모델 출력에서 [a,b,c,d] 파싱 후 submission 포맷으로 역변환. 실패 시 fallback."""
    try:
        start_idx = output_text.find('[')
        end_idx = output_text.rfind(']') + 1
        if start_idx != -1 and end_idx != -1:
            raw_list = ast.literal_eval(output_text[start_idx:end_idx])
            if isinstance(raw_list, list) and len(raw_list) == 4 and sorted(raw_list) == [1, 2, 3, 4]:
                submission_format = [0] * 4
                for index, image_num in enumerate(raw_list):
                    submission_format[image_num - 1] = index + 1
                return submission_format
    except Exception:
        pass
    return [1, 2, 3, 4]


def wilson_ci(correct: int, total: int, z: float = 1.96):
    """이항 비율의 95% 신뢰구간 (Wilson score interval). 표본이 작을 때 정규근사보다 안정적."""
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    margin = (z * math.sqrt((p * (1 - p) / total) + (z ** 2 / (4 * total ** 2)))) / denom
    return (max(0.0, center - margin) * 100, min(1.0, center + margin) * 100)


def main():
    args = parse_args()

    if not os.path.exists(args.stratified_csv):
        raise FileNotFoundError(f"층화 검증셋 csv를 찾을 수 없습니다: {args.stratified_csv}")
    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"이미지 폴더를 찾을 수 없습니다: {args.image_dir}")

    valid_df = pd.read_csv(args.stratified_csv)
    print(f"검증셋 로드: {len(valid_df)}개 샘플")

    if "Type" in valid_df.columns:
        type_counts = valid_df["Type"].value_counts()
        print("Type별 샘플 수:")
        print(type_counts.to_string())
        fine_count = type_counts.get("Fine-grained", 0)
        if fine_count < 10:
            print(f"⚠️  경고: Fine-grained 샘플이 {fine_count}개뿐입니다. "
                  f"이 그룹의 정확도는 표본이 너무 작아 신뢰하기 어렵습니다.")
    else:
        print("⚠️  'Type' 컬럼이 없어 그룹별 분석을 건너뜁니다.")

    print(f"\n모델 로드 중: {args.model_name}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    experiment_records = []

    for long_side in args.grid_long_sides:
        grid_id = f"long_side_{long_side}px"
        print(f"\n🚀 {grid_id} 벤치마크 시작 (총 {len(valid_df)}개 샘플)")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        cache_dir = os.path.join(args.tmp_resize_dir, f"long_{long_side}")
        start_time = time.time()
        results = []
        logged_resolution = False

        for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc=f"{grid_id} 추론 중"):
            sentence = row["Sentence"]
            sample_path = os.path.join(args.image_dir, str(row["Id"]))

            img_files = sorted(glob.glob(os.path.join(sample_path, "*.jpg")))
            if len(img_files) != 4:
                continue

            resized_paths = [resize_to_long_side(p, long_side, cache_dir) for p in img_files]

            if not logged_resolution:
                with Image.open(resized_paths[0]) as sample_img:
                    print(f"  ↳ 실제 리사이즈 확인: {sample_img.size} (요청 장축: {long_side}px)")
                logged_resolution = True

            content = []
            for i, img_path in enumerate(resized_paths):
                content.append({"type": "image", "image": img_path})
                content.append({"type": "text", "text": f"\nImage {i+1}\n"})

            user_text = (
                f"Thinking about the sentence: \"{sentence}\"\n"
                "Look at the 4 images above labeled Image 1 to Image 4. "
                "Determine the correct chronological order of these images to match the sentence. "
                "Provide the answer ONLY as a Python list of integers. "
                "Example: [1, 2, 3, 4]"
            )
            content.append({"type": "text", "text": user_text})
            messages = [{"role": "user", "content": content}]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            pred_list = parse_and_convert_output(output_text)
            actual_list = ast.literal_eval(row["Answer"])
            is_correct = 1 if pred_list == actual_list else 0

            results.append({
                "Id": row["Id"],
                "Type": row.get("Type", "Unknown"),
                "Is_Correct": is_correct,
            })

        if not results:
            print(f"⚠️ {grid_id}에 연산된 데이터가 없습니다. 이미지 경로를 다시 점검해 주세요.")
            continue

        total_duration = time.time() - start_time
        sec_per_sample = total_duration / len(results)
        vram_used = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        res_df = pd.DataFrame(results)
        n_total = len(res_df)
        n_correct = int(res_df["Is_Correct"].sum())
        total_acc = n_correct / n_total * 100
        ci_low, ci_high = wilson_ci(n_correct, n_total)

        type_summary = {}
        for t, group in res_df.groupby("Type"):
            n = len(group)
            c = int(group["Is_Correct"].sum())
            acc = c / n * 100 if n else 0.0
            lo, hi = wilson_ci(c, n)
            type_summary[t] = f"{acc:.2f}% (n={n}, 95%CI [{lo:.1f}, {hi:.1f}])"

        experiment_records.append({
            "실험 ID": grid_id,
            "전체 정확도": f"{total_acc:.2f}% (95%CI [{ci_low:.1f}, {ci_high:.1f}])",
            "미세행동(Fine-grained)": type_summary.get("Fine-grained", "n=0"),
            "장면전환(Scene Cut)": type_summary.get("Scene Cut", "n=0"),
            "추론 속도(초/샘플)": f"{sec_per_sample:.4f}s",
            "VRAM 사용량": f"{vram_used:.2f} GB",
        })

    print("\n" + "=" * 100)
    print("📊 [최종 결과] 실제 리사이즈 기반 해상도별 벤치마크 스코어보드")
    print("=" * 100)
    summary_df = pd.DataFrame(experiment_records)
    print(summary_df.to_string(index=False))
    print("=" * 100)

    summary_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"\n결과 저장 완료: {args.output_csv}")


if __name__ == "__main__":
    main()
