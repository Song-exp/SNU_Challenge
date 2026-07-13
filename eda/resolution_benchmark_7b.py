import os
import ast
import time
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# =========================================================================
# [1. 설정 및 경로 설정]
# =========================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 7B 모델 지정
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

# 업로드하신 stratified_valid.csv 경로
STRATIFIED_CSV = '../input/datasets/leebyeongcheol/dsadasdasf/stratified_valid.csv' 
valid_df = pd.read_csv(STRATIFIED_CSV)

# 원본 이미지 디렉토리 경로 (캐글 내부 경로 확인 필요)
TRAIN_IMAGE_DIR = '../input/datasets/leebyeongcheol/snu-ai-challenge-data/snuaichallenge_data/train'

# 실험할 해상도 그리드 설정 (28픽셀 배수)
grid_experiments = {
    "Grid_1 (224px)": {"min": 56 * 28 * 28, "max": 112 * 28 * 28},
    "Grid_2 (336px)": {"min": 84 * 28 * 28, "max": 252 * 28 * 28},
    "Grid_3 (448px)": {"min": 112 * 28 * 28, "max": 448 * 28 * 28},
    "Grid_4 (560px)": {"min": 140 * 28 * 28, "max": 700 * 28 * 28}
}

# -------------------------------------------------------------------------
# [2. 모델 및 프로세서 로드 (7B bfloat16 최적화)]
# -------------------------------------------------------------------------
print(f"Loading {MODEL_NAME} with mixed precision...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# 파싱 함수 (기본값 설정 포함)
def parse_model_output(output_text):
    try:
        start_idx = output_text.find('[')
        end_idx = output_text.find(']') + 1
        if start_idx != -1 and end_idx != -1:
            return ast.literal_eval(output_text[start_idx:end_idx])
    except:
        pass
    return [1, 2, 3, 4]

# 최종 결과를 모아둘 테이블
experiment_records = []

# =========================================================================
# [3. 그리드 해상도별 순차 벤치마크 실행]
# =========================================================================
for grid_id, pixels in grid_experiments.items():
    print(f"\n🚀 {grid_id} 벤치마크 시작 (총 {len(valid_df)}개 샘플)")
    
    # 공정한 메모리 측정을 위한 캐시 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    start_time = time.time()
    results = []

    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc=f"{grid_id} 추론 중"):
        min_p = pixels["min"]
        max_p = pixels["max"]
        
        img_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        sentence = row['Sentence']

        content = []
        for i, img_file in enumerate(img_files):
            img_path = os.path.join(TRAIN_IMAGE_DIR, row['Id'], img_file)
            content.append({
                "type": "image",
                "image": img_path,
                "min_pixels": min_p,
                "max_pixels": max_p
            })
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
        
        # apply_chat_template & process_vision_info
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        pred_list = parse_model_output(output_text)
        actual_list = ast.literal_eval(row['Answer']) 
        
        # 각 샘플별 정답 여부를 기록
        is_correct = 1 if pred_list == actual_list else 0
        results.append({
            "Id": row['Id'],
            "Type": row['Type'],  # 'Fine-grained' 또는 'Scene Cut'
            "Is_Correct": is_correct
        })
        
    # 시간 및 자원 측정
    total_duration = time.time() - start_time
    sec_per_sample = total_duration / len(valid_df)
    vram_used = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    
    # ----------------------------------------------------------------
    # 📊 서브셋(타입)별 성능 분석 및 누적
    # ----------------------------------------------------------------
    res_df = pd.DataFrame(results)
    
    # 1. 전체(Aggregate) 정확도
    total_acc = res_df['Is_Correct'].mean() * 100
    
    # 2. 타입별 정확도 분할 계산
    type_summary = res_df.groupby('Type')['Is_Correct'].mean() * 100
    fine_acc = type_summary.get('Fine-grained', 0.0)
    scene_acc = type_summary.get('Scene Cut', 0.0)
    
    # 결과 수집
    experiment_records.append({
        "실험 ID": grid_id,
        "전체 정확도 (%)": f"{total_acc:.2f}%",
        "미세행동 정확도 (%)": f"{fine_acc:.2f}%",
        "장면전환 정확도 (%)": f"{scene_acc:.2f}%",
        "추론 속도 (초/샘플)": f"{sec_per_sample:.4f}s",
        "VRAM 사용량": f"{vram_used:.2f} GB"
    })

# =========================================================================
# [4. 최종 벤치마크 결과 스코어보드 출력]
# =========================================================================
print("\n" + "="*80)
print("📊 [최종 결과] 층화 검증셋 기반 해상도별 벤치마크 스코어보드")
print("="*80)
summary_df = pd.DataFrame(experiment_records)
print(summary_df.to_string(index=False))
print("="*80 + "\n")
