# -*- coding: utf-8 -*-
"""OWL-ViT 기반 Open-Vocabulary 객체 탐지 및 궤적 피처 추출 스크립트.

이 스크립트는 train/test 이미지에 대해 OWL-ViT를 실행하여 
주인공 피사체의 X/Y 중심 좌표와 화면 면적 비율(Area) 트렌드를 오프라인 피처로 추출합니다.
Kaggle GPU 환경 등에서 실행할 경우 CUDA 가속을 지원하여 매우 빠른 속도로 동작합니다.
"""
import os
import ast
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import re

# OpenMP 중복 런타임 방지
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from transformers import OwlViTProcessor, OwlViTForObjectDetection

WORKSPACE_DIR = "./"
TRAIN_CSV = os.path.join(WORKSPACE_DIR, "snuaichallenge_data/train.csv")
TEST_CSV = os.path.join(WORKSPACE_DIR, "snuaichallenge_data/test.csv")
IMAGE_DIR = os.path.join(WORKSPACE_DIR, "snuaichallenge_data")

# 불용어 정의 (기본 명사구 추출용 폴백)
STOPWORDS = {
    "the", "a", "an", "and", "in", "on", "at", "with", "is", "are", "of", "to",
    "holding", "sits", "gets", "water", "using", "front", "behind", "next",
    "man", "woman", "person", "people", "holding", "standing", "sitting", "walking"
}

def extract_candidates(sentence, gemma_subjects=None):
    """문장과 Gemma 라벨의 subjects를 결합하여 OWL-ViT 쿼리 후보 단어 목록을 생성합니다."""
    candidates = []
    if gemma_subjects:
        candidates += [str(s).strip().lower() for s in gemma_subjects if len(str(s).strip()) > 1]
    
    # 폴백: 문장 단어 분해 및 불용어 제거
    words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())
    for w in words:
        if w not in STOPWORDS and w not in candidates:
            candidates.append(w)
            
    # 최소한의 기본값 보장
    if not candidates:
        candidates = ["object", "person"]
        
    return list(set(candidates))[:4] # 과도한 연산 방지 위해 최대 4개로 제한

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"OWL-ViT 피처 추출 시작 | 디바이스: {device}")
    
    # 모델 로드 (로컬 캐시 우선 사용)
    model_name = "./models/owlvit-base-patch32" if os.path.exists("./models/owlvit-base-patch32") else "google/owlvit-base-patch32"
    processor = OwlViTProcessor.from_pretrained(model_name)
    model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)
    model.eval()
    
    # Gemma 라벨 로드 (subjects 후보 획득 목적)
    gemma_subjects_dict = {}
    gemma_path = "./outputs/gemma_labels/labels.jsonl"
    if os.path.exists(gemma_path):
        with open(gemma_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("ok"):
                        gemma_subjects_dict[r["Id"]] = r.get("subjects", [])
                except Exception:
                    continue

    for split in ["train", "test"]:
        csv_path = TRAIN_CSV if split == "train" else TEST_CSV
        if not os.path.exists(csv_path):
            print(f"{split} CSV 파일이 존재하지 않아 스킵합니다: {csv_path}")
            continue
            
        df = pd.read_csv(csv_path)
        print(f"\n{split} 데이터셋 피처 추출 중 ({len(df)}개 샘플)...")
        
        records = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sample_id = str(row["Id"])
            sentence = row["Sentence"]
            
            # 4장의 이미지 (원본 정렬 순서대로)
            shuffled_files = [row["Input_1"], row["Input_2"], row["Input_3"], row["Input_4"]]
            if split == "train":
                ans = ast.literal_eval(row["Answer"])
                ordered_files = [None] * 4
                for i, pos in enumerate(ans):
                    ordered_files[pos - 1] = shuffled_files[i]
            else:
                ordered_files = shuffled_files # Test 셋은 정답이 없으므로 인풋 셔플 순서 그대로 저장 후 추론 시 재매핑
                
            img_paths = [os.path.join(IMAGE_DIR, split, sample_id, f) for f in ordered_files]
            
            # 이미지 파일 실존성 체크
            if not all(os.path.exists(p) for p in img_paths):
                records.append({
                    "Id": sample_id, "query": "none",
                    "coord_1": "skip", "coord_2": "skip", "coord_3": "skip", "coord_4": "skip"
                })
                continue
                
            # 후보 쿼리 리스트
            subjects = gemma_subjects_dict.get(sample_id, [])
            candidates = extract_candidates(sentence, subjects)
            
            # 각 후보별로 4프레임 이미지 로드 및 OWL-ViT 실행하여 최적 쿼리(평균 score 최고) 선정
            best_query = candidates[0]
            best_avg_score = -1.0
            best_results_by_frame = []
            
            images = [Image.open(p).convert("RGB") for p in img_paths]
            
            for query in candidates:
                scores_sum = 0.0
                frames_data = []
                
                for img in images:
                    inputs = processor(text=[[query]], images=img, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        
                    # transformers 버전에 의존하지 않도록 raw outputs에서 직접 [cx, cy, w, h] 추출
                    logits = outputs.logits[0]  # shape: (num_boxes, num_queries)
                    pred_boxes = outputs.pred_boxes[0]  # shape: (num_boxes, 4) [cx, cy, w, h] (0~1 정규화값)
                    
                    scores = torch.sigmoid(logits)[:, 0]  # shape: (num_boxes,)
                    keep = scores >= 0.08
                    
                    if not keep.any():
                        frames_data.append("skip")
                        continue
                        
                    filtered_boxes = pred_boxes[keep].cpu().numpy()
                    filtered_scores = scores[keep].cpu().numpy()
                    
                    # 면적이 가장 큰 BBox를 타깃으로 선택
                    best_idx = 0
                    max_area = 0.0
                    for i, box in enumerate(filtered_boxes):
                        cx, cy, w, h = box
                        area = w * h
                        if area > max_area:
                            max_area = area
                            best_idx = i
                            
                    cx, cy, w, h = filtered_boxes[best_idx]
                    best_score = filtered_scores[best_idx]
                    scores_sum += best_score
                    
                    frames_data.append(f"X={cx:.3f}, Y={cy:.3f}, Area={max_area*100:.1f}%")
                    
                avg_score = scores_sum / 4.0
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_query = query
                    best_results_by_frame = frames_data
                    
            records.append({
                "Id": sample_id,
                "query": best_query,
                "coord_1": best_results_by_frame[0],
                "coord_2": best_results_by_frame[1],
                "coord_3": best_results_by_frame[2],
                "coord_4": best_results_by_frame[3]
            })
            
        out_df = pd.DataFrame(records)
        out_name = f"./snu_owlvit_features_{split}.csv"
        out_df.to_csv(out_name, index=False)
        print(f"피처 추출 완료 ➔ 저장 위치: {out_name}")

if __name__ == "__main__":
    main()
