import os
import re
import ast
import pandas as pd
import numpy as np
from PIL import Image

# OpenMP Duplicate Runtime Workaround for Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

WORKSPACE_DIR = "C:/Users/user/Desktop/서울대"
TRAIN_CSV = os.path.join(WORKSPACE_DIR, "train_검토_최종_완료.csv")
IMAGE_DIR = os.path.join(WORKSPACE_DIR, "snuaichallenge_data/train")

def filter_camera_samples(df):
    """
    캡션에 카메라 기법, 줌, 거리 변화 등이 명시된 샘플들을 필터링합니다.
    """
    keywords = [
        r"\bzoom(s)?\b", r"\bpan(s)?\b", r"\bshot(s)?\b", r"\bcloser\b", 
        r"\bfurther\b", r"\bcamera\b", r"\bdistance\b", r"\bshifts\s+close\b",
        r"\bmoves\s+close\b", r"\bmoves\s+in\b", r"\bmoves\s+out\b"
    ]
    pattern = "|".join(keywords)
    
    mask = df['Sentence'].str.lower().str.contains(pattern, regex=True, na=False)
    return df[mask].copy()

def analyze_bbox_and_verify(df_samples, limit=50):
    """
    필터링된 샘플들에 대해 Faster R-CNN을 돌려 bbox 면적 변화율과 실제 정답 순서(Answer)가 
    인과적으로 일치하는지 분석합니다.
    """
    print(f"\n--- torchvision Faster R-CNN 기반 BBox 면적 변화 검증 시작 (샘플 상한: {limit}개) ---")
    
    # Pre-trained Faster R-CNN 모델 로드
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.4).to("cpu")
    model.eval()
    
    preprocess = weights.transforms()
    
    results_summary = []
    
    count = 0
    for idx, row in df_samples.iterrows():
        if count >= limit:
            break
            
        sample_id = str(row['Id'])
        sentence = row['Sentence']
        ans = ast.literal_eval(row['Answer'])
        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        
        # 1. 정답 시간 순서대로 이미지 파일 재정렬
        ordered_files = [None] * 4
        for i, pos in enumerate(ans):
            ordered_files[pos - 1] = shuffled_files[i]
            
        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in ordered_files]
        
        # Check files exist
        if not all(os.path.exists(p) for p in img_paths):
            continue
            
        # 2. 각 이미지의 객체(person 또는 주요 객체) 면적 추출
        areas = []
        for img_path in img_paths:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            img_area = w * h
            
            # Preprocess and infer
            input_tensor = preprocess(img).unsqueeze(0).to("cpu")
            with torch.no_grad():
                predictions = model(input_tensor)[0]
                
            boxes = predictions['boxes'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            
            max_box_area = 0.0
            for box, label, score in zip(boxes, labels, scores):
                box_w = box[2] - box[0]
                box_h = box[3] - box[1]
                box_area = (box_w * box_h) / img_area
                
                # 이미지 내의 가장 큰 객체 면적 추적
                if box_area > max_box_area:
                    max_box_area = box_area
                    
            areas.append(max_box_area)
            
        # 3. 면적 트렌드 분석
        # 캡션에 줌 단서 분석
        has_zoom_in = any(w in sentence.lower() for w in ["closer", "shifts close", "moves in"])
        has_zoom_out = any(w in sentence.lower() for w in ["further", "zooms out", "moves out"])
        
        status = "미매치(일관성 없음)"
        if has_zoom_in and (areas[-1] > areas[0]):
            status = "매치 (Zoom-In / 면적 증가)"
        elif has_zoom_out and (areas[-1] < areas[0]):
            status = "매치 (Zoom-Out / 면적 감소)"
        elif not has_zoom_in and not has_zoom_out:
            status = f"중립 (변화율: {areas[0]:.2f} -> {areas[-1]:.2f})"
            
        results_summary.append({
            'Id': sample_id,
            'Sentence': sentence,
            'Areas': [round(a, 3) for a in areas],
            'Status': status
        })
        count += 1
        if count % 5 == 0:
            print(f"진행 상황: {count}/{limit} 완료...")
        
    df_res = pd.DataFrame(results_summary)
    
    match_count = sum(1 for r in results_summary if "매치" in r['Status'])
    neutral_count = sum(1 for r in results_summary if "중립" in r['Status'])
    mismatch_count = len(results_summary) - match_count - neutral_count
    
    print("\n=== 최종 50개 검증 통계 결과 ===")
    print(f"전체 평가 개수: {len(results_summary)}")
    print(f"매치 성공: {match_count}개 ({match_count/len(results_summary)*100:.1f}%)")
    print(f"중립 (키워드 없음): {neutral_count}개 ({neutral_count/len(results_summary)*100:.1f}%)")
    print(f"미매치 (불일치): {mismatch_count}개 ({mismatch_count/len(results_summary)*100:.1f}%)")

def main():
    if not os.path.exists(TRAIN_CSV):
        print(f"원본 CSV 파일이 없습니다: {TRAIN_CSV}")
        return
        
    df = pd.read_csv(TRAIN_CSV, encoding='cp949')
    
    # 1. 카메라 기법 필터링
    df_cam = filter_camera_samples(df)
    print(f"전체 {len(df)}개 중 카메라 기법 관련 문장 개수: {len(df_cam)}개")
    
    # 2. Faster R-CNN 기반 50개 대용량 검증 시작
    analyze_bbox_and_verify(df_cam, limit=50)

if __name__ == "__main__":
    main()
