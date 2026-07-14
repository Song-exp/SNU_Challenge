import os
import ast
import ssl
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from itertools import combinations

# Force stdout to UTF-8 to prevent cp949 encode errors in Windows
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Windows/캐글 환경 우회
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

# =========================================================================
# ⚙️ [경로 설정 - 캐글 데이터셋 경로에 맞게 확인하세요]
# =========================================================================
DATA_DIR = "/kaggle/input/datasets/leebyeongcheol/snu-ai-challenge-data/snuaichallenge_data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train")

# 기준선(Threshold) 세팅
CLIP_THRESHOLD = 0.20
MSE_THRESHOLD = 1200

def compute_image_mse(img_path1, img_path2):
    try:
        with Image.open(img_path1).convert('L') as img1, Image.open(img_path2).convert('L') as img2:
            img1_r = img1.resize((64, 64), Image.Resampling.BILINEAR)
            img2_r = img2.resize((64, 64), Image.Resampling.BILINEAR)
            arr1 = np.array(img1_r, dtype=np.float32)
            arr2 = np.array(img2_r, dtype=np.float32)
            return float(np.mean((arr1 - arr2) ** 2))
    except Exception:
        return 0.0

def map_similar_pairs_to_cuts(similar_pairs):
    # 유사쌍 개수 -> 장면 전환 횟수 매핑 법칙
    if similar_pairs >= 5:
        return 0  # 0회 전환 (동일 장면)
    elif 2 <= similar_pairs <= 4:
        return 1  # 1회 전환 (2개 씬)
    elif similar_pairs == 1:
        return 2  # 2회 전환 (3개 씬)
    else:
        return 3  # 3회 전환 (매 프레임 전환)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} 경로에 파일이 없습니다.")
        return
        
    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} samples.")

    # 1. CLIP 모델 로드
    print("Loading CLIP model...")
    try:
        clip_model, clip_preprocess = torch.hub.load("openai/CLIP", "ViT_B_32", trust_repo=True)
        clip_model = clip_model.to(device).eval()
        print("CLIP Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load CLIP model: {e}")
        return

    # 2. 이미지 수집
    print("Collecting image paths...")
    image_paths = []
    for _, row in df.iterrows():
        sample_id = str(row['Id'])
        for f in [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]:
            image_paths.append(os.path.join(IMAGE_DIR, sample_id, f))
    unique_paths = list(set(image_paths))

    # 3. GPU 배치 연산으로 CLIP 임베딩 추출
    embedding_cache = {}
    batch_size = 128
    for i in tqdm(range(0, len(unique_paths), batch_size), desc="CLIP Inference"):
        batch_paths = unique_paths[i:i+batch_size]
        batch_imgs = []
        valid_paths = []
        for p in batch_paths:
            if os.path.exists(p):
                try:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(clip_preprocess(img))
                    valid_paths.append(p)
                except:
                    pass
        if not batch_imgs:
            continue
        img_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            features = clip_model.encode_image(img_tensor)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            features_cpu = features.cpu().numpy()
        for path, feat in zip(valid_paths, features_cpu):
            embedding_cache[path] = feat

    # 4. CLIP 유사도 및 MSE 물리 오차 동시 분석
    print("\nAnalyzing dataset using CLIP & MSE...")
    ids = []
    clip_cuts = []
    mse_cuts = []
    
    pairs = list(combinations(range(4), 2))
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Cuts"):
        sample_id = str(row['Id'])
        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in shuffled_files]
        
        # 6개 조합에 대해 계산
        clip_dist_list = []
        mse_list = []
        
        for p1, p2 in pairs:
            # 픽셀 MSE 계산
            mse_val = compute_image_mse(img_paths[p1], img_paths[p2])
            mse_list.append(mse_val)
            
            # CLIP 거리 계산
            feat1 = embedding_cache.get(img_paths[p1])
            feat2 = embedding_cache.get(img_paths[p2])
            if feat1 is not None and feat2 is not None:
                cos_sim = float(np.dot(feat1, feat2))
                clip_dist = 1.0 - cos_sim
            else:
                clip_dist = 0.0
            clip_dist_list.append(clip_dist)
            
        # 임계값 필터 통과한 유사쌍 카운트
        similar_clip_pairs = sum([1 for c in clip_dist_list if c < CLIP_THRESHOLD])
        similar_mse_pairs = sum([1 for m in mse_list if m < MSE_THRESHOLD])
        
        # 장면 전환 횟수로 매핑
        clip_cut_cnt = map_similar_pairs_to_cuts(similar_clip_pairs)
        mse_cut_cnt = map_similar_pairs_to_cuts(similar_mse_pairs)
        
        ids.append(row['Id'])
        clip_cuts.append(clip_cut_cnt)
        mse_cuts.append(mse_cut_cnt)
        
    analysis_df = pd.DataFrame({
        'Id': ids,
        'clip_cuts': clip_cuts,
        'mse_cuts': mse_cuts
    })
    
    # 5. 결과 분포 비교 대조표 출력
    print("\n" + "="*70)
    print("📊 [의미적 vs 물리적] 장면 전환 횟수(0~3회) 분포 비교 대조표")
    print("="*70)
    
    clip_counts = analysis_df['clip_cuts'].value_counts().sort_index()
    mse_counts = analysis_df['mse_cuts'].value_counts().sort_index()
    
    compare_df = pd.DataFrame({
        'CLIP Cuts (Count)': clip_counts,
        'CLIP Cuts (%)': (clip_counts / len(df) * 100).round(2),
        'MSE Cuts (Count)': mse_counts,
        'MSE Cuts (%)': (mse_counts / len(df) * 100).round(2)
    })
    print(compare_df.to_string())
    
    # 6. CSV 파일로 백업 저장
    OUTPUT_FILE = "snu_clip_mse_cuts_analysis.csv"
    analysis_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 분석 완료 데이터 저장 성공: {OUTPUT_FILE}로 저장되었습니다.")

if __name__ == "__main__":
    main()
