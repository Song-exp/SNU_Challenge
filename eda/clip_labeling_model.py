import os
import ast
import ssl
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from itertools import combinations
from sklearn.preprocessing import QuantileTransformer

# Force stdout to UTF-8 to prevent cp949 encode errors in Windows
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Windows/캐글 환경 인증서 우회
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

# =========================================================================
# ⚙️ [경로 설정 - 캐글 데이터셋 경로에 맞게 확인하세요]
# =========================================================================
DATA_DIR = "/kaggle/input/datasets/leebyeongcheol/snu-ai-challenge-data/snuaichallenge_data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train")

# 장면 전환 판단 임계치
CLIP_THRESHOLD = 0.20

def map_similar_pairs_to_cuts(similar_clip_pairs, max_clip):
    # 엄격 기준 적용: Max 오차가 0.20 미만이면 무조건 0회 장면 전환 (동일 씬)
    if max_clip < CLIP_THRESHOLD:
        return 0
    
    if 2 <= similar_clip_pairs <= 5:
        return 1  # 1회 장면 전환 (2개 씬 분할)
    elif similar_clip_pairs == 1:
        return 2  # 2회 장면 전환 (3개 씬 분할)
    else:
        return 3  # 3회 장면 전환 (4개 씬 모두 다른 장면)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} 경로에 파일이 없습니다.")
        return
        
    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} samples.")

    # 1. OpenAI CLIP 모델 로드
    print("Loading CLIP model via Torch Hub...")
    try:
        clip_model, clip_preprocess = torch.hub.load("openai/CLIP", "ViT_B_32", trust_repo=True)
        clip_model = clip_model.to(device).eval()
        print("CLIP Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load CLIP model: {e}")
        return

    # 2. 이미지 고유 경로 수집
    print("Collecting unique image paths...")
    image_paths = []
    for _, row in df.iterrows():
        sample_id = str(row['Id'])
        for f in [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]:
            image_paths.append(os.path.join(IMAGE_DIR, sample_id, f))
    unique_paths = list(set(image_paths))
    print(f"Total unique images to extract: {len(unique_paths):,}")

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

    # 4. CLIP 장면 분석 루프 (전체 데이터 대상)
    print("\nExtracting CLIP-based features and predicted cuts...")
    
    ids = []
    dist_12, dist_13, dist_14, dist_23, dist_24, dist_34 = [], [], [], [], [], []
    clip_max_list, clip_mean_list, ratios = [], [], []
    predicted_scene_cuts = []
    
    pairs = list(combinations(range(4), 2))
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
        sample_id = str(row['Id'])
        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in shuffled_files]
        
        feats = [embedding_cache.get(p) for p in img_paths]
        
        def get_dist(f1, f2):
            if f1 is not None and f2 is not None:
                return float(1.0 - np.dot(f1, f2))
            return 0.0
            
        c12 = get_dist(feats[0], feats[1])
        c13 = get_dist(feats[0], feats[2])
        c14 = get_dist(feats[0], feats[3])
        c23 = get_dist(feats[1], feats[2])
        c24 = get_dist(feats[1], feats[3])
        c34 = get_dist(feats[2], feats[3])
        
        clip_dist_arr = np.array([c12, c13, c14, c23, c24, c34])
        max_clip = np.max(clip_dist_arr)
        mean_clip = np.mean(clip_dist_arr)
        ratio = max_clip / mean_clip if mean_clip > 0 else 1.0
        
        similar_clip_pairs = sum([1 for c in clip_dist_arr if c < CLIP_THRESHOLD])
        cuts = map_similar_pairs_to_cuts(similar_clip_pairs, max_clip)
        
        ids.append(row['Id'])
        dist_12.append(c12); dist_13.append(c13); dist_14.append(c14)
        dist_23.append(c23); dist_24.append(c24); dist_34.append(c34)
        
        clip_max_list.append(max_clip)
        clip_mean_list.append(mean_clip)
        ratios.append(ratio)
        predicted_scene_cuts.append(cuts)
        
    res_df = pd.DataFrame({
        'Id': ids,
        'predicted_scene_cuts': predicted_scene_cuts,
        'Max': clip_max_list,
        'Mean': clip_mean_list,
        'Ratio': ratios,
        'dist_12': dist_12, 'dist_13': dist_13, 'dist_14': dist_14,
        'dist_23': dist_23, 'dist_24': dist_24, 'dist_34': dist_34
    })
    
    # 5. 정규분포(Z-Score) 변환기 적용
    print("\nScaling to normal distribution (Z-score)...")
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
    scaled_data = qt.fit_transform(res_df[['Max', 'Mean']])
    res_df['Max_scaled'] = scaled_data[:, 0]
    res_df['Mean_scaled'] = scaled_data[:, 1]
    
    # 6. 전수조사 요약 통계 출력
    print("\n" + "="*80)
    print("📊 [9,535개 비디오 전수조사] 엄격 기준 CLIP 장면 전환 분석 결과")
    print("="*80)
    
    cut_counts = res_df['predicted_scene_cuts'].value_counts().sort_index()
    total_samples = len(res_df)
    for cuts, count in cut_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  🎬 장면 전환 {cuts}회 비디오: {count:5d}개 ({percentage:.2f}%)")
    print("-" * 80)
    print(f"👉 총 분석 비디오 세트 수  : {total_samples:5d}개 (100.00%)")
        
    # 7. 최종 CSV 저장
    OUTPUT_FILE = "snu_clip_features.csv"
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 전체 데이터 피처 CSV 저장 성공: '{OUTPUT_FILE}'로 저장 완료!")

if __name__ == "__main__":
    main()
