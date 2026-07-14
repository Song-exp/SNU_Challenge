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

# Windows/캐글 환경 인증서 우회 설정
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

# =========================================================================
# ⚙️ [경로 설정 - 캐글 데이터셋 경로에 맞게 확인하세요]
# =========================================================================
DATA_DIR = "/kaggle/input/datasets/leebyeongcheol/snu-ai-challenge-data/snuaichallenge_data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} 경로에 파일이 없습니다.")
        return
        
    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} samples.")

    # 1. OpenAI CLIP 모델 로드
    print("Loading CLIP model...")
    try:
        clip_model, clip_preprocess = torch.hub.load("openai/CLIP", "ViT_B_32", trust_repo=True)
        clip_model = clip_model.to(device).eval()
        print("CLIP Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load CLIP model: {e}")
        return

    # 2. 고유 이미지 수집
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

    # 4. 전체 쌍의 CLIP 거리 및 Max/Mean 비율 연산
    print("\nCalculating pairwise CLIP distances...")
    ratios = []
    max_clips = []
    mean_clips = []
    ids = []
    
    pairs = list(combinations(range(4), 2))
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing"):
        sample_id = str(row['Id'])
        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in shuffled_files]
        
        clip_dist_list = []
        for p1, p2 in pairs:
            feat1 = embedding_cache.get(img_paths[p1])
            feat2 = embedding_cache.get(img_paths[p2])
            if feat1 is not None and feat2 is not None:
                cos_sim = float(np.dot(feat1, feat2))
                clip_dist = 1.0 - cos_sim
            else:
                clip_dist = 0.0
            clip_dist_list.append(clip_dist)
            
        clip_dist_arr = np.array(clip_dist_list)
        max_clip = np.max(clip_dist_arr)
        mean_clip = np.mean(clip_dist_arr)
        ratio = max_clip / mean_clip if mean_clip > 0 else 1.0
        
        ids.append(row['Id'])
        max_clips.append(max_clip)
        mean_clips.append(mean_clip)
        ratios.append(ratio)
        
    res_df = pd.DataFrame({
        'Id': ids,
        'Max': max_clips,
        'Mean': mean_clips,
        'Ratio': ratios
    })
    
    # 5. 정규화(Z-Score) 수행
    print("\nScaling to normal distribution (Z-score)...")
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
    scaled_data = qt.fit_transform(res_df[['Max', 'Mean']])
    res_df['Max_scaled'] = scaled_data[:, 0]
    res_df['Mean_scaled'] = scaled_data[:, 1]
    
    # 6. 통계 요약표 출력
    print("\n" + "="*75)
    print("📊 [전체 9,535개 비디오 세트 기준] CLIP Max, Mean, Ratio 통계 분포 요약")
    print("="*75)
    print(res_df[['Max', 'Mean', 'Ratio']].describe())
    
    print("\n" + "="*75)
    print("📊 [전체 데이터 기준] 정규화 완료된 Z-score 통계 요약")
    print("="*75)
    print(res_df[['Max_scaled', 'Mean_scaled']].describe())

if __name__ == "__main__":
    main()
