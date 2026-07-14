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

# 장면 전환 판단 임계치 (0.20 미만은 동일장면, 0.20 이상은 장면 전환)
CLIP_THRESHOLD = 0.20

def map_similar_pairs_to_cuts(similar_pairs):
    # 6개 조합 중 서로 유사한(0.20 미만) 이미지 쌍 개수에 따른 장면 전환 횟수 매핑
    if similar_pairs >= 5:
        return 0  # 0회 전환 (4장 모두 동일 장면)
    elif 2 <= similar_pairs <= 4:
        return 1  # 1회 전환 (두 그룹으로 나뉨)
    elif similar_pairs == 1:
        return 2  # 2회 전환 (세 그룹으로 나뉨)
    else:
        return 3  # 3회 전환 (네 프레임 모두 다른 장면)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} 경로에 파일이 없습니다.")
        return
        
    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} samples from train.csv.")

    # 1. CLIP 모델 로드
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

    # 3. 임베딩 추출 (GPU 배치 연산)
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

    # 4. 각 비디오별 CLIP 기반 장면 전환 횟수(0~3회) 연산
    print("\nClassifying video scene cut counts based on CLIP distance...")
    ids = []
    predicted_cuts = []
    similar_pairs_list = []
    
    pairs = list(combinations(range(4), 2))
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing cuts"):
        sample_id = str(row['Id'])
        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in shuffled_files]
        
        # 6개 조합 코사인 거리 계산
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
            
        # 0.20 미만(유사 프레임) 개수 구하기
        similar_pairs = sum([1 for c in clip_dist_list if c < CLIP_THRESHOLD])
        
        # 컷 개수 맵핑
        cuts = map_similar_pairs_to_cuts(similar_pairs)
        
        ids.append(row['Id'])
        predicted_cuts.append(cuts)
        similar_pairs_list.append(similar_pairs)
        
    analysis_df = pd.DataFrame({
        'Id': ids,
        'similar_pairs_count': similar_pairs_list,
        'predicted_cuts': predicted_cuts
    })
    
    # 5. 전수 데이터 기준 CLIP 장면 전환 횟수(0~3회) 분포 결과 출력
    print("\n" + "="*70)
    print("📊 [전수 데이터 기준] CLIP 기반 장면 전환 횟수(0~3회) 분포 통계")
    print("="*70)
    
    cut_counts = analysis_df['predicted_cuts'].value_counts().sort_index()
    total_samples = len(analysis_df)
    
    for cuts, count in cut_counts.items():
        percentage = (count / total_samples) * 100
        print(f"🎬 장면 전환 {cuts}회 비디오: {count:5d}개 ({percentage:5.2f}%)")
    print("-" * 70)
    print(f"👉 총 분석 비디오 세트 수  : {total_samples:5d}개 (100.00%)")
    
    # 6. CSV 파일 저장 (Kaggle Output 다운로드용)
    OUTPUT_FILE = "snu_clip_predicted_cuts.csv"
    analysis_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 분석 완료 데이터 저장 성공: {OUTPUT_FILE}로 저장되었습니다.")

if __name__ == "__main__":
    main()
