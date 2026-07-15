import os
import ast
import ssl
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
import torch

# =========================================================================
# ⚙️ [경로 설정 - 캐글 데이터셋 경로에 맞게 확인하세요]
# =========================================================================
DATA_DIR = "/kaggle/input/datasets/leebyeongcheol/snu-ai-challenge-data/snuaichallenge_data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train")

# 장면 전환 임계 기준 설정
CLIP_THRESHOLD = 0.20

def map_similar_pairs_to_cuts(similar_clip_pairs, max_clip):
    # 안엄격 기준 적용: 유사쌍 개수 5개 이상이면 0회 장면 전환 (Type 0)
    if similar_clip_pairs >= 5:
        return 0
    
    if 2 <= similar_clip_pairs <= 4:
        return 1  # 1회 장면 전환 (2개 씬 분할)
    elif similar_clip_pairs == 1:
        return 2  # 2회 장면 전환 (3개 씬 분할)
    else:
        return 3  # 3회 장면 전환 (4개 씬 모두 다른 장면)

def calculate_mse(img1, img2):
    # 320x180 해상도로 리사이즈하여 물리적 픽셀 오차 계산
    img1_res = img1.resize((320, 180), Image.Resampling.LANCZOS)
    img2_res = img2.resize((320, 180), Image.Resampling.LANCZOS)
    a1 = np.array(img1_res, dtype=np.float32)
    a2 = np.array(img2_res, dtype=np.float32)
    return float(np.mean((a1 - a2) ** 2))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(TRAIN_CSV):
        print(f"Error: {TRAIN_CSV} 경로에 파일이 없습니다.")
        return
        
    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} samples from train.csv.")

    # 1. CLIP 모델 로드
    print("Loading CLIP model (openai/CLIP) via Torch Hub...")
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

    # 4. CLIP 및 MSE 거리 동시 계산
    print("\nCalculating CLIP and MSE distances for all video sets...")
    
    ids = []
    predicted_scene_cuts = []
    
    # CLIP lists
    clip_max_list, clip_mean_list, clip_ratios = [], [], []
    c12_list, c13_list, c14_list, c23_list, c24_list, c34_list = [], [], [], [], [], []
    
    # MSE lists
    mse_max_list, mse_mean_list, mse_ratios = [], [], []
    m12_list, m13_list, m14_list, m23_list, m24_list, m34_list = [], [], [], [], [], []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        sample_id = str(row['Id'])
        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in shuffled_files]
        
        # PIL 이미지들 로드 (MSE용)
        images = []
        for p in img_paths:
            if os.path.exists(p):
                try:
                    images.append(Image.open(p).convert("RGB"))
                except:
                    images.append(None)
            else:
                images.append(None)
                
        feats = [embedding_cache.get(p) for p in img_paths]
        
        # CLIP 계산
        def get_clip_dist(f1, f2):
            if f1 is not None and f2 is not None:
                return float(1.0 - np.dot(f1, f2))
            return 0.0
            
        c12 = get_clip_dist(feats[0], feats[1])
        c13 = get_clip_dist(feats[0], feats[2])
        c14 = get_clip_dist(feats[0], feats[3])
        c23 = get_clip_dist(feats[1], feats[2])
        c24 = get_clip_dist(feats[1], feats[3])
        c34 = get_clip_dist(feats[2], feats[3])

        clip_dist_arr = np.array([c12, c13, c14, c23, c24, c34])
        max_clip = np.max(clip_dist_arr)
        mean_clip = np.mean(clip_dist_arr)
        ratio_clip = max_clip / mean_clip if mean_clip > 0 else 1.0
        
        # MSE 계산
        def get_mse_dist(im1, im2):
            if im1 is not None and im2 is not None:
                return calculate_mse(im1, im2)
            return 0.0
            
        m12 = get_mse_dist(images[0], images[1])
        m13 = get_mse_dist(images[0], images[2])
        m14 = get_mse_dist(images[0], images[3])
        m23 = get_mse_dist(images[1], images[2])
        m24 = get_mse_dist(images[1], images[3])
        m34 = get_mse_dist(images[2], images[3])
        
        mse_dist_arr = np.array([m12, m13, m14, m23, m24, m34])
        max_mse = np.max(mse_dist_arr)
        mean_mse = np.mean(mse_dist_arr)
        ratio_mse = max_mse / mean_mse if mean_mse > 0 else 1.0
        
        # 장면 전환 판단 (CLIP 비비교쌍 0.20 기준)
        similar_clip_pairs = sum([1 for c in clip_dist_arr if c < CLIP_THRESHOLD])
        cuts = map_similar_pairs_to_cuts(similar_clip_pairs, max_clip)
        
        ids.append(row['Id'])
        predicted_scene_cuts.append(cuts)
        
        # Append CLIP
        clip_max_list.append(max_clip)
        clip_mean_list.append(mean_clip)
        clip_ratios.append(ratio_clip)
        c12_list.append(c12); c13_list.append(c13); c14_list.append(c14)
        c23_list.append(c23); c24_list.append(c24); c34_list.append(c34)
        
        # Append MSE
        mse_max_list.append(max_mse)
        mse_mean_list.append(mean_mse)
        mse_ratios.append(ratio_mse)
        m12_list.append(m12); m13_list.append(m13); m14_list.append(m14)
        m23_list.append(m23); m24_list.append(m24); m34_list.append(m34)
        
    res_df = pd.DataFrame({
        'Id': ids,
        'predicted_scene_cuts': predicted_scene_cuts,
        # CLIP Features
        'Max_clip': clip_max_list,
        'Mean_clip': clip_mean_list,
        'Ratio_clip': clip_ratios,
        'clip_12': c12_list, 'clip_13': c13_list, 'clip_14': c14_list,
        'clip_23': c23_list, 'clip_24': c24_list, 'clip_34': c34_list,
        # MSE Features
        'Max_mse': mse_max_list,
        'Mean_mse': mse_mean_list,
        'Ratio_mse': mse_ratios,
        'mse_12': m12_list, 'mse_13': m13_list, 'mse_14': m14_list,
        'mse_23': m23_list, 'mse_24': m24_list, 'mse_34': m34_list
    })
    
    # 5. 정규분포(Z-Score) 변환 적용
    print("\nScaling to normal distribution (Z-score) using QuantileTransformer...")
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
    
    # Fit & Transform for CLIP
    scaled_clip = qt.fit_transform(res_df[['Max_clip', 'Mean_clip']])
    res_df['Max_clip_scaled'] = scaled_clip[:, 0]
    res_df['Mean_clip_scaled'] = scaled_clip[:, 1]
    
    # Fit & Transform for MSE
    scaled_mse = qt.fit_transform(res_df[['Max_mse', 'Mean_mse']])
    res_df['Max_mse_scaled'] = scaled_mse[:, 0]
    res_df['Mean_mse_scaled'] = scaled_mse[:, 1]
    
    # 6. 전수 데이터 통계 출력
    print("\n" + "="*70)
    print("📊 [9,535개 비디오 전수조사] 안엄격 기준 장면 전환 횟수(0~3회) 분포")
    print("="*70)
    cut_counts = res_df['predicted_scene_cuts'].value_counts().sort_index()
    total_samples = len(res_df)
    for cuts, count in cut_counts.items():
        print(f"  🎬 장면 전환 {cuts}회 비디오 (안엄격): {count:5d}개 ({(count/total_samples*100):.2f}%)")
    print("-" * 70)
    print(f"👉 총 비디오 세트 수        : {total_samples:5d}개 (100.00%)")
    
    # 6.5. 통계 요약표 출력
    print("\n" + "="*75)
    print("📊 [원본 CLIP & MSE 피처] 통계 요약")
    print("="*75)
    print(res_df[['Max_clip', 'Mean_clip', 'Ratio_clip', 'Max_mse', 'Mean_mse', 'Ratio_mse']].describe().to_string())

    print("\n" + "="*75)
    print("📊 [정규화 완료된 Z-score 피처] 통계 요약")
    print("="*75)
    print(res_df[['Max_clip_scaled', 'Mean_clip_scaled', 'Max_mse_scaled', 'Mean_mse_scaled']].describe().to_string())

    # 7. CSV 저장
    OUTPUT_FILE = "snu_clip_features.csv"
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 전체 데이터 피처 CSV 저장 성공: '{OUTPUT_FILE}'로 저장 완료!")
    
if __name__ == "__main__":
    main()
