import os
import ast
import ssl
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.notebook import tqdm
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

# 임계 기준 설정 (검수기 최적 스펙 반영)
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

def map_similar_pairs_to_cuts(similar_clip_pairs):
    # 유사쌍 개수 -> 장면 전환 횟수 매핑
    if similar_clip_pairs >= 5:
        return 0  # 0회 전환 (동일 장면)
    elif 2 <= similar_clip_pairs <= 4:
        return 1  # 1회 전환 (2개 씬 분할)
    elif similar_clip_pairs == 1:
        return 2  # 2회 전환 (3개 씬 분할)
    else:
        return 3  # 3회 전환 (매 프레임 장면 전환)

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

    # 2. 이미지 고유 경로 수집
    print("Collecting image paths...")
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

    # 4. 하이브리드 장면 분석 루프 (전체 데이터 대상)
    print("\nExtracting Hybrid features (Scene Cuts and Composition changes)...")
    
    # 저장용 리스트 정의
    ids = []
    
    # 6개 개별 CLIP 거리 저장
    dist_12, dist_13, dist_14, dist_23, dist_24, dist_34 = [], [], [], [], [], []
    # 6개 개별 MSE 물리오차 저장
    mse_12, mse_13, mse_14, mse_23, mse_24, mse_34 = [], [], [], [], [], []
    
    # 요약 통계 지표들
    clip_max_list, clip_mean_list = [], []
    mse_max_list, mse_mean_list = [], []
    
    # 하이브리드 가공 라벨
    predicted_scene_cuts = [] # 진짜 장면 전환 횟수 (0~3)
    dynamic_action_pairs = [] # 동일 씬 내 큰 구도/동작 변화 횟수 (0~6)
    
    pairs = list(combinations(range(4), 2))
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
        sample_id = str(row['Id'])
        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in shuffled_files]
        
        # 6개 이미지 조합 분석
        feats = [embedding_cache.get(p) for p in img_paths]
        
        # 픽셀 MSE 계산
        m12 = compute_image_mse(img_paths[0], img_paths[1])
        m13 = compute_image_mse(img_paths[0], img_paths[2])
        m14 = compute_image_mse(img_paths[0], img_paths[3])
        m23 = compute_image_mse(img_paths[1], img_paths[2])
        m24 = compute_image_mse(img_paths[1], img_paths[3])
        m34 = compute_image_mse(img_paths[2], img_paths[3])
        
        # CLIP 거리 계산
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
        mse_arr = np.array([m12, m13, m14, m23, m24, m34])
        
        # [하이브리드 판정 알고리즘]
        # 1) 장면 전환 쌍 개수 카운트: CLIP 임계값 0.20 이상인 경우
        true_cuts_pairs = sum([1 for c in clip_dist_arr if c >= CLIP_THRESHOLD])
        # 6개 중 장면이 안 바뀐(CLIP < 0.20) 유사한 쌍의 개수
        similar_clip_pairs = 6 - true_cuts_pairs
        
        # 2) 동일 장면 내 구도 변화 쌍 개수 카운트: CLIP < 0.20 인데 MSE >= 1200 인 경우
        dynamic_actions = sum([1 for c, m in zip(clip_dist_arr, mse_arr) if (c < CLIP_THRESHOLD) and (m >= MSE_THRESHOLD)])
        
        # 유사쌍 개수를 바탕으로 영상 전체의 장면 전환 횟수(0~3) 매핑
        scene_cut_count = map_similar_pairs_to_cuts(similar_clip_pairs)
        
        # 리스트에 추가
        ids.append(row['Id'])
        
        dist_12.append(c12); dist_13.append(c13); dist_14.append(c14)
        dist_23.append(c23); dist_24.append(c24); dist_34.append(c34)
        
        mse_12.append(m12); mse_13.append(m13); mse_14.append(m14)
        mse_23.append(m23); mse_24.append(m24); mse_34.append(m34)
        
        clip_max_list.append(np.max(clip_dist_arr))
        clip_mean_list.append(np.mean(clip_dist_arr))
        mse_max_list.append(np.max(mse_arr))
        mse_mean_list.append(np.mean(mse_arr))
        
        predicted_scene_cuts.append(scene_cut_count)
        dynamic_action_pairs.append(dynamic_actions)
        
    # 결과 최종 테이블 병합
    res_df = pd.DataFrame({
        'Id': ids,
        'predicted_scene_cuts': predicted_scene_cuts,
        'dynamic_action_pairs': dynamic_action_pairs,
        'CLIP_Max': clip_max_list,
        'CLIP_Mean': clip_mean_list,
        'MSE_Max': mse_max_list,
        'MSE_Mean': mse_mean_list,
        'dist_12': dist_12, 'dist_13': dist_13, 'dist_14': dist_14,
        'dist_23': dist_23, 'dist_24': dist_24, 'dist_34': dist_34,
        'mse_12': mse_12, 'mse_13': mse_13, 'mse_14': mse_14,
        'mse_23': mse_23, 'mse_24': mse_24, 'mse_34': mse_34
    })
    
    # 5. 전수조사 요약 통계 출력
    print("\n" + "="*80)
    print("📊 [9,535개 비디오 전수조사] 하이브리드 장면 분석 통계 분포 결과")
    print("="*80)
    
    cut_counts = res_df['predicted_scene_cuts'].value_counts().sort_index()
    total_samples = len(res_df)
    print("\n🟢 [의미적] 장면 전환 횟수(0~3회) 분포:")
    for cuts, count in cut_counts.items():
        print(f"  🎬 장면 전환 {cuts}회 비디오: {count:5d}개 ({(count/total_samples*100):.2f}%)")
        
    action_counts = res_df['dynamic_action_pairs'].value_counts().sort_index()
    print("\n🟢 [물리적] 동일 씬 내 큰 구도/행동 변화 쌍 개수(0~6개) 분포:")
    for actions, count in action_counts.items():
        print(f"  🔄 구도 변화 {actions}개 쌍 발견 비디오: {count:5d}개 ({(count/total_samples*100):.2f}%)")
        
    # 6. 최종 CSV 저장
    OUTPUT_FILE = "snu_video_hybrid_features.csv"
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 전수 데이터 라벨링 및 피처 CSV 저장 성공: '{OUTPUT_FILE}'로 저장 완료!")
    print("👉 캐글 우측 패널의 'Output' 경로에서 다운로드해 주세요.")

if __name__ == "__main__":
    main()
