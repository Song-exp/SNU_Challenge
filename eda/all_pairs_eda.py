import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Force stdout to UTF-8 to prevent cp949 encode errors in Windows
sys.stdout.reconfigure(encoding='utf-8')

# =========================================================================
# [경로 설정]
# =========================================================================
DATA_DIR = "C:/Users/user/Desktop/서울대/snuaichallenge_data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train")

def compute_image_mse(img_path1, img_path2):
    try:
        with Image.open(img_path1).convert('L') as img1, Image.open(img_path2).convert('L') as img2:
            img1_r = img1.resize((64, 64), Image.Resampling.BILINEAR)
            img2_r = img2.resize((64, 64), Image.Resampling.BILINEAR)
            arr1 = np.array(img1_r, dtype=np.float32)
            arr2 = np.array(img2_r, dtype=np.float32)
            return float(np.mean((arr1 - arr2) ** 2))
    except Exception as e:
        return None

def process_sample(row):
    sample_id = str(row['Id'])
    img_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
    img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in img_files]
    
    # 4C2 = 6가지 가능한 쌍의 조합 계산
    pairs = [
        (img_paths[0], img_paths[1]),
        (img_paths[0], img_paths[2]),
        (img_paths[0], img_paths[3]),
        (img_paths[1], img_paths[2]),
        (img_paths[1], img_paths[3]),
        (img_paths[2], img_paths[3])
    ]
    
    mses = []
    for p1, p2 in pairs:
        val = compute_image_mse(p1, p2)
        if val is None:
            return None
        mses.append(val)
        
    return sample_id, mses

def main():
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: train.csv not found at {TRAIN_CSV}")
        return
        
    print("Loading train.csv...")
    df = pd.read_csv(TRAIN_CSV)
    
    print(f"Analyzing all {len(df)} samples using 4C2 combinations (6 pairs per sample)...")
    
    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_sample, row) for _, row in df.iterrows()]
        for f in futures:
            res = f.result()
            if res is not None:
                results.append(res)
                
    print(f"Successfully processed {len(results)} samples.")
    
    # 모든 6쌍의 MSE 수집 (총 9,535 * 6 = 57,210개)
    all_mses = []
    for s_id, mses in results:
        all_mses.extend(mses)
        
    all_mses = np.array(all_mses)
    
    print("\n=======================================================")
    print("1. 4C2 조합 픽셀 오차값(MSE) 전체 통계 분포 (57,210개 쌍)")
    print("=======================================================")
    print(f"최소값 (Min): {all_mses.min():.2f}")
    print(f"25% 백분위수: {np.percentile(all_mses, 25):.2f}")
    print(f"50% 백분위수 (중앙값): {np.percentile(all_mses, 50):.2f}")
    print(f"75% 백분위수: {np.percentile(all_mses, 75):.2f}")
    print(f"90% 백분위수: {np.percentile(all_mses, 90):.2f}")
    print(f"95% 백분위수: {np.percentile(all_mses, 95):.2f}")
    print(f"최대값 (Max): {all_mses.max():.2f}")
    
    buckets = [0, 500, 1000, 1500, 2000, 3000, 5000, 10000, 30000]
    print("\n=======================================================")
    print("2. 구간별 픽셀 오차 빈도 (동일장면 유사성 판정용)")
    print("=======================================================")
    for i in range(len(buckets) - 1):
        low, high = buckets[i], buckets[i+1]
        cnt = np.sum((all_mses >= low) & (all_mses < high))
        pct = cnt / len(all_mses) * 100
        print(f"MSE {low:5d} ~ {high:5d} : {cnt:4d}개 ({pct:5.2f}%)")
        
    # 동일 장면으로 간주하는 임계값을 1200으로 설정
    # 각 샘플별로 6쌍 중 '유사한 쌍(MSE < 1200)'의 개수를 카운트
    threshold = 1200
    print(f"\n=======================================================")
    print(f"3. 샘플당 유사한 프레임 쌍(MSE < {threshold})의 개수 분포")
    print("=======================================================")
    
    similar_pairs_counts = []
    for s_id, mses in results:
        similar_count = sum(1 for m in mses if m < threshold)
        similar_pairs_counts.append(similar_count)
        
    similar_pairs_counts = np.array(similar_pairs_counts)
    
    # 6개 유사쌍 -> 0 cuts (Type 0)
    # 3개 유사쌍 -> 1 cut (Type 1)
    # 1개 유사쌍 -> 2 cuts (Type 2)
    # 0개 유사쌍 -> 3 cuts (Type 3)
    # 그 외 노이즈(2, 4, 5개)도 통계에 그대로 출력
    for i in range(7):
        cnt = np.sum(similar_pairs_counts == i)
        pct = cnt / len(similar_pairs_counts) * 100
        desc = ""
        if i == 6: desc = " -> 장면 전환 0회 (전체 미세 행동)"
        elif i == 3: desc = " -> 장면 전환 1회 (2개 씬 분할)"
        elif i == 1: desc = " -> 장면 전환 2회 (3개 씬 분할)"
        elif i == 0: desc = " -> 장면 전환 3회 (매 프레임 장면 전환)"
        else: desc = " -> 전이 노이즈 및 과도기 프레임"
        
        print(f"유사한 쌍 {i}개 존재 : {cnt:4d}개 ({pct:5.2f}%){desc}")

if __name__ == "__main__":
    main()
