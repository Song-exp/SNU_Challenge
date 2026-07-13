import os
import ast
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
    ans = ast.literal_eval(row['Answer'])
    
    shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
    ordered_files = [None] * 4
    for idx, pos in enumerate(ans):
        ordered_files[pos - 1] = shuffled_files[idx]
        
    img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in ordered_files]
    
    mse1 = compute_image_mse(img_paths[0], img_paths[1])
    mse2 = compute_image_mse(img_paths[1], img_paths[2])
    mse3 = compute_image_mse(img_paths[2], img_paths[3])
    
    return sample_id, mse1, mse2, mse3

def main():
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: train.csv not found at {TRAIN_CSV}")
        return
        
    print("Loading train.csv...")
    df = pd.read_csv(TRAIN_CSV)
    
    sample_df = df.sample(n=min(1000, len(df)), random_state=42).reset_index(drop=True)
    print(f"Analyzing {len(sample_df)} representative samples...")
    
    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_sample, row) for _, row in sample_df.iterrows()]
        for f in futures:
            res = f.result()
            if None not in res[1:]:
                results.append(res)
                
    print(f"Successfully processed {len(results)} samples.")
    
    all_mses = []
    for s_id, m1, m2, m3 in results:
        all_mses.extend([m1, m2, m3])
        
    all_mses = np.array(all_mses)
    
    print("\n=======================================================")
    print("1. 픽셀 오차값(MSE) 전체 통계 분포 (3,000개 전환점)")
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
    print("2. 구간별 전이 오차 빈도 (동일장면 vs 장면전환 판단용)")
    print("=======================================================")
    for i in range(len(buckets) - 1):
        low, high = buckets[i], buckets[i+1]
        cnt = np.sum((all_mses >= low) & (all_mses < high))
        pct = cnt / len(all_mses) * 100
        print(f"MSE {low:5d} ~ {high:5d} : {cnt:4d}개 ({pct:5.2f}%)")
        
    threshold = 1200
    print(f"\n=======================================================")
    print(f"3. 장면 전환 횟수 분석 (임계값 {threshold} 기준)")
    print("=======================================================")
    
    cut_counts = []
    for s_id, m1, m2, m3 in results:
        cuts = 0
        if m1 >= threshold: cuts += 1
        if m2 >= threshold: cuts += 1
        if m3 >= threshold: cuts += 1
        cut_counts.append(cuts)
        
    cut_counts = np.array(cut_counts)
    for i in range(4):
        cnt = np.sum(cut_counts == i)
        pct = cnt / len(cut_counts) * 100
        print(f"유형 {i} (장면 전환 {i}회) : {cnt:4d}개 ({pct:5.2f}%)")

if __name__ == "__main__":
    main()
