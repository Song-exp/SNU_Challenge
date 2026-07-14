import os
import ast
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import time

# 경로 설정
BASE_DIR = r"C:\Users\bella\Desktop\대학\공모전\트리플에이치\snu_ai_공모전"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "data_train")
CACHE_CSV = os.path.join(BASE_DIR, "eda", "pairwise_mse_cache.csv")

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
    
    # 4C2 pairwise 6쌍 조합
    pairs = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3), (2, 3)
    ]
    
    mses = []
    for idx1, idx2 in pairs:
        val = compute_image_mse(img_paths[idx1], img_paths[idx2])
        if val is None:
            return sample_id, None, None
        mses.append(val)
        
    mses = np.array(mses)
    sim_pairs_count = int(np.sum(mses < 1200))
    median_mse = float(np.median(mses))
    
    return sample_id, sim_pairs_count, median_mse

def main():
    if not os.path.exists(TRAIN_CSV):
        print(f"train.csv not found at {TRAIN_CSV}")
        return
        
    df = pd.read_csv(TRAIN_CSV)
    print(f"Total samples to process: {len(df)}")
    
    # 캐시 파일이 이미 존재하면 스킵할 수 있도록 구현 가능하지만,
    # 정확한 계산을 위해 새로 수행합니다. (스레드 16개 구동으로 초고속 수행)
    start_time = time.time()
    results = []
    
    print("Calculating 4C2 pairwise MSE for all samples in parallel...")
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_sample, row) for _, row in df.iterrows()]
        for f in futures:
            res = f.result()
            results.append(res)
            
    # 매핑 데이터프레임 생성
    res_df = pd.DataFrame(results, columns=['Id', 'sim_pairs', 'median_mse'])
    res_df = res_df.dropna()
    
    # 저장
    os.makedirs(os.path.dirname(CACHE_CSV), exist_ok=True)
    res_df.to_csv(CACHE_CSV, index=False)
    
    elapsed = time.time() - start_time
    print(f"Calculation complete. Saved cache to {CACHE_CSV}")
    print(f"Time taken: {elapsed:.2f} seconds. Processed samples: {len(res_df)}")

if __name__ == "__main__":
    main()
