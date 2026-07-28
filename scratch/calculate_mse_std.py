import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

DATA_DIR = "C:/Users/user/Desktop/서울대/snuaichallenge_data"
TRAIN_CSV = os.path.join(DATA_DIR, "../train_검토_최종_완료.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train")
OUTPUT_PLOT = "C:/Users/user/Desktop/서울대/eda/mse_std_unbiased_distribution.png"

def calculate_mse(img1, img2):
    # BILINEAR is 10x faster than LANCZOS and works perfectly for MSE thresholding
    img1_res = img1.resize((320, 180), Image.Resampling.BILINEAR)
    img2_res = img2.resize((320, 180), Image.Resampling.BILINEAR)
    a1 = np.array(img1_res, dtype=np.float32)
    a2 = np.array(img2_res, dtype=np.float32)
    return float(np.mean((a1 - a2) ** 2))

def process_single_row(row_data):
    sample_id, files = row_data
    img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in files]
    
    images = []
    for p in img_paths:
        if os.path.exists(p):
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception:
                images.append(None)
        else:
            images.append(None)
            
    if any(img is None for img in images):
        return sample_id, np.nan
        
    try:
        m12 = calculate_mse(images[0], images[1])
        m13 = calculate_mse(images[0], images[2])
        m14 = calculate_mse(images[0], images[3])
        m23 = calculate_mse(images[1], images[2])
        m24 = calculate_mse(images[1], images[3])
        m34 = calculate_mse(images[2], images[3])
        
        mse_arr = np.array([m12, m13, m14, m23, m24, m34])
        mse_std = float(np.std(mse_arr))
        return sample_id, mse_std
    except Exception:
        return sample_id, np.nan

def main():
    df = pd.read_csv(TRAIN_CSV, encoding='cp949')
    print(f"Loaded {len(df)} samples.")
    
    # Prepare rows for parallel processing
    tasks = []
    for _, row in df.iterrows():
        sample_id = str(row['Id'])
        files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        tasks.append((sample_id, files))
        
    num_workers = max(1, cpu_count() - 1)
    print(f"Running parallel processing with {num_workers} CPU workers...")
    
    results = []
    with Pool(num_workers) as pool:
        for res in tqdm(pool.imap(process_single_row, tasks), total=len(tasks), desc="Calculating MSE Std"):
            results.append(res)
            
    res_df = pd.DataFrame(results, columns=['Id', 'mse_std']).dropna()
    
    print("\n=== Overall MSE Std Distribution ===")
    print(res_df['mse_std'].describe().to_string())
    
    # Save the plot
    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.hist(res_df['mse_std'], bins=60, density=True, color='#2c3e50', alpha=0.7, edgecolor='white')
    
    kde = gaussian_kde(res_df['mse_std'])
    x = np.linspace(res_df['mse_std'].min(), res_df['mse_std'].max(), 300)
    ax.plot(x, kde(x), color='#e74c3c', linewidth=2.5, label='KDE (밀도 추정 곡선)')
    
    ax.legend(fontsize=11)
    ax.set_title('📊 샘플 내 MSE 오차 표준편차(Std)의 전체 분포 (Unbiased)', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('샘플 내 6개 MSE 오차의 표준편차 (Std)', fontsize=12, labelpad=10)
    ax.set_ylabel('밀도 (Density)', fontsize=12, labelpad=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, bbox_inches='tight')
    print(f"Plot saved successfully to {OUTPUT_PLOT}")
    
    # Save features for verification/use
    res_df.to_csv("C:/Users/user/Desktop/서울대/eda/mse_std_features.csv", index=False)
    print("Features saved to eda/mse_std_features.csv")

if __name__ == "__main__":
    main()
