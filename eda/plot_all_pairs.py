import os
import sys
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
OUTPUT_PLOT = "C:/Users/user/Desktop/서울대/eda/pairwise_mse_distribution.png"
ARTIFACT_PLOT = "C:/Users/user/.gemini/antigravity-cli/brain/8c0c8c15-ad37-4207-b8c5-0210c0ab1b36/pairwise_mse_distribution.png"

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
        if val is not None:
            mses.append(val)
    return mses

def main():
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: train.csv not found at {TRAIN_CSV}")
        return
        
    df = pd.read_csv(TRAIN_CSV)
    
    # 2,000개 무작위 표본(12,000개 쌍)으로 매끄럽고 정확한 분포 시각화 (빠른 랜더링을 위함)
    sample_df = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)
    print(f"Sampling {len(sample_df)} videos for plotting (12,000 pairs)...")
    
    all_mses = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_sample, row) for _, row in sample_df.iterrows()]
        for f in futures:
            res = f.result()
            if res:
                all_mses.extend(res)
                
    all_mses = np.array(all_mses)
    print(f"Successfully calculated {len(all_mses)} pairwise MSE values.")
    
    # 그래프 스타일 설정
    plt.figure(figsize=(10, 6))
    
    # 히스토그램 그리기 (오차 범위 가시성을 위해 X축 15,000으로 제한)
    bins = np.linspace(0, 15000, 150)
    plt.hist(all_mses, bins=bins, color='#34495E', edgecolor='none', alpha=0.9, label='Frame Pairs (4C2)')
    
    # 영역 표시 (Shading)
    plt.axvspan(0, 1000, color='#2ECC71', alpha=0.2, label='Same Scene (MSE < 1,000)')
    plt.axvspan(1000, 2000, color='#F1C40F', alpha=0.2, label='Transitional Area (1,000-2,000)')
    plt.axvspan(2000, 15000, color='#E74C3C', alpha=0.15, label='Scene Cut (MSE > 2,000)')
    
    # 경계 세로선
    plt.axvline(1000, color='#27AE60', linestyle='--', linewidth=1.5)
    plt.axvline(2000, color='#D35400', linestyle='--', linewidth=1.5)
    
    # 라벨 및 타이틀
    plt.title('Pairwise Frame MSE Distribution (4C2 Combinations)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Mean Squared Error (MSE)', fontsize=12)
    plt.ylabel('Frequency (Count)', fontsize=12)
    
    plt.xlim(0, 15000)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.legend(loc='upper right', fontsize=10)
    
    # 텍스트 오버레이로 임계값 가이드라인 표시
    plt.text(350, plt.gca().get_ylim()[1]*0.8, 'Same Scene\n(MSE < 1,000)', color='#1E8449', fontsize=10, fontweight='bold', ha='center')
    plt.text(1500, plt.gca().get_ylim()[1]*0.8, 'Transition\n(1K-2K)', color='#B7950B', fontsize=9, fontweight='bold', ha='center')
    plt.text(5000, plt.gca().get_ylim()[1]*0.8, 'Scene Cut (Different Background)\n(MSE > 2,000)', color='#922B21', fontsize=11, fontweight='bold', ha='left')
    
    plt.tight_layout()
    
    # 저장
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300)
    
    os.makedirs(os.path.dirname(ARTIFACT_PLOT), exist_ok=True)
    plt.savefig(ARTIFACT_PLOT, dpi=300)
    
    print(f"Plot saved to:\n- {OUTPUT_PLOT}\n- {ARTIFACT_PLOT}")

if __name__ == "__main__":
    main()
