import os
import glob
import ast
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def compute_avg_mse(sample_id, train_dir):
    """
    Loads 4 images of a sample, resizes them to 32x32 for speed,
    and returns the average adjacent frame MSE.
    """
    try:
        sample_path = os.path.join(train_dir, sample_id)
        img_paths = sorted(glob.glob(os.path.join(sample_path, "*.jpg")))
        if len(img_paths) != 4:
            return sample_id, None
            
        imgs = []
        for p in img_paths:
            with Image.open(p) as img:
                # Resize to 32x32 for ultra-fast calculation
                img_small = img.convert('L').resize((32, 32))
                imgs.append(np.array(img_small, dtype=np.float32))
                
        mse1 = np.mean((imgs[0] - imgs[1]) ** 2)
        mse2 = np.mean((imgs[1] - imgs[2]) ** 2)
        mse3 = np.mean((imgs[2] - imgs[3]) ** 2)
        return sample_id, np.mean([mse1, mse2, mse3])
    except Exception as e:
        return sample_id, None

def main():
    data_dir = './snuaichallenge_data/'
    train_dir = os.path.join(data_dir, 'train')
    train_csv_path = os.path.join(data_dir, 'train.csv')
    
    print("Loading train.csv...")
    train_df = pd.read_csv(train_csv_path)
    
    # 1. Multi-threaded MSE scanning
    print("Scanning entire dataset for frame MSE (this will take about 30 seconds)...")
    sample_ids = train_df['Id'].tolist()
    
    mse_results = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(compute_avg_mse, s_id, train_dir) for s_id in sample_ids]
        for f in futures:
            s_id, val = f.result()
            if val is not None:
                mse_results[s_id] = val
                
    # Map MSE back to DataFrame
    train_df['Avg_MSE'] = train_df['Id'].map(mse_results)
    
    # Drop rows without valid MSE values
    train_df = train_df.dropna(subset=['Avg_MSE']).copy()
    
    # Categorize
    train_df['Type'] = train_df['Avg_MSE'].apply(lambda x: 'Fine-grained' if x < 800 else 'Scene Cut')
    
    # Save cache
    cache_path = './eda/dataset_mse_cache.csv'
    train_df[['Id', 'Sentence', 'Answer', 'Avg_MSE', 'Type']].to_csv(cache_path, index=False, encoding='utf-8-sig')
    print(f"Saved full dataset MSE cache to {cache_path}")
    
    # Print distribution
    dist = train_df['Type'].value_counts()
    print("\nFull Dataset Type Distribution:")
    print(dist)
    
    # 2. Stratified/Oversampled Sampling
    print("\nPerforming Stratified Sampling...")
    fine_grained_pool = train_df[train_df['Type'] == 'Fine-grained']
    scene_cut_pool = train_df[train_df['Type'] == 'Scene Cut']
    
    # Target validation size: 200
    # Include 40 Fine-grained samples (approx 20%) and 160 Scene Cut samples (approx 80%)
    num_fine = min(40, len(fine_grained_pool))
    num_scene = 200 - num_fine
    
    # Sample randomly using a fixed seed for reproducibility
    seed = 42
    fine_sampled = fine_grained_pool.sample(n=num_fine, random_state=seed)
    scene_sampled = scene_cut_pool.sample(n=num_scene, random_state=seed)
    
    # Combine
    stratified_valid = pd.concat([fine_sampled, scene_sampled]).sample(frac=1.0, random_state=seed) # Shuffle
    
    valid_save_path = './eda/stratified_valid.csv'
    stratified_valid[['Id', 'Sentence', 'Answer', 'Avg_MSE', 'Type']].to_csv(valid_save_path, index=False, encoding='utf-8-sig')
    
    print(f"Generated stratified validation set containing:")
    print(f" - Fine-grained samples: {num_fine} (Avg MSE: {fine_sampled['Avg_MSE'].mean():.2f})")
    print(f" - Scene Cut samples: {num_scene} (Avg MSE: {scene_sampled['Avg_MSE'].mean():.2f})")
    print(f"Saved stratified validation set to {valid_save_path}")

if __name__ == "__main__":
    main()
