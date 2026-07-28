# ================================================================================
# SNU AI Challenge — CLIP-based Feature Extraction and Scene Boundary Labeling
# ================================================================================

import os
import ssl
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from sklearn.preprocessing import QuantileTransformer

# ---- Configuration -----------------------------------------------------------
DATA_DIR = "/kaggle/input/datasets/leebyeongcheol/snu-ai-challenge-data/snuaichallenge_data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train")
CLIP_THRESHOLD = 0.20
OUTPUT_FILE = "snu_clip_features.csv"

def map_similar_pairs_to_cuts(similar_clip_pairs, max_clip):
    """
    Loose Cut scene partition mapping logic.
    Identifies 0 to 3 transitions based on pairwise CLIP distances.
    """
    if similar_clip_pairs >= 5:
        return 0
    if 2 <= similar_clip_pairs <= 4:
        return 1
    elif similar_clip_pairs == 1:
        return 2
    else:
        return 3

def calculate_mse(img1, img2):
    """
    Resizes images to 320x180 grayscale equivalent to compute physical mean-squared error.
    """
    img1_res = img1.resize((320, 180), Image.Resampling.LANCZOS)
    img2_res = img2.resize((320, 180), Image.Resampling.LANCZOS)
    a1 = np.array(img1_res, dtype=np.float32)
    a2 = np.array(img2_res, dtype=np.float32)
    return float(np.mean((a1 - a2) ** 2))

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: dataset CSV not found at {TRAIN_CSV}")
        return
        
    df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(df)} samples.")

    # 1. Load CLIP Model
    print("Loading CLIP ViT-B/32 via Torch Hub...")
    ssl._create_default_https_context = ssl._create_unverified_context
    clip_model, clip_preprocess = torch.hub.load("openai/CLIP", "ViT_B_32", trust_repo=True)
    clip_model = clip_model.to(device).eval()

    # 2. Collect unique image paths to avoid redundant GPU forwards
    image_paths = []
    for _, row in df.iterrows():
        sample_id = str(row['Id'])
        for f in [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]:
            image_paths.append(os.path.join(IMAGE_DIR, sample_id, f))
    unique_paths = list(set(image_paths))
    print(f"Unique images to extract: {len(unique_paths)}")

    # 3. GPU Batch Extraction
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
                except Exception:
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

    # 4. Pairwise Distance Computations
    ids = []
    clip_max_list, clip_mean_list = [], []
    c12, c13, c14, c23, c24, c34 = [], [], [], [], [], []
    mse_max_list, mse_mean_list = [], []
    m12, m13, m14, m23, m24, m34 = [], [], [], [], [], []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Metrics"):
        sample_id = str(row['Id'])
        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in shuffled_files]
        
        images = []
        for p in img_paths:
            if os.path.exists(p):
                try:
                    images.append(Image.open(p).convert("RGB"))
                except Exception:
                    images.append(None)
            else:
                images.append(None)
                
        feats = [embedding_cache.get(p) for p in img_paths]
        
        def get_clip_dist(f1, f2):
            if f1 is not None and f2 is not None:
                return float(1.0 - np.dot(f1, f2))
            return 0.0
            
        c_vals = [
            get_clip_dist(feats[0], feats[1]),
            get_clip_dist(feats[0], feats[2]),
            get_clip_dist(feats[0], feats[3]),
            get_clip_dist(feats[1], feats[2]),
            get_clip_dist(feats[1], feats[3]),
            get_clip_dist(feats[2], feats[3]),
        ]
        clip_max_list.append(np.max(c_vals))
        clip_mean_list.append(np.mean(c_vals))
        
        c12.append(c_vals[0]); c13.append(c_vals[1]); c14.append(c_vals[2])
        c23.append(c_vals[3]); c24.append(c_vals[4]); c34.append(c_vals[5])
        
        def get_mse_dist(im1, im2):
            if im1 is not None and im2 is not None:
                return calculate_mse(im1, im2)
            return 0.0
            
        m_vals = [
            get_mse_dist(images[0], images[1]),
            get_mse_dist(images[0], images[2]),
            get_mse_dist(images[0], images[3]),
            get_mse_dist(images[1], images[2]),
            get_mse_dist(images[1], images[3]),
            get_mse_dist(images[2], images[3]),
        ]
        mse_max_list.append(np.max(m_vals))
        mse_mean_list.append(np.mean(m_vals))
        
        m12.append(m_vals[0]); m13.append(m_vals[1]); m14.append(m_vals[2])
        m23.append(m_vals[3]); m24.append(m_vals[4]); m34.append(m_vals[5])
        ids.append(row['Id'])

    res_df = pd.DataFrame({
        'Id': ids,
        'Max_clip': clip_max_list,
        'Mean_clip': clip_mean_list,
        'dist_12': c12, 'dist_13': c13, 'dist_14': c14,
        'dist_23': c23, 'dist_24': c24, 'dist_34': c34,
        'Max_mse': mse_max_list,
        'Mean_mse': mse_mean_list,
        'mse_12': m12, 'mse_13': m13, 'mse_14': m14,
        'mse_23': m23, 'mse_24': m24, 'mse_34': m34
    })
    
    # 5. Apply Z-Score Normalization
    print("Normalizing distributions via QuantileTransformer...")
    qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
    
    scaled_clip = qt.fit_transform(res_df[['Max_clip', 'Mean_clip']])
    res_df['Max_clip_scaled'] = scaled_clip[:, 0]
    res_df['Mean_clip_scaled'] = scaled_clip[:, 1]
    
    scaled_mse = qt.fit_transform(res_df[['Max_mse', 'Mean_mse']])
    res_df['Max_mse_scaled'] = scaled_mse[:, 0]
    res_df['Mean_mse_scaled'] = scaled_mse[:, 1]
    
    # 6. Save Features
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Features saved successfully to: '{OUTPUT_FILE}'")
    
if __name__ == "__main__":
    main()
