import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageEnhance
from tqdm import tqdm

WORKSPACE_DIR = "C:/Users/user/Desktop/서울대"
TRAIN_CSV = os.path.join(WORKSPACE_DIR, "train_검토_최종_완료.csv")
IMAGE_DIR = os.path.join(WORKSPACE_DIR, "snuaichallenge_data/train")
OUTPUT_CSV = os.path.join(WORKSPACE_DIR, "eda/unsupervised_gui_results.csv")

def augment_image(pil_img):
    w, h = pil_img.size
    # Crop 90% from center
    left = int(w * 0.05)
    top = int(h * 0.05)
    right = int(w * 0.95)
    bottom = int(h * 0.95)
    cropped = pil_img.crop((left, top, right, bottom))
    # Enhance contrast and brightness slightly
    enhancer_c = ImageEnhance.Contrast(cropped)
    cropped_c = enhancer_c.enhance(1.1)
    enhancer_b = ImageEnhance.Brightness(cropped_c)
    return enhancer_b.enhance(1.1)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP model
    print("Loading CLIP model...")
    clip_model, clip_preprocess = torch.hub.load("openai/CLIP", "ViT_B_32", trust_repo=True)
    clip_model = clip_model.to(device).eval()
    
    # Load dataset
    df = pd.read_csv(TRAIN_CSV, encoding='cp949')
    test_df = df.iloc[:100].copy()  # First 100 samples
    print(f"Processing {len(test_df)} samples...")
    
    ids = []
    cuts_list = []
    groups_list = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Unsupervised Grouping"):
        sample_id = str(row['Id'])
        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        ans = eval(row['Answer'])  # Chronological order mapping
        
        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in shuffled_files]
        
        # Load and preprocess original & augmented images
        images = []
        aug_images = []
        for p in img_paths:
            if os.path.exists(p):
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    aug_images.append(augment_image(img))
                except:
                    images.append(None)
                    aug_images.append(None)
            else:
                images.append(None)
                aug_images.append(None)
                
        if any(img is None for img in images):
            ids.append(sample_id)
            cuts_list.append(np.nan)
            groups_list.append("")
            continue
            
        # Get embeddings
        orig_tensors = torch.stack([clip_preprocess(img) for img in images]).to(device)
        aug_tensors = torch.stack([clip_preprocess(img) for img in aug_images]).to(device)
        
        with torch.no_grad():
            orig_feats = clip_model.encode_image(orig_tensors)
            orig_feats = orig_feats / orig_feats.norm(p=2, dim=-1, keepdim=True)
            orig_cpu = orig_feats.cpu().numpy()
            
            aug_feats = clip_model.encode_image(aug_tensors)
            aug_feats = aug_feats / aug_feats.norm(p=2, dim=-1, keepdim=True)
            aug_cpu = aug_feats.cpu().numpy()
            
        # Calculate anchor a = mean similarity of original to augmented
        sim_anchors = [float(np.dot(orig_cpu[i], aug_cpu[i])) for i in range(4)]
        a = float(np.mean(sim_anchors))
        
        # Calculate 6 pairwise similarities s_ij
        s = {}
        s[(0, 1)] = float(np.dot(orig_cpu[0], orig_cpu[1]))
        s[(0, 2)] = float(np.dot(orig_cpu[0], orig_cpu[2]))
        s[(0, 3)] = float(np.dot(orig_cpu[0], orig_cpu[3]))
        s[(1, 2)] = float(np.dot(orig_cpu[1], orig_cpu[2]))
        s[(1, 3)] = float(np.dot(orig_cpu[1], orig_cpu[3]))
        s[(2, 3)] = float(np.dot(orig_cpu[2], orig_cpu[3]))
        
        s_arr = np.array(list(s.values()))
        mean_s = np.mean(s_arr)
        
        # Step 1. Level Judgment
        level = mean_s / a
        
        if level >= 0.92:
            # 1 scene (0 cuts)
            groups = [[0, 1, 2, 3]]
            predicted_cuts = 0
        elif level < 0.60 and all(val < 0.65 * a for val in s.values()):
            # 4 scenes (3 cuts)
            groups = [[0], [1], [2], [3]]
            predicted_cuts = 3
        else:
            # Step 2. Structure Judgment (Gap split)
            sorted_pairs = sorted(s.items(), key=lambda x: x[1], reverse=True)
            
            max_gap = -1.0
            split_idx = -1
            for k in range(len(sorted_pairs) - 1):
                gap = sorted_pairs[k][1] - sorted_pairs[k+1][1]
                if gap > max_gap:
                    max_gap = gap
                    split_idx = k
                    
            same_scene_pairs = [pair for pair, sim_val in sorted_pairs[:split_idx + 1]]
            
            # Step 3. Union-Find
            parent = list(range(4))
            def find(x):
                while parent[x] != x:
                    x = parent[x]
                return x
            def union(x, y):
                root_x = find(x)
                root_y = find(y)
                if root_x != root_y:
                    parent[root_y] = root_x
                    
            for (i, j) in same_scene_pairs:
                union(i, j)
                
            groups_dict = {}
            for idx in range(4):
                root = find(idx)
                if root not in groups_dict:
                    groups_dict[root] = []
                groups_dict[root].append(idx)
                
            groups = list(groups_dict.values())
            predicted_cuts = len(groups) - 1
            
        # Map groups to chronological frame order (1 to 4)
        chrono_groups = []
        for g in groups:
            chrono_g = sorted([ans[idx] for idx in g])
            chrono_groups.append(chrono_g)
        chrono_groups.sort(key=lambda x: x[0])
        
        groups_str = " | ".join([f"{{{', '.join(map(str, cg))}}}" for cg in chrono_groups])
        
        ids.append(sample_id)
        cuts_list.append(predicted_cuts)
        groups_list.append(groups_str)
        
    out_df = pd.DataFrame({
        'Id': ids,
        'unsupervised_cuts': cuts_list,
        'unsupervised_groups': groups_list
    }).dropna()
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
