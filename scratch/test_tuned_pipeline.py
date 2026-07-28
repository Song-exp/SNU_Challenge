import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageEnhance
from tqdm import tqdm

WORKSPACE_DIR = "C:/Users/user/Desktop/서울대"
TRAIN_CSV = os.path.join(WORKSPACE_DIR, "train_검토_최종_완료.csv")
FEATURES_CSV = os.path.join(WORKSPACE_DIR, "snu_clip_features.csv")
IMAGE_DIR = os.path.join(WORKSPACE_DIR, "snuaichallenge_data/train")
OUTPUT_CSV = os.path.join(WORKSPACE_DIR, "eda/unsupervised_gui_results.csv")

def augment_image_light(pil_img):
    # Safe semantic-preserving augmentation: minor contrast & brightness changes
    enhancer_c = ImageEnhance.Contrast(pil_img)
    img_c = enhancer_c.enhance(1.05)
    enhancer_b = ImageEnhance.Brightness(img_c)
    return enhancer_b.enhance(1.05)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_preprocess = torch.hub.load("openai/CLIP", "ViT_B_32", trust_repo=True)
    clip_model = clip_model.to(device).eval()
    
    df_train = pd.read_csv(TRAIN_CSV, encoding='cp949')[['Id', 'Answer', 'Input_1', 'Input_2', 'Input_3', 'Input_4']].iloc[:100]
    df_feat = pd.read_csv(FEATURES_CSV)
    df = pd.merge(df_train, df_feat, on='Id')
    
    ids = []
    predicted_cuts_list = []
    groups_list = []
    actual_cuts_list = []
    
    correct = 0
    total = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tuned Pipeline"):
        sample_id = str(row['Id'])
        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        ans = eval(row['Answer'])
        
        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in shuffled_files]
        images = [Image.open(p).convert("RGB") for p in img_paths if os.path.exists(p)]
        if len(images) < 4:
            continue
            
        aug_images = [augment_image_light(img) for img in images]
        
        orig_tensors = torch.stack([clip_preprocess(img) for img in images]).to(device)
        aug_tensors = torch.stack([clip_preprocess(img) for img in aug_images]).to(device)
        
        with torch.no_grad():
            orig_feats = clip_model.encode_image(orig_tensors)
            orig_feats = orig_feats / orig_feats.norm(p=2, dim=-1, keepdim=True)
            orig_cpu = orig_feats.cpu().numpy()
            
            aug_feats = clip_model.encode_image(aug_tensors)
            aug_feats = aug_feats / aug_feats.norm(p=2, dim=-1, keepdim=True)
            aug_cpu = aug_feats.cpu().numpy()
            
        sim_anchors = [float(np.dot(orig_cpu[i], aug_cpu[i])) for i in range(4)]
        a = float(np.mean(sim_anchors))
        
        s = {}
        s[(0, 1)] = float(np.dot(orig_cpu[0], orig_cpu[1]))
        s[(0, 2)] = float(np.dot(orig_cpu[0], orig_cpu[2]))
        s[(0, 3)] = float(np.dot(orig_cpu[0], orig_cpu[3]))
        s[(1, 2)] = float(np.dot(orig_cpu[1], orig_cpu[2]))
        s[(1, 3)] = float(np.dot(orig_cpu[1], orig_cpu[3]))
        s[(2, 3)] = float(np.dot(orig_cpu[2], orig_cpu[3]))
        
        s_arr = np.array(list(s.values()))
        mean_s = np.mean(s_arr)
        level = mean_s / a
        max_s = np.max(s_arr)
        
        # TUNED METHODOLOGY
        # 1. Level check for 0 cuts
        if level >= 0.86:
            groups = [[0, 1, 2, 3]]
            pred_cuts = 0
        # 2. Max similarity check for 3 cuts
        elif max_s < 0.73:
            groups = [[0], [1], [2], [3]]
            pred_cuts = 3
        # 3. Otherwise: relative Gap split + Union-Find
        else:
            sorted_pairs = sorted(s.items(), key=lambda x: x[1], reverse=True)
            
            max_gap = -1.0
            split_idx = -1
            for k in range(len(sorted_pairs) - 1):
                gap = sorted_pairs[k][1] - sorted_pairs[k+1][1]
                if gap > max_gap:
                    max_gap = gap
                    split_idx = k
                    
            same_scene_pairs = [pair for pair, sim_val in sorted_pairs[:split_idx + 1]]
            
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
            pred_cuts = len(groups) - 1
            
        # Chronological frame mapping
        chrono_groups = []
        for g in groups:
            chrono_g = sorted([ans[idx] for idx in g])
            chrono_groups.append(chrono_g)
        chrono_groups.sort(key=lambda x: x[0])
        
        groups_str = " | ".join([f"{{{', '.join(map(str, cg))}}}" for cg in chrono_groups])
        
        actual_cuts = int(row['predicted_scene_cuts'])
        if pred_cuts == actual_cuts:
            correct += 1
        total += 1
        
        ids.append(sample_id)
        predicted_cuts_list.append(pred_cuts)
        groups_list.append(groups_str)
        actual_cuts_list.append(actual_cuts)
        
    print(f"\nAccuracy on 100 samples compared to baseline: {correct}/{total} ({correct/total*100:.1f}%)")
    
    # Save the updated results
    out_df = pd.DataFrame({
        'Id': ids,
        'unsupervised_cuts': predicted_cuts_list,
        'unsupervised_groups': groups_list
    })
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Tuned results saved successfully to {OUTPUT_CSV}")

if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
