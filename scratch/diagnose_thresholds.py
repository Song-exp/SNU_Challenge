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
    
    levels = []
    anchors = []
    means = []
    mins = []
    maxs = []
    cuts = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
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
            
        # a = mean sim of original to augmented
        sim_anchors = [float(np.dot(orig_cpu[i], aug_cpu[i])) for i in range(4)]
        a = float(np.mean(sim_anchors))
        
        # 6 pairwise similarities s_ij
        s = []
        s.append(float(np.dot(orig_cpu[0], orig_cpu[1])))
        s.append(float(np.dot(orig_cpu[0], orig_cpu[2])))
        s.append(float(np.dot(orig_cpu[0], orig_cpu[3])))
        s.append(float(np.dot(orig_cpu[1], orig_cpu[2])))
        s.append(float(np.dot(orig_cpu[1], orig_cpu[3])))
        s.append(float(np.dot(orig_cpu[2], orig_cpu[3])))
        
        mean_s = np.mean(s)
        level = mean_s / a
        
        levels.append(level)
        anchors.append(a)
        means.append(mean_s)
        mins.append(np.min(s))
        maxs.append(np.max(s))
        cuts.append(row['predicted_scene_cuts'])
        
    diag_df = pd.DataFrame({
        'cuts': cuts,
        'anchor': anchors,
        'mean_s': means,
        'min_s': mins,
        'max_s': maxs,
        'level': levels
    })
    
    print("\n=== Level Statistics by Scene Cuts ===")
    for c in sorted(diag_df['cuts'].unique()):
        sub = diag_df[diag_df['cuts'] == c]
        print(f"Cuts: {c} (Count: {len(sub)})")
        print(sub['level'].describe()[['mean', 'std', 'min', '50%', 'max']].to_string())
        print(f"Min similarity: min={sub['min_s'].min():.3f}, mean_min={sub['min_s'].mean():.3f}")
        print(f"Max similarity: max={sub['max_s'].max():.3f}, mean_max={sub['max_s'].mean():.3f}")
        print("-" * 40)

if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
