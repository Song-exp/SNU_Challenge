import os
import re
import ast
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import CLIPProcessor, CLIPModel

# Set Windows duplicate DLL fix
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

WORKSPACE_DIR = "C:/Users/user/Desktop/서울대"
TRAIN_IMAGE_DIR = os.path.join(WORKSPACE_DIR, "snuaichallenge_data/train")
HOLDOUT_CSV = os.path.join(WORKSPACE_DIR, "splits/holdout_300.csv")
DEBUG_DIR = os.path.join(WORKSPACE_DIR, "scratch/high_dist_debug")
REPORT_PATH = os.path.join(WORKSPACE_DIR, "scratch/high_dist_report.md")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    # 1. Load holdout samples
    df = pd.read_csv(HOLDOUT_CSV)
    
    # Target words to filter samples containing subjects
    target_words = ["person", "man", "woman", "child", "player", "skier", "girl", "boy", "athlete", "character", "he", "she", "people", "guy"]
    filtered_df = df[df['Sentence'].str.lower().str.contains('|'.join(target_words))].head(80)
    
    print(f"Loaded {len(filtered_df)} candidate samples containing subjects.")
    
    # 2. Load OWL-ViT and CLIP
    print("Loading OWL-ViT...")
    owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    owl_model.eval()
    
    print("Loading CLIP...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    
    # 3. Extract crops
    crops_by_sample = {}
    
    for idx, row in filtered_df.iterrows():
        sid = row['Id']
        sentence = row['Sentence'].lower()
        
        query_text = "person"
        for word in target_words:
            if word in sentence:
                query_text = word
                break
                
        img_names = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        img_paths = [os.path.join(TRAIN_IMAGE_DIR, sid, name) for name in img_names]
        
        sample_crops = []
        
        for f_idx, p in enumerate(img_paths):
            if not os.path.exists(p):
                continue
            img = Image.open(p).convert("RGB")
            w, h = img.size
            
            inputs = owl_processor(text=[[query_text]], images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = owl_model(**inputs)
                
            target_sizes = torch.tensor([img.size[::-1]], dtype=torch.float32).to(device)
            results = owl_processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.08
            )[0]
            
            boxes = results["boxes"].cpu().numpy()
            
            if len(boxes) == 0:
                continue
                
            best_idx = 0
            max_area = 0
            for b_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                box_area = (x2 - x1) * (y2 - y1)
                if box_area > max_area:
                    max_area = box_area
                    best_idx = b_idx
                    
            x1, y1, x2, y2 = boxes[best_idx]
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
            
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                crop_img = img.crop((x1, y1, x2, y2))
                sample_crops.append((f_idx + 1, crop_img))
                
        if len(sample_crops) >= 2:
            crops_by_sample[sid] = sample_crops
            
    print(f"Extracted crops successfully for {len(crops_by_sample)} samples.")
    
    # 4. Compute CLIP Embeddings
    print("Computing CLIP embeddings...")
    embeddings_by_sample = {}
    for sid, crop_tuples in crops_by_sample.items():
        embeds = []
        for f_idx, crop in crop_tuples:
            inputs = clip_processor(images=crop, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = clip_model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            embeds.append((f_idx, crop, feat.cpu().numpy()[0]))
        embeddings_by_sample[sid] = embeds
        
    # 5. Find same-subject pairs with distance > 0.30
    high_dist_pairs = []
    
    for sid, embeds in embeddings_by_sample.items():
        n = len(embeds)
        sentence = filtered_df[filtered_df['Id'] == sid]['Sentence'].values[0]
        for i in range(n):
            for j in range(i + 1, n):
                f_a, crop_a, emb_a = embeds[i]
                f_b, crop_b, emb_b = embeds[j]
                
                dist = 1.0 - np.dot(emb_a, emb_b)
                
                if dist > 0.30:
                    high_dist_pairs.append({
                        'sid': sid,
                        'sentence': sentence,
                        'frame_a': f_a,
                        'frame_b': f_b,
                        'crop_a': crop_a,
                        'crop_b': crop_b,
                        'distance': dist
                    })
                    
    print(f"Found {len(high_dist_pairs)} same-video crop pairs with CLIP distance > 0.30.")
    
    # 6. Save visual comparisons and write markdown report
    markdown_lines = [
        "# 🔍 Same-Video Crop Pairs with CLIP Distance > 0.30",
        "",
        "이 리포트는 동일 비디오 내에서 추출되었으나 CLIP 거리가 0.30을 초과하는 크롭 이미지 쌍들을 모아둔 리포트입니다.",
        "목적: 진짜 동일 피사체의 포즈 급변인지, 아니면 다른 피사체/배경으로 바운딩 박스가 오탐지(Tracker Drift)된 경우인지 눈으로 직접 확인합니다.",
        "",
        "| ID | Distance | Frame Pair | Caption | Debug Image Link |",
        "| --- | --- | --- | --- | --- |"
    ]
    
    for idx, p in enumerate(high_dist_pairs):
        sid = p['sid']
        dist = p['distance']
        f_a, f_b = p['frame_a'], p['frame_b']
        crop_a, crop_b = p['crop_a'], p['crop_b']
        sentence = p['sentence']
        
        # Concatenate crops side-by-side for comparison
        # Resize to have same height
        h_target = 150
        w_a = int(crop_a.width * h_target / crop_a.height)
        w_b = int(crop_b.width * h_target / crop_b.height)
        
        r_a = crop_a.resize((w_a, h_target))
        r_b = crop_b.resize((w_b, h_target))
        
        combined_img = Image.new('RGB', (w_a + w_b + 10, h_target), color=(255, 0, 0)) # Red separator
        combined_img.paste(r_a, (0, 0))
        combined_img.paste(r_b, (w_a + 10, 0))
        
        filename = f"{sid}_f{f_a}_f{f_b}_dist_{dist:.3f}.jpg"
        save_path = os.path.join(DEBUG_DIR, filename)
        combined_img.save(save_path)
        
        # Clickable link format for Windows files
        win_link = f"file:///{save_path.replace(os.sep, '/')}"
        
        markdown_lines.append(
            f"| `{sid}` | **{dist:.4f}** | Image {f_a} - Image {f_b} | \"{sentence}\" | [View Crop Pair Comparison]({win_link}) |"
        )
        
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(markdown_lines))
        
    print(f"Visual debug report saved successfully at: {REPORT_PATH}")

if __name__ == "__main__":
    main()
