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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    # 1. Load holdout samples
    df = pd.read_csv(HOLDOUT_CSV)
    
    # Target words to filter samples containing subjects (expanded to get more data)
    target_words = ["person", "man", "woman", "child", "player", "skier", "girl", "boy", "athlete", "character", "he", "she", "people", "guy"]
    filtered_df = df[df['Sentence'].str.lower().str.contains('|'.join(target_words))]
    
    print(f"Filtered {len(filtered_df)} candidate samples containing subjects out of 300 holdout entries.")
    
    # 2. Load OWL-ViT and CLIP
    print("Loading OWL-ViT model (google/owlvit-base-patch32)...")
    owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    owl_model.eval()
    
    print("Loading CLIP model (openai/clip-vit-base-patch32)...")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    
    # 3. Extract crops
    crops_by_sample = {}
    processed_count = 0
    
    for idx, row in filtered_df.iterrows():
        sid = row['Id']
        sentence = row['Sentence'].lower()
        
        # Determine query query_text
        query_text = "person"
        for word in target_words:
            if word in sentence:
                query_text = word
                break
                
        # Collect image paths
        img_names = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        img_paths = [os.path.join(TRAIN_IMAGE_DIR, sid, name) for name in img_names]
        
        sample_crops = []
        
        for p in img_paths:
            if not os.path.exists(p):
                continue
            img = Image.open(p).convert("RGB")
            w, h = img.size
            
            # Run OWL-ViT
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
                
            # Select box with maximum area to represent the main subject
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
                sample_crops.append(crop_img)
                
        if len(sample_crops) >= 2:
            crops_by_sample[sid] = sample_crops
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} samples successfully...")
                
    print(f"\nExtracted crops successfully for {len(crops_by_sample)} samples.")
    
    # 4. Compute CLIP Embeddings
    print("\nComputing CLIP features for all crops...")
    embeddings_by_sample = {}
    for sid, crop_list in crops_by_sample.items():
        embeds = []
        for crop in crop_list:
            inputs = clip_processor(images=crop, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = clip_model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            embeds.append(feat.cpu().numpy()[0])
        embeddings_by_sample[sid] = embeds
        
    # 5. Build Same-Subject pairs vs Different-Subject pairs
    same_subject_dists = []
    diff_subject_dists = []
    
    # Same subject (pairwise within same video sample)
    for sid, embeds in embeddings_by_sample.items():
        n = len(embeds)
        for i in range(n):
            for j in range(i + 1, n):
                dist = 1.0 - np.dot(embeds[i], embeds[j])
                same_subject_dists.append(dist)
                
    # Different subjects (pairwise across different video samples)
    sids = list(embeddings_by_sample.keys())
    # Limit combinations slightly if sids is huge to avoid memory/time blowup, but keep it > 2000
    for idx_a in range(len(sids)):
        # Compare with up to 15 other random samples to get balanced sample size
        compare_indices = np.random.choice(range(len(sids)), size=min(15, len(sids)), replace=False)
        for idx_b in compare_indices:
            if idx_a == idx_b:
                continue
            sid_a = sids[idx_a]
            sid_b = sids[idx_b]
            for emb_a in embeddings_by_sample[sid_a]:
                for emb_b in embeddings_by_sample[sid_b]:
                    dist = 1.0 - np.dot(emb_a, emb_b)
                    diff_subject_dists.append(dist)
                    
    same_subject_dists = np.array(same_subject_dists)
    diff_subject_dists = np.array(diff_subject_dists)
    
    # 6. Analyze statistics
    print("\n" + "="*50)
    print("=== REAL CLIP CROP DISTANCE STATISTICAL ANALYSIS (A) ===")
    print("="*50)
    
    print(f"\n[Real Same Subject Pairs] (N = {len(same_subject_dists)})")
    print(f"Mean distance:   {np.mean(same_subject_dists):.4f}")
    print(f"Median distance: {np.median(same_subject_dists):.4f}")
    print(f"Min distance:    {np.min(same_subject_dists):.4f}")
    print(f"Max distance:    {np.max(same_subject_dists):.4f}")
    print(f"Std dev:         {np.std(same_subject_dists):.4f}")
    
    print(f"\n[Real Different Subject/Object Pairs] (N = {len(diff_subject_dists)})")
    print(f"Mean distance:   {np.mean(diff_subject_dists):.4f}")
    print(f"Median distance: {np.median(diff_subject_dists):.4f}")
    print(f"Min distance:    {np.min(diff_subject_dists):.4f}")
    print(f"Max distance:    {np.max(diff_subject_dists):.4f}")
    print(f"Std dev:         {np.std(diff_subject_dists):.4f}")
    
    # 7. Evaluate threshold splits (F1, Balanced Acc, False Alarm, Miss Rate)
    print("\n" + "-"*50)
    print("=== Real Threshold Sweep Calibration ===")
    print("-"*50)
    
    best_t_f1 = 0
    best_f1 = 0
    best_t_acc = 0
    best_acc = 0
    
    thresholds = np.arange(0.10, 0.50, 0.01)
    print(f"{'Threshold':<10} | {'False Alarm':<12} | {'Miss Rate':<10} | {'F1-Score':<10} | {'Balanced Acc':<12}")
    print("-" * 65)
    
    for t in thresholds:
        tp = np.sum(diff_dists > t) if 'diff_dists' in locals() else np.sum(diff_subject_dists > t)
        fp = np.sum(same_dists > t) if 'same_dists' in locals() else np.sum(same_subject_dists > t)
        tn = np.sum(same_dists <= t) if 'same_dists' in locals() else np.sum(same_subject_dists <= t)
        fn = np.sum(diff_dists <= t) if 'diff_dists' in locals() else np.sum(diff_subject_dists <= t)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        balanced_acc = (sensitivity + specificity) / 2
        
        if round(t * 100) % 2 == 0:  # print every 0.02 step robustly
            fp_rate = fp / len(same_subject_dists) * 100
            fn_rate = fn / len(diff_subject_dists) * 100
            print(f"{t:9.2f} | {fp_rate:10.1f}% | {fn_rate:8.1f}% | {f1:9.4f} | {balanced_acc*100:10.2f}%")
            
        if f1 > best_f1:
            best_f1 = f1
            best_t_f1 = t
            
        if balanced_acc > best_acc:
            best_acc = balanced_acc
            best_t_acc = t
            
    print("-" * 65)
    print(f"[*] Best Threshold by F1-Score:      {best_t_f1:.2f} (F1: {best_f1:.4f})")
    print(f"[*] Best Threshold by Balanced Acc:   {best_t_acc:.2f} (Acc: {best_acc*100:.2f}%)")

if __name__ == "__main__":
    main()
