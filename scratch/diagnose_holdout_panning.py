import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import re
import ast
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

WORKSPACE_DIR = "C:/Users/user/Desktop/서울대"
HOLDOUT_CSV = os.path.join(WORKSPACE_DIR, "splits/holdout_300.csv")
IMAGE_DIR = os.path.join(WORKSPACE_DIR, "snuaichallenge_data/train")
PROGRESS_FILE = os.path.join(WORKSPACE_DIR, "scratch/diagnose_progress.txt")

def log_progress(message, append=True):
    mode = 'a' if append else 'w'
    with open(PROGRESS_FILE, mode, encoding='utf-8') as f:
        f.write(message + "\n")
    print(message)

def main():
    log_progress("Starting holdout diagnosis (Optimized Keyword-First version)...", append=False)
    
    if not os.path.exists(HOLDOUT_CSV):
        log_progress(f"Holdout CSV not found: {HOLDOUT_CSV}")
        return

    df = pd.read_csv(HOLDOUT_CSV)
    log_progress(f"Loaded {len(df)} samples from holdout_300.csv")

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_progress(f"Using device: {device}")

    # Load Faster R-CNN model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.3).to(device)
    model.eval()
    preprocess = weights.transforms()

    results = []

    for idx, row in df.iterrows():
        sample_id = str(row['Id'])
        sentence = row['Sentence'].lower()
        ans = ast.literal_eval(row['Answer'])
        
        # Categorize camera/object movement keywords first
        zoom_in_kw = ["zoom in", "zooms in", "zoomed in", "zooming in", "closer", "shifts close", "moves in"]
        zoom_out_kw = ["zoom out", "zooms out", "zoomed out", "zooming out", "further", "moves out", "receding", "moves away"]
        
        pan_left_kw = ["pan left", "pans left", "panned left", "panning left", "camera moves left", "camera shifts left"]
        pan_right_kw = ["pan right", "pans right", "panned right", "panning right", "camera moves right", "camera shifts right"]
        
        move_left_kw = ["moves left", "moving left", "slides left", "skis left", "runs left", "walks left"]
        move_right_kw = ["moves right", "moving right", "slides right", "skis right", "runs right", "walks right"]

        has_zoom_in = any(kw in sentence for kw in zoom_in_kw)
        has_zoom_out = any(kw in sentence for kw in zoom_out_kw)
        has_pan_left = any(kw in sentence for kw in pan_left_kw)
        has_pan_right = any(kw in sentence for kw in pan_right_kw)
        has_move_left = any(kw in sentence for kw in move_left_kw)
        has_move_right = any(kw in sentence for kw in move_right_kw)

        is_zoom_sample = has_zoom_in or has_zoom_out
        is_pan_sample = has_pan_left or has_pan_right or has_move_left or has_move_right

        if not is_zoom_sample and not is_pan_sample:
            # Skip Faster R-CNN entirely for speed
            results.append({
                'Id': sample_id,
                'Sentence': row['Sentence'],
                'DetectCount': 0,
                'Status': 'N/A',
                'Reason': 'No spatial keywords',
                'IsPan': False,
                'IsZoom': False
            })
            continue

        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        # Reorder images to chronological order
        ordered_files = [None] * 4
        for i, pos in enumerate(ans):
            ordered_files[pos - 1] = shuffled_files[i]

        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in ordered_files]
        if not all(os.path.exists(p) for p in img_paths):
            results.append({
                'Id': sample_id,
                'Sentence': row['Sentence'],
                'DetectCount': 0,
                'Status': 'N/A',
                'Reason': 'Files missing',
                'IsPan': is_pan_sample,
                'IsZoom': is_zoom_sample
            })
            continue

        # Detect the largest object in each frame
        centers_x = []
        areas = []
        detect_count = 0

        for img_path in img_paths:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            img_area = w * h

            input_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                predictions = model(input_tensor)[0]

            boxes = predictions['boxes'].cpu().numpy()

            max_box_area = 0.0
            best_x_center = 0.5

            for box in boxes:
                box_w = box[2] - box[0]
                box_h = box[3] - box[1]
                box_area = (box_w * box_h) / img_area

                if box_area > max_box_area:
                    max_box_area = box_area
                    best_x_center = ((box[0] + box[2]) / 2) / w

            if max_box_area > 0:
                detect_count += 1
            centers_x.append(best_x_center)
            areas.append(max_box_area)

        status = "N/A"
        reason = "N/A"

        if detect_count >= 2:
            if is_pan_sample:
                valid_idx = [i for i, a in enumerate(areas) if a > 0]
                first_valid = valid_idx[0]
                last_valid = valid_idx[-1]
                x_start = centers_x[first_valid]
                x_end = centers_x[last_valid]

                expected_right = False
                expected_left = False

                if has_pan_left or has_move_right:
                    expected_right = True
                if has_pan_right or has_move_left:
                    expected_left = True

                if expected_right and expected_left:
                    status = "N/A"
                    reason = "Mixed pan direction"
                elif expected_right:
                    if x_end > x_start:
                        status = "✅ Consistent"
                        reason = f"Object moved right (X: {x_start:.2f} -> {x_end:.2f})"
                    else:
                        status = "❌ Inconsistent"
                        reason = f"Object did not move right (X: {x_start:.2f} -> {x_end:.2f})"
                elif expected_left:
                    if x_end < x_start:
                        status = "✅ Consistent"
                        reason = f"Object moved left (X: {x_start:.2f} -> {x_end:.2f})"
                    else:
                        status = "❌ Inconsistent"
                        reason = f"Object did not move left (X: {x_start:.2f} -> {x_end:.2f})"
            
            elif is_zoom_sample:
                valid_idx = [i for i, a in enumerate(areas) if a > 0]
                first_valid = valid_idx[0]
                last_valid = valid_idx[-1]
                a_start = areas[first_valid]
                a_end = areas[last_valid]

                if has_zoom_in and has_zoom_out:
                    status = "N/A"
                    reason = "Mixed zoom direction"
                elif has_zoom_in:
                    if a_end > a_start:
                        status = "✅ Consistent"
                        reason = f"Zoom-in matched area increase ({a_start*100:.1f}% -> {a_end*100:.1f}%)"
                    else:
                        status = "❌ Inconsistent"
                        reason = f"Zoom-in failed (area decreased: {a_start*100:.1f}% -> {a_end*100:.1f}%)"
                elif has_zoom_out:
                    if a_end < a_start:
                        status = "✅ Consistent"
                        reason = f"Zoom-out matched area decrease ({a_start*100:.1f}% -> {a_end*100:.1f}%)"
                    else:
                        status = "❌ Inconsistent"
                        reason = f"Zoom-out failed (area increased: {a_start*100:.1f}% -> {a_end*100:.1f}%)"

        results.append({
            'Id': sample_id,
            'Sentence': row['Sentence'],
            'DetectCount': detect_count,
            'Status': status,
            'Reason': reason,
            'IsPan': is_pan_sample,
            'IsZoom': is_zoom_sample
        })

        if (idx + 1) % 10 == 0:
            log_progress(f"Processed {idx+1}/{len(df)} samples...")

    res_df = pd.DataFrame(results)
    
    # Generate statistics
    stats_str = []
    stats_str.append("\n" + "="*50)
    stats_str.append("FINAL STATISTICS")
    stats_str.append("="*50)
    pan_df = res_df[res_df['IsPan']]
    zoom_df = res_df[res_df['IsZoom'] & ~res_df['IsPan']]
    
    pan_valid = pan_df[pan_df['Status'] != "N/A"]
    zoom_valid = zoom_df[zoom_df['Status'] != "N/A"]
    
    stats_str.append(f"Total samples with spatial keywords: {len(pan_df) + len(zoom_df)}")
    stats_str.append(f"Panning samples evaluated: {len(pan_valid)}")
    if len(pan_valid) > 0:
        pan_ok = sum(pan_valid['Status'].str.contains("✅"))
        stats_str.append(f"  Panning consistency: {pan_ok}/{len(pan_valid)} ({pan_ok/len(pan_valid)*100:.1f}%)")
    stats_str.append(f"Zoom samples evaluated: {len(zoom_valid)}")
    if len(zoom_valid) > 0:
        zoom_ok = sum(zoom_valid['Status'].str.contains("✅"))
        stats_str.append(f"  Zoom consistency: {zoom_ok}/{len(zoom_valid)} ({zoom_ok/len(zoom_valid)*100:.1f}%)")

    stats_str.append("\n" + "="*50)
    stats_str.append("PANNING SAMPLES DETAILS")
    stats_str.append("="*50)
    for _, row in pan_df.iterrows():
        stats_str.append(f"Id: {row['Id']} | Status: {row['Status']} | Reason: {row['Reason']}")
        stats_str.append(f"Sentence: {row['Sentence']}")
        stats_str.append("-" * 50)

    log_progress("\n".join(stats_str))
    log_progress("Holdout diagnosis finished successfully!")

if __name__ == "__main__":
    main()
