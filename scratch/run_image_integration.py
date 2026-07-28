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
FEATURES_CSV = os.path.join(WORKSPACE_DIR, "snu_clip_features.csv")
IMAGE_DIR = os.path.join(WORKSPACE_DIR, "snuaichallenge_data/train")

def main():
    if not os.path.exists(HOLDOUT_CSV):
        print(f"Holdout CSV not found: {HOLDOUT_CSV}")
        return
    if not os.path.exists(FEATURES_CSV):
        print(f"Features CSV not found: {FEATURES_CSV}")
        return

    df_holdout = pd.read_csv(HOLDOUT_CSV)
    df_feat = pd.read_csv(FEATURES_CSV)
    
    # Merge holdout with features to get scene cuts count
    df = pd.merge(df_holdout, df_feat[['Id', 'predicted_scene_cuts']], on='Id', how='left')
    print(f"Loaded {len(df)} samples from holdout_300.csv and merged with features.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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
        
        # Determine number of scene groups (cuts + 1)
        scene_cuts = row['predicted_scene_cuts']
        scene_groups = int(scene_cuts + 1) if not pd.isna(scene_cuts) else 4
        
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
                'Subject': 'person',
                'DetectCount': 0,
                'SceneGroups': scene_groups,
                'Status': '➖ 해당 없음 (N/A)',
                'Reason': 'N/A',
                'IsPan': False,
                'IsZoom': False
            })
            continue

        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        ordered_files = [None] * 4
        for i, pos in enumerate(ans):
            ordered_files[pos - 1] = shuffled_files[i]

        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in ordered_files]
        if not all(os.path.exists(p) for p in img_paths):
            results.append({
                'Id': sample_id,
                'Sentence': row['Sentence'],
                'Subject': 'person',
                'DetectCount': 0,
                'SceneGroups': scene_groups,
                'Status': '➖ 해당 없음 (N/A)',
                'Reason': 'N/A',
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

        status = "➖ 해당 없음 (N/A)"
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
                    status = "➖ 해당 없음 (N/A)"
                    reason = "Mixed pan direction"
                elif expected_right:
                    if x_end > x_start:
                        status = "✅ 일치 (Consistent)"
                        reason = f"Object moved right (X: {x_start:.2f} -> {x_end:.2f})"
                    else:
                        status = "❌ 불일치 (Inconsistent)"
                        reason = f"Object did not move right (X: {x_start:.2f} -> {x_end:.2f})"
                elif expected_left:
                    if x_end < x_start:
                        status = "✅ 일치 (Consistent)"
                        reason = f"Object moved left (X: {x_start:.2f} -> {x_end:.2f})"
                    else:
                        status = "❌ 불일치 (Inconsistent)"
                        reason = f"Object did not move left (X: {x_start:.2f} -> {x_end:.2f})"
            
            elif is_zoom_sample:
                valid_idx = [i for i, a in enumerate(areas) if a > 0]
                first_valid = valid_idx[0]
                last_valid = valid_idx[-1]
                a_start = areas[first_valid]
                a_end = areas[last_valid]

                if has_zoom_in and has_zoom_out:
                    status = "➖ 해당 없음 (N/A)"
                    reason = "Mixed zoom direction"
                elif has_zoom_in:
                    if a_end > a_start:
                        status = "✅ 일치 (Consistent)"
                        reason = f"Zoom-in matched area increase ({a_start*100:.1f}% -> {a_end*100:.1f}%)"
                    else:
                        status = "❌ 불일치 (Inconsistent)"
                        reason = f"Zoom-in failed (area decreased: {a_start*100:.1f}% -> {a_end*100:.1f}%)"
                elif has_zoom_out:
                    if a_end < a_start:
                        status = "✅ 일치 (Consistent)"
                        reason = f"Zoom-out matched area decrease ({a_start*100:.1f}% -> {a_end*100:.1f}%)"
                    else:
                        status = "❌ 불일치 (Inconsistent)"
                        reason = f"Zoom-out failed (area increased: {a_start*100:.1f}% -> {a_end*100:.1f}%)"

        results.append({
            'Id': sample_id,
            'Sentence': row['Sentence'],
            'Subject': 'person',
            'DetectCount': detect_count,
            'SceneGroups': scene_groups,
            'Status': status,
            'Reason': reason,
            'IsPan': is_pan_sample,
            'IsZoom': is_zoom_sample
        })

    res_df = pd.DataFrame(results)
    
    # Generate statistics
    pan_df = res_df[res_df['IsPan']]
    zoom_df = res_df[res_df['IsZoom'] & ~res_df['IsPan']]
    
    pan_valid = pan_df[pan_df['Status'] != "➖ 해당 없음 (N/A)"]
    zoom_valid = zoom_df[zoom_df['Status'] != "➖ 해당 없음 (N/A)"]
    
    pan_ok = sum(pan_valid['Status'].str.contains("✅"))
    zoom_ok = sum(zoom_valid['Status'].str.contains("✅"))
    total_valid = len(pan_valid) + len(zoom_valid)
    total_ok = pan_ok + zoom_ok
    
    print("\n" + "="*50)
    print("FINAL SUMMARY METRICS")
    print("="*50)
    print(f"Total holdout samples: {len(df)}")
    print(f"Average objects detected per sample: {res_df['DetectCount'].mean():.2f}/4 ({res_df['DetectCount'].mean()/4*100:.1f}%)")
    print(f"Average scene groups count (CLIP): {res_df['SceneGroups'].mean():.2f}")
    print(f"Evaluated physical consistency samples: {total_valid}")
    print(f"Overall Consistency Rate: {total_ok}/{total_valid} ({total_ok/total_valid*100:.1f}%)")
    print(f"  Zoom Consistency Rate: {zoom_ok}/{len(zoom_valid)} ({zoom_ok/len(zoom_valid)*100:.1f}%)")
    print(f"  Panning Consistency Rate: {pan_ok}/{len(pan_valid)} ({pan_ok/len(pan_valid)*100:.1f}%)")

if __name__ == "__main__":
    main()
