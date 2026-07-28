import os
import shutil
import pandas as pd
import numpy as np

# Path configurations
WORKSPACE_DIR = "C:/Users/user/Desktop/서울대"
FEATURES_CSV = os.path.join(WORKSPACE_DIR, "snu_clip_features.csv")
TRAIN_CSV = os.path.join(WORKSPACE_DIR, "train_검토_최종_완료.csv")
IMAGE_DIR = os.path.join(WORKSPACE_DIR, "snuaichallenge_data/train")

# Brain artifact directory (destination for copied images)
BRAIN_DIR = "C:/Users/user/.gemini/antigravity-cli/brain/d9b8332e-fa7d-461b-9ae8-b7ac6e6acc0e"
BRAIN_IMAGES_DIR = os.path.join(BRAIN_DIR, "images")
REPORT_PATH = os.path.join(BRAIN_DIR, "unsupervised_scene_grouping_report.md")

os.makedirs(BRAIN_IMAGES_DIR, exist_ok=True)

def find_scene_groups(distances):
    # distances dict: { (0,1): dist_12, (0,2): dist_13, ... }
    # 6 pairs sorted by distance (ascending)
    sorted_pairs = sorted(distances.items(), key=lambda x: x[1])
    
    # Check extreme cases using empirical thresholds derived from EDA
    max_dist = sorted_pairs[-1][1]
    min_dist = sorted_pairs[0][1]
    
    # 1. 0 cuts check: if the maximum distance between any frames is very small, they are all in 1 scene
    if max_dist < 0.20:
        return [[0, 1, 2, 3]], 0
        
    # 2. 3 cuts check: if even the most similar frames are very different, they are all in different scenes
    if min_dist > 0.22:
        return [[0], [1], [2], [3]], 3
        
    # 3. Otherwise, perform adaptive gap splitting
    max_gap = -1.0
    split_idx = -1
    for k in range(len(sorted_pairs) - 1):
        gap = sorted_pairs[k+1][1] - sorted_pairs[k][1]
        if gap > max_gap:
            max_gap = gap
            split_idx = k
            
    # Same scene pairs are those with distances below the maximum gap split point
    same_scene_pairs = [pair for pair, dist in sorted_pairs[:split_idx + 1]]
    
    # Union-Find
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
        
    # Group indices by their root parent
    groups_dict = {}
    for idx in range(4):
        root = find(idx)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(idx)
        
    groups = list(groups_dict.values())
    cuts = len(groups) - 1
    return groups, cuts

def main():
    print("Loading features and train datasets...")
    feat_df = pd.read_csv(FEATURES_CSV)
    train_df = pd.read_csv(TRAIN_CSV, encoding='cp949')
    
    # Merge datasets to get the sentence along with distances
    merged_df = pd.merge(train_df[['Id', 'Sentence', 'Answer', 'Input_1', 'Input_2', 'Input_3', 'Input_4']], feat_df, on='Id')
    
    # Take the first 100 rows for the validation set
    test_df = merged_df.iloc[:100].copy()
    
    results = []
    
    for idx, row in test_df.iterrows():
        sample_id = str(row['Id'])
        # Map 6 distances in shuffled order (1-indexed in CSV: dist_12 represents 0-1 indices)
        distances = {
            (0, 1): float(row['dist_12']),
            (0, 2): float(row['dist_13']),
            (0, 3): float(row['dist_14']),
            (1, 2): float(row['dist_23']),
            (1, 3): float(row['dist_24']),
            (2, 3): float(row['dist_34'])
        }
        
        groups, predicted_cuts = find_scene_groups(distances)
        
        results.append({
            'Id': sample_id,
            'Sentence': row['Sentence'],
            'Answer': row['Answer'],
            'files': [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']],
            'groups': groups,
            'predicted_cuts': predicted_cuts
        })
        
    # Select 5 representative samples to copy images and embed in report
    # We want to find: 0 cuts (1 sample), 1 cut (3+1) (1 sample), 1 cut (2+2) (1 sample), 2 cuts (1 sample), 3 cuts (1 sample)
    selected_samples = {}
    
    for item in results:
        cuts = item['predicted_cuts']
        groups = item['groups']
        # Check specific structures
        if cuts == 0 and '0_cuts' not in selected_samples:
            selected_samples['0_cuts'] = item
        elif cuts == 1:
            group_lens = sorted([len(g) for g in groups])
            if group_lens == [1, 3] and '1_cut_3_1' not in selected_samples:
                selected_samples['1_cut_3_1'] = item
            elif group_lens == [2, 2] and '1_cut_2_2' not in selected_samples:
                selected_samples['1_cut_2_2'] = item
        elif cuts == 2 and '2_cuts' not in selected_samples:
            selected_samples['2_cuts'] = item
        elif cuts == 3 and '3_cuts' not in selected_samples:
            selected_samples['3_cuts'] = item
            
        # Break early if we found all 5 types
        if len(selected_samples) == 5:
            break
            
    # If we couldn't find some, pick whatever is available
    for item in results:
        cuts = item['predicted_cuts']
        label = f"{cuts}_cuts_fallback"
        if len(selected_samples) < 5 and label not in selected_samples and item['Id'] not in [x['Id'] for x in selected_samples.values()]:
            selected_samples[label] = item
            
    print(f"Selected {len(selected_samples)} representative samples for visual inspection.")
    
    # Copy images for the selected samples to the brain folder
    for key, item in selected_samples.items():
        sample_id = item['Id']
        ans = eval(item['Answer'])  # e.g., [3, 1, 4, 2]
        shuffled_files = item['files']
        
        # Sort files chronologically for clear visual checking
        ordered_files = [None] * 4
        for i, pos in enumerate(ans):
            ordered_files[pos - 1] = shuffled_files[i]
            
        copied_paths = []
        for i, f in enumerate(ordered_files):
            src_path = os.path.join(IMAGE_DIR, sample_id, f)
            dest_filename = f"{sample_id}_frame_{i+1}.jpg"
            dest_path = os.path.join(BRAIN_IMAGES_DIR, dest_filename)
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                copied_paths.append(dest_path)
            else:
                copied_paths.append(None)
        item['copied_paths'] = copied_paths

    # Generate Markdown Report
    print("Generating Markdown Report...")
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("# 🎬 무감독 장면 그룹화 (Unsupervised Scene Grouping) 검증 리포트\n\n")
        f.write("이 리포트는 병철 님 팀의 **'이미지 단위 방법론 설계서'**에 기술된 **[상대적 Gap 분할 + Union-Find]** 알고리즘을 100개 데이터 샘플에 적용하여 장면 전환 및 프레임 그룹화를 탐지한 실증 검증서입니다.\n\n")
        
        f.write("## 1. 🌟 대표 유형별 시각적 검증 (5선)\n")
        f.write("각 프레임은 시간 순서대로 정렬하여 나열했습니다. 하단의 장면 그룹화 예측 결과를 통해 알고리즘이 화면 경계를 얼마나 정밀하게 분류했는지 육안으로 확인하실 수 있습니다.\n\n")
        
        # Write the 5 representative types
        type_labels = {
            '0_cuts': "🟢 장면 전환 0회 (동일 씬 원컷)",
            '1_cut_3_1': "🎬 장면 전환 1회 (3+1 분할 구조)",
            '1_cut_2_2': "🎬 장면 전환 1회 (2+2 분할 구조)",
            '2_cuts': "🎬 장면 전환 2회 (3개 씬 분할 구조)",
            '3_cuts': "🎬 장면 전환 3회 (4개 씬 분할 구조)"
        }
        
        for key, item in selected_samples.items():
            label = type_labels.get(key, f"🎬 장면 전환 {item['predicted_cuts']}회")
            f.write(f"### {label} (Sample ID: `{item['Id']}`)\n")
            f.write(f"**문장**: *\"{item['Sentence']}\"*\n\n")
            
            # Display the 4 frames in a row
            f.write("| Frame 1 | Frame 2 | Frame 3 | Frame 4 |\n")
            f.write("|:---:|:---:|:---:|:---:|\n")
            
            img_markdowns = []
            for path in item['copied_paths']:
                if path and os.path.exists(path):
                    # Relative path for brain artifact markdown embedding
                    rel_path = "./images/" + os.path.basename(path)
                    img_markdowns.append(f"![frame]({rel_path})")
                else:
                    img_markdowns.append("[이미지 없음]")
                    
            f.write(f"| {img_markdowns[0]} | {img_markdowns[1]} | {img_markdowns[2]} | {img_markdowns[3]} |\n\n")
            
            # Format group description (e.g. Frame 1 & 2 are Scene A, Frame 3 & 4 are Scene B)
            # Find the actual shuffled index mapping to sorted chronological frames
            ans = eval(item['Answer'])
            # groups have indices (0, 1, 2, 3) representing shuffled frames.
            # Convert shuffled index to chronological order (1 to 4) using `ans` (which maps shuffled -> chronological)
            chrono_groups = []
            for g in item['groups']:
                # pos is 1-indexed, representing the chronological position of shuffled index `idx`
                chrono_g = sorted([ans[idx] for idx in g])
                chrono_groups.append(chrono_g)
                
            group_desc = []
            for i, cg in enumerate(chrono_groups):
                frames_str = ", ".join([f"Frame {x}" for x in cg])
                group_desc.append(f"  * **장면 그룹 {chr(65+i)}**: {frames_str}")
                
            f.write("**[알고리즘 장면 분할 결과]**\n")
            f.write(f"- **예측된 장면 전환 횟수**: `{item['predicted_cuts']}회`\n")
            f.write("- **프레임 매핑 그룹**:\n")
            f.write("\n".join(group_desc) + "\n\n")
            f.write("---\n\n")
            
        f.write("## 2. 📊 100개 샘플 검수 예측 리스트 요약\n")
        f.write("아래 표는 처음 100개 샘플의 전체 요약 리스트입니다. (장면 그룹의 숫자는 시간 순서대로 정렬된 프레임 1~4번을 뜻합니다.)\n\n")
        f.write("| Sample ID | 문장 (Sentence) | 예측 장면 수 | 프레임 그룹화 결과 |\n")
        f.write("|:---:|:---|:---:|:---|\n")
        
        for item in results:
            ans = eval(item['Answer'])
            chrono_groups = []
            for g in item['groups']:
                chrono_g = sorted([ans[idx] for idx in g])
                chrono_groups.append(chrono_g)
            # Convert groups to string representation e.g. "{1, 2}, {3, 4}"
            groups_str = " | ".join([f"{{{', '.join(map(str, cg))}}}" for cg in chrono_groups])
            f.write(f"| `{item['Id']}` | {item['Sentence']} | `{item['predicted_cuts'] + 1}개` | `{groups_str}` |\n")
            
    print(f"Report successfully generated at: {REPORT_PATH}")

if __name__ == "__main__":
    main()
