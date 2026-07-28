import os
import shutil
import sys

# Use relative paths to avoid Windows path encoding issues with "서울대"
FINAL_CODE_DIR = "final_code"
EDA_DIR = "eda"

os.makedirs(FINAL_CODE_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

# 1. Copy final pipeline and helper scripts to final_code/
COPIES = {
    os.path.join("kaggle", "FINAL_8B_v2.py"): os.path.join(FINAL_CODE_DIR, "FINAL_8B_v2.py"),
    os.path.join("kaggle", "INFER_ONLY_K4.py"): os.path.join(FINAL_CODE_DIR, "INFER_ONLY_K4.py"),
    os.path.join("eda", "clip_labeling_model.py"): os.path.join(FINAL_CODE_DIR, "clip_labeling_model.py"),
    os.path.join("src", "features", "flag_detector.py"): os.path.join(FINAL_CODE_DIR, "flag_detector.py"),
    os.path.join("scripts", "extract_features.py"): os.path.join(FINAL_CODE_DIR, "extract_features.py"),
    os.path.join("scripts", "analyze_pipeline.py"): os.path.join(FINAL_CODE_DIR, "analyze_pipeline.py"),
}

print("--- 1. Copying Final Pipeline Files ---")
for src, dst in COPIES.items():
    try:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied: {src} -> {dst}")
        else:
            # Check if it was already moved to backup
            # e.g., if kaggle is already at eda/kaggle_backup
            backup_src = os.path.join(EDA_DIR, src.replace("kaggle", "kaggle_backup").replace("src", "src_backup").replace("scripts", "scripts_backup"))
            if os.path.exists(backup_src):
                shutil.copy2(backup_src, dst)
                print(f"Copied from backup: {backup_src} -> {dst}")
            else:
                print(f"Warning: Source not found: {src}")
    except Exception as e:
        print(f"Error copying {src}: {e}")

# 2. Move root directories to eda/ backups (except final_code, eda, .git, .agents, snuaichallenge_data)
# Note: kaggle, src, scripts, splits, outputs were already moved in the previous run.
# Let's clean up and double check any remaining folders.
DIRS_TO_MOVE = [
    "kaggle",
    "src",
    "scripts",
    "splits",
    "outputs",
    "검수기프로그램",
    "이미지전처리",
]

print("\n--- 2. Moving Subdirectories to eda/ ---")
for dname in DIRS_TO_MOVE:
    if os.path.exists(dname) and dname not in [FINAL_CODE_DIR, EDA_DIR]:
        dst_dir = os.path.join(EDA_DIR, f"{dname}_backup" if dname not in ["검수기프로그램", "이미지전처리"] else dname)
        try:
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.move(dname, dst_dir)
            print(f"Moved directory: {dname} -> {dst_dir}")
        except Exception as e:
            print(f"Error moving directory {dname}: {e}")

# 3. Move root files to eda/ (except README.md, requirements.txt, snu_clip_features.csv, .gitignore, snuaichallenge_data.zip, scratch)
KEEP_FILES = {
    "README.md",
    "requirements.txt",
    "snu_clip_features.csv",
    ".gitignore",
    "snuaichallenge_data.zip",
}

print("\n--- 3. Moving Root Files to eda/ ---")
for fname in os.listdir("."):
    if os.path.isfile(fname):
        if fname in KEEP_FILES:
            print(f"Kept in root: {fname}")
            continue
        # Skip scratch files
        if fname == "organize.py":
            continue
        dst_path = os.path.join(EDA_DIR, fname)
        try:
            if os.path.exists(dst_path):
                os.remove(dst_path)
            shutil.move(fname, dst_path)
            print(f"Moved file: {fname} -> eda/{fname}")
        except Exception as e:
            print(f"Error moving file {fname}: {e} (skipping)")

print("\nOrganize script finished successfully!")
