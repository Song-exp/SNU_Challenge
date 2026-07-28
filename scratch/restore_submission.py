import os
import ast
import pandas as pd
import glob

def restore():
    # 1. Search for submission files in current directory or Downloads folder
    download_dir = os.path.expanduser("~/Downloads")
    
    # We prioritize the file named "submission (1).csv" in the current directory
    # since we know it is there!
    current_cands = glob.glob("submission (1).csv") + glob.glob("submission*.csv") + glob.glob("*.csv")
    download_cands = glob.glob(os.path.join(download_dir, "submission*.csv"))
    
    cands = current_cands + download_cands
    
    if not cands:
        print("[Error] No CSV files starting with 'submission' found.")
        print("Please move the 0.44 score CSV file to the current folder.")
        return
        
    print("[Info] Found CSV candidate files:")
    for idx, c in enumerate(cands):
        print(f"  [{idx}] {c}")
        
    # We choose the first candidate (which will be submission (1).csv in the current directory)
    target_file = cands[0]
    print(f"\n[Info] Targeting file for recovery: {target_file}")
    
    try:
        df = pd.read_csv(target_file)
        if "Id" not in df.columns or "Answer" not in df.columns:
            print("[Error] Invalid Kaggle submission format (missing Id or Answer columns).")
            return
            
        restored_recs = []
        for _, row in df.iterrows():
            ans = ast.literal_eval(row["Answer"])
            
            # [Symmetric Permutation Restoration] Invert the permutation
            sub = [0] * 4
            for i, n in enumerate(ans):
                sub[n - 1] = i + 1
            restored_recs.append({"Id": row["Id"], "Answer": str(sub)})
            
        out_df = pd.DataFrame(restored_recs)
        out_path = os.path.join(os.path.dirname(target_file), "submission_restored_final.csv")
        out_df.to_csv(out_path, index=False)
        print(f"\n[Success] Restored submission file successfully created at:")
        print("Path: " + os.path.abspath(out_path))
        print("Submit this file to Kaggle to restore the 0.88+ score!")
        
    except Exception as e:
        print(f"[Error] Exception occurred: {e}")

if __name__ == "__main__":
    restore()
