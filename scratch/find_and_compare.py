import os
import ast
import pandas as pd
import glob

def find_and_compare():
    greedy_file = "submission.csv"  # The greedy file user just put in C:\Users\user\Desktop\서울대
    
    if not os.path.exists(greedy_file):
        print(f"[Error] Greedy file '{greedy_file}' not found in the current workspace directory.")
        print("Available files in this directory:")
        print(os.listdir("."))
        return
        
    download_dir = os.path.expanduser("~/Downloads")
    # Find all CSV files in Downloads containing "submission"
    dl_files = glob.glob(os.path.join(download_dir, "*submission*.csv")) + glob.glob(os.path.join(download_dir, "*submission*"))
    # Also find files in current directory
    local_files = glob.glob("submission*.csv")
    
    all_files = list(set(dl_files + local_files))
    
    print(f"[Info] Found {len(all_files)} potential submission files:")
    for f in all_files:
        print(f"  - {f} (Size: {os.path.getsize(f)} bytes)")
        
    df_greedy = pd.read_csv(greedy_file)
    total = len(df_greedy)
    
    for fpath in all_files:
        if os.path.abspath(fpath) == os.path.abspath(greedy_file):
            continue # Skip comparing the greedy file to itself
            
        try:
            df_other = pd.read_csv(fpath)
            if len(df_other) != total:
                continue # Skip if row count mismatch
            if "Id" not in df_other.columns or "Answer" not in df_other.columns:
                continue
                
            raw_match = 0
            inverted_match = 0
            
            restored_recs = []
            
            for i in range(total):
                id_g = df_greedy.iloc[i]["Id"]
                id_o = df_other.iloc[i]["Id"]
                if id_g != id_o:
                    break
                    
                ans_g = ast.literal_eval(df_greedy.iloc[i]["Answer"])
                ans_o = ast.literal_eval(df_other.iloc[i]["Answer"])
                
                # Invert ans_o
                ans_o_inv = [0] * 4
                for idx, val in enumerate(ans_o):
                    ans_o_inv[val - 1] = idx + 1
                    
                if ans_g == ans_o:
                    raw_match += 1
                if ans_g == ans_o_inv:
                    inverted_match += 1
                    
                restored_recs.append({"Id": id_g, "Answer": str(ans_o_inv)})
                
            print(f"\n--- Comparing with {os.path.basename(fpath)} ---")
            print(f"  Raw match rate: {raw_match}/{total} ({raw_match/total*100:.2f}%)")
            print(f"  Inverted match rate: {inverted_match}/{total} ({inverted_match/total*100:.2f}%)")
            
            # If the inverted match rate is very high (meaning this is the inverted likelihood file),
            # save it as our final restored submission!
            if inverted_match > 700: # Over 85% match rate
                out_path = "submission_restored_correct.csv"
                pd.DataFrame(restored_recs).to_csv(out_path, index=False)
                print(f"  [SUCCESS] This is the correct likelihood file! Saved restored file to: {os.path.abspath(out_path)}")
                
        except Exception as e:
            # Silently ignore format errors for non-submission CSVs
            pass

if __name__ == "__main__":
    find_and_compare()
