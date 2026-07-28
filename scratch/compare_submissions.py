import os
import pandas as pd

working_dir = "C:/Users/user/Desktop/서울대"
new_file = os.path.join(working_dir, "submission(최종).csv")

# Candidates for comparison
candidates = [
    "submission (1).csv",
    "submission.csv",
    "submission_restored_correct.csv",
    "submission_restored_final.csv"
]

print(f"Comparing new file: {os.path.basename(new_file)}")
df_new = pd.read_csv(new_file)
print(f"New file shape: {df_new.shape}")

for name in candidates:
    path = os.path.join(working_dir, name)
    if not os.path.exists(path):
        print(f"File {name} does not exist.")
        continue
    df_cand = pd.read_csv(path)
    if df_cand.shape != df_new.shape:
        print(f"File {name} has different shape: {df_cand.shape}")
        continue
    
    # Calculate match count and ratio
    matches = (df_cand["Answer"] == df_new["Answer"]).sum()
    ratio = (matches / len(df_new)) * 100
    print(f"Comparison with {name}: Match count: {matches} / {len(df_new)} ({ratio:.2f}%)")
