import os
import glob
import pandas as pd

working_dir = "C:/Users/user/Desktop/서울대"
print("All CSV files in workspace:")
csv_files = glob.glob(os.path.join(working_dir, "*.csv"))
for f in sorted(csv_files):
    stat = os.stat(f)
    print(f"File: {os.path.basename(f)}, Size: {stat.st_size} bytes, Modified: {stat.st_mtime}")

print("\nDetail of files:")
for f in sorted(csv_files):
    if "snu_clip" in f or "train" in f or "errors" in f:
        continue
    try:
        df = pd.read_csv(f)
        print(f"File: {os.path.basename(f)}, Rows: {len(df)}, Columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"  First 2 rows:\n{df.head(2).to_string()}")
    except Exception as e:
        print(f"  Error reading {os.path.basename(f)}: {e}")
