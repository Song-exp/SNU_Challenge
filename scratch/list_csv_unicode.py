import os
import glob
import pandas as pd

working_dir = "C:/Users/user/Desktop/서울대"
csv_files = glob.glob(os.path.join(working_dir, "*.csv"))

print("Listing all CSV files with exact unicode representation:")
for f in sorted(csv_files):
    base = os.path.basename(f)
    print(f"Name: {repr(base)}, Size: {os.path.getsize(f)}, Modified: {os.path.getmtime(f)}")
