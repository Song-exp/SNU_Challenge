import os
import pandas as pd

working_dir = "C:/Users/user/Desktop/서울대"
report_path = os.path.join(working_dir, "untrained_samples_report.csv")
ids_csv_path = os.path.join(working_dir, "untrained_ids.csv")

if os.path.exists(report_path):
    df = pd.read_csv(report_path)
    # Extract only the 'Id' column
    df_ids = df[["Id"]]
    df_ids.to_csv(ids_csv_path, index=False)
    print(f"Success: Extracted {len(df_ids)} IDs and saved to: {ids_csv_path}")
else:
    print("Error: untrained_samples_report.csv not found!")
