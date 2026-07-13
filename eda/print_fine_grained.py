import pandas as pd
import os
import sys

# Ensure UTF-8 output formatting for stdout on Windows
sys.stdout.reconfigure(encoding='utf-8')

CACHE_CSV = "./eda/dataset_mse_cache.csv"

if os.path.exists(CACHE_CSV):
    df = pd.read_csv(CACHE_CSV)
    # Filter for Fine-grained and sort by Avg_MSE ascending (lowest pixel = most static)
    fg = df[df['Type'] == 'Fine-grained'].sort_values(by='Avg_MSE')
    
    print(f"Total Fine-grained Samples detected: {len(fg)} / 9,535\n")
    print("| Index | Id | Avg_MSE | Sentence |")
    print("| :---: | :--- | :---: | :--- |")
    
    # Print top 50
    for idx, (row_idx, row) in enumerate(fg.head(50).iterrows()):
        sentence_clean = row['Sentence'].replace('|', '\\|') # Escape markdown tables
        print(f"| {idx + 1} | `{row['Id']}` | {row['Avg_MSE']:.2f} | {sentence_clean} |")
else:
    print(f"Error: Cache file not found at {CACHE_CSV}")
