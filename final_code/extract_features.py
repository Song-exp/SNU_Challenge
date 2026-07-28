# ================================================================================
# SNU AI Challenge — Batch Feature Extraction for Sentences
# ================================================================================

import os
import pandas as pd
from flag_detector import OrthogonalFlagDetector

def extract_features(input_path, output_path, name):
    if not os.path.exists(input_path):
        print(f"Error: [{name}] input file not found at: {input_path}")
        return
    
    print(f"\n===== Extracting features for {name} dataset =====")
    df = pd.read_csv(input_path)
    detector = OrthogonalFlagDetector()
    
    results = []
    total = len(df)
    for i, row in df.iterrows():
        sentence = row['Sentence']
        features = detector.process_sentence(sentence)
        row_dict = row.to_dict()
        row_dict.update(features)
        results.append(row_dict)
        
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"Progress: {i + 1}/{total} completed.")
            
    df_featured = pd.DataFrame(results)
    df_featured.to_csv(output_path, index=False)
    print(f"Featured {name} dataset saved to: {output_path}")

if __name__ == "__main__":
    extract_features("train.csv", "train_with_flags.csv", "Train")
    extract_features("test.csv", "test_with_flags.csv", "Test")
