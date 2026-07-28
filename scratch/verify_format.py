import os
import pandas as pd
import ast

working_dir = "C:/Users/user/Desktop/서울대"
file_path = os.path.join(working_dir, "submission(최종).csv")

print(f"Verifying formatting for {os.path.basename(file_path)}:")
try:
    df = pd.read_csv(file_path)
    
    # 1. Row count check
    if len(df) == 819:
        print("  [PASS] Row count is exactly 819.")
    else:
        print(f"  [FAIL] Row count is {len(df)}, expected 819.")
        
    # 2. Columns check
    if list(df.columns) == ["Id", "Answer"]:
        print("  [PASS] Columns are exactly ['Id', 'Answer'].")
    else:
        print(f"  [FAIL] Columns are {df.columns}, expected ['Id', 'Answer'].")
        
    # 3. Null values check
    null_ids = df["Id"].isnull().sum()
    null_answers = df["Answer"].isnull().sum()
    if null_ids == 0 and null_answers == 0:
        print("  [PASS] No null or empty values found.")
    else:
        print(f"  [FAIL] Found nulls: Id nulls={null_ids}, Answer nulls={null_answers}.")
        
    # 4. Answer value format check
    valid_format = True
    format_errors = 0
    for idx, row in df.iterrows():
        ans_str = row["Answer"]
        try:
            val = ast.literal_eval(ans_str)
            if not isinstance(val, list) or len(val) != 4 or sorted(val) != [1, 2, 3, 4]:
                valid_format = False
                format_errors += 1
                if format_errors <= 3:
                    print(f"    Invalid answer value at row {idx}: {ans_str}")
        except Exception as e:
            valid_format = False
            format_errors += 1
            if format_errors <= 3:
                print(f"    Parse error at row {idx}: '{ans_str}' -> {e}")
                
    if valid_format:
        print("  [PASS] All Answer values are valid lists of [1, 2, 3, 4] permutations.")
    else:
        print(f"  [FAIL] Found {format_errors} rows with invalid Answer values.")

except Exception as e:
    print(f"  [FAIL] Failed to read or parse file: {e}")
