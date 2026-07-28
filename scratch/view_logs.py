import os
import json

LOG_PATH = "C:/Users/user/.gemini/antigravity-cli/brain/1cca5105-d784-4635-802e-96a1d537a04e/.system_generated/logs/transcript.jsonl"
OUT_PATH = "C:/Users/user/Desktop/서울대/scratch/view_logs_output.txt"

def main():
    if not os.path.exists(LOG_PATH):
        print(f"Log path not found: {LOG_PATH}")
        return
        
    matches = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if "H9PMiH" in line:
                matches.append(f"--- Line {idx+1} ---")
                matches.append(line)
                
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        out.write("\n".join(matches))
    print(f"Done. Wrote matches to {OUT_PATH}")

if __name__ == "__main__":
    main()
