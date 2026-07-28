import os
import time

working_dir = "C:/Users/user/Desktop/서울대"
now = time.time()
print("Recently modified files (last 10 minutes):")
for root, dirs, files in os.walk(working_dir):
    for f in files:
        path = os.path.join(root, f)
        try:
            mtime = os.path.getmtime(path)
            age = now - mtime
            if age < 600: # 10 minutes
                print(f"File: {repr(f)}, Age: {age:.1f}s, Path: {path}")
        except Exception as e:
            pass
