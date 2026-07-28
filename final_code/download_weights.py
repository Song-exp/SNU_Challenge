import os
import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download Qwen3-VL-8B-Instruct base model from Hugging Face")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--out-dir", default="./models/Qwen3-VL-8B-Instruct", help="Local target directory")
    args = parser.parse_args()

    print(f"Downloading {args.model_id} to {args.out_dir}...")
    os.makedirs(args.out_dir, exist_ok=True)
    snapshot_download(
        repo_id=args.model_id,
        local_dir=args.out_dir,
        local_dir_use_symlinks=False,
    )
    print("Download completed successfully.")

if __name__ == "__main__":
    main()
