import os
import glob
import shutil
import json

def copy_checkpoint_to_working():
    """
    Finds the latest QLoRA adapter checkpoint in the Kaggle input directories,
    copies it to the working directory, and updates the metadata to resume training.
    """
    ckpt_paths = glob.glob("/kaggle/input/**/ckpt", recursive=True)
    if ckpt_paths:
        shutil.copytree(ckpt_paths[0], "/kaggle/working/ckpt", dirs_exist_ok=True)
        print("Checkpoint files copied successfully to working directory!")

        meta_path = "/kaggle/working/ckpt/meta.json"
        if os.path.exists(meta_path):
            saved_step = json.load(open(meta_path))["step"]
            print(f"Validation successful: Ready to resume from Step {saved_step}.")
        else:
            print("Warning: meta.json file not found in the checkpoint directory.")
    else:
        print("Error: No checkpoint directory found. Please ensure the prior session output is added to the input resources.")

if __name__ == "__main__":
    copy_checkpoint_to_working()
