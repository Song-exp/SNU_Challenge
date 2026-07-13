import os
import glob
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def resize_image_aspect_ratio(img_path, output_path, max_dim=448):
    """
    Resizes an image maintaining aspect ratio such that the longer side is at most max_dim.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with Image.open(img_path) as img:
            w, h = img.size
            if max(w, h) > max_dim:
                # Calculate scale factor
                scale = max_dim / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                # Use Lanczos/Resampling for high quality
                img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                img_resized.save(output_path, "JPEG", quality=90)
            else:
                # If already smaller, just save as copy
                img.save(output_path, "JPEG", quality=90)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def process_split(data_dir, split_name, output_dir, max_dim=448, num_workers=8):
    src_split_dir = os.path.join(data_dir, split_name)
    dst_split_dir = os.path.join(output_dir, split_name)
    
    if not os.path.exists(src_split_dir):
        print(f"Source directory {src_split_dir} does not exist.")
        return
        
    print(f"Finding images in {src_split_dir}...")
    # Find all sample directories
    sample_ids = [d for d in os.listdir(src_split_dir) if os.path.isdir(os.path.join(src_split_dir, d))]
    print(f"Found {len(sample_ids)} samples.")
    
    tasks = []
    for sample_id in sample_ids:
        src_sample_dir = os.path.join(src_split_dir, sample_id)
        dst_sample_dir = os.path.join(dst_split_dir, sample_id)
        
        img_files = glob.glob(os.path.join(src_sample_dir, "*.jpg"))
        for img_path in img_files:
            img_name = os.path.basename(img_path)
            dst_path = os.path.join(dst_sample_dir, img_name)
            tasks.append((img_path, dst_path))
            
    print(f"Total images to resize: {len(tasks)}")
    
    # Process in parallel using thread pool (good for I/O and PIL operations)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(
            executor.map(lambda t: resize_image_aspect_ratio(t[0], t[1], max_dim), tasks),
            total=len(tasks),
            desc=f"Resizing {split_name}"
        ))

if __name__ == "__main__":
    import argparse
    
    # Default parameters (relative paths for cross-platform compatibility)
    DATA_DIR = "./snuaichallenge_data"
    OUTPUT_DIR = "./snuaichallenge_data_resized"
    MAX_DIM = 448 # Default max dimension (multiple of 28 is best for Qwen2-VL)
    
    print("--- Starting Image Pre-resize Caching Script ---")
    print(f"Source Data Dir: {DATA_DIR}")
    print(f"Output Resized Dir: {OUTPUT_DIR}")
    print(f"Max Dimension: {MAX_DIM}")
    
    # Run for train
    process_split(DATA_DIR, "train", OUTPUT_DIR, max_dim=MAX_DIM, num_workers=8)
    
    # Run for test
    process_split(DATA_DIR, "test", OUTPUT_DIR, max_dim=MAX_DIM, num_workers=8)
    
    print("\nResizing completed successfully.")
