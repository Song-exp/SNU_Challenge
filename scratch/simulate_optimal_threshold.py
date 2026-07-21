import numpy as np

# Let's load the collected data from the previous experiment (calculated in calculate_crop_clip_distances.py)
# same_subject_dists: Mean: 0.2140, Median: 0.2132, Min: 0.1056, Max: 0.2863 (N=24)
# diff_subject_dists: Mean: 0.3374, Median: 0.3292, Min: 0.1498, Max: 0.5500 (N=327)

def find_optimal_threshold():
    # Simulated distribution based on the exact statistics of our holdout empirical run
    np.random.seed(42)
    
    # Generate representative normal distributions based on our real empirical sample statistics
    # to simulate a larger scale N=1000 experiment for threshold calibration
    same_dists = np.random.normal(loc=0.2140, scale=0.0517, size=500)
    # Clip to physical limits observed (min ~0.10)
    same_dists = np.clip(same_dists, 0.10, 0.40)
    
    diff_dists = np.random.normal(loc=0.3374, scale=0.0909, size=1500)
    # Clip to physical limits observed (min ~0.15)
    diff_dists = np.clip(diff_dists, 0.15, 0.60)
    
    print("=== Simulated Large-Scale Crop CLIP Threshold Sweep Search (N=2000) ===")
    print(f"Same-Subject simulated pairs: {len(same_dists)}")
    print(f"Diff-Subject simulated pairs: {len(diff_dists)}\n")
    
    best_t_f1 = 0
    best_f1 = 0
    best_t_acc = 0
    best_acc = 0
    
    # Sweep threshold t from 0.10 to 0.50 with step 0.01
    thresholds = np.arange(0.10, 0.50, 0.01)
    
    print(f"{'Threshold':<10} | {'False Alarm':<12} | {'Miss Rate':<10} | {'F1-Score':<10} | {'Balanced Acc':<12}")
    print("-" * 65)
    
    for t in thresholds:
        # Task: Binary Classification
        # Positive Class (1): Identity Jump (Different Subject) -> We want to detect this!
        # Negative Class (0): Same Subject -> We want to keep this!
        
        # True Positive (TP): Different subject and distance > t (Correctly detected jump)
        tp = np.sum(diff_dists > t)
        # False Positive (FP): Same subject and distance > t (Falsely detected jump - False Alarm)
        fp = np.sum(same_dists > t)
        # True Negative (TN): Same subject and distance <= t (Correctly kept same subject)
        tn = np.sum(same_dists <= t)
        # False Negative (FN): Different subject and distance <= t (Missed identity jump)
        fn = np.sum(diff_dists <= t)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Balanced Accuracy = (Sensitivity + Specificity) / 2
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        balanced_acc = (sensitivity + specificity) / 2
        
        # Print selection of thresholds
        if abs(t * 100 % 5) < 0.1:  # print every 0.05 step
            fp_rate = fp / len(same_dists) * 100
            fn_rate = fn / len(diff_dists) * 100
            print(f"{t:9.2f} | {fp_rate:10.1f}% | {fn_rate:8.1f}% | {f1:9.4f} | {balanced_acc*100:10.2f}%")
            
        if f1 > best_f1:
            best_f1 = f1
            best_t_f1 = t
            
        if balanced_acc > best_acc:
            best_acc = balanced_acc
            best_t_acc = t
            
    print("-" * 65)
    print(f"[*] Best Threshold by F1-Score (Detecting Jumps):  {best_t_f1:.2f} (F1: {best_f1:.4f})")
    print(f"[*] Best Threshold by Balanced Accuracy:            {best_t_acc:.2f} (Acc: {best_acc*100:.2f}%)")
    print("\n* Note: Since our objective is to maximize sequence accuracy by keeping as much good trajectory data as possible,")
    print("  minimizing False Alarms (Same -> Diff) is prioritized. If we want False Alarm < 5.0%, we should choose t >= 0.30.")

if __name__ == "__main__":
    find_optimal_threshold()
