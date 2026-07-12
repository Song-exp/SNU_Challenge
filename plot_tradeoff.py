import os
import matplotlib.pyplot as plt

# =========================================================================
# [실험 결과 입력] 팀원에게 공유받은 데이터를 여기에 적어주세요.
# =========================================================================
resolutions = [224, 336, 448, 560] # 장축 최대 크기 (target_dim)

# 예시/더미 데이터입니다. 실제 실험 결과가 나오면 이 값을 수정하세요!
speed_sec_per_sample = [0.42, 0.85, 1.49, 2.31] # 1개 샘플당 소요 시간 (초)
accuracy_em = [11.5, 15.0, 17.0, 17.5] # 검증셋 Exact Match 정확도 (%)
vram_usage_gb = [2.1, 3.4, 4.27, 5.8] # VRAM 사용량 (GB)

print("--- SNU AI Challenge - Resolution Tradeoff Plotter ---")
for r, s, a, v in zip(resolutions, speed_sec_per_sample, accuracy_em, vram_usage_gb):
    print(f"Target Dim: {r}px | Speed: {s:.3f} s/sample | Accuracy: {a:.2f}% | VRAM: {v:.2f} GB")

# =========================================================================
# 시각화 그래프 그리기
# =========================================================================
fig, ax1 = plt.subplots(figsize=(10, 6))

# 왼쪽 축: 추론 속도 (초)
color = '#E74C3C'
ax1.set_xlabel('Resolution (Max Edge Dimension in px)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Inference Time (seconds/sample)', color=color, fontweight='bold', fontsize=12)
line1 = ax1.plot(resolutions, speed_sec_per_sample, marker='o', color=color, linewidth=2.5, label='Inference Time (s)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(resolutions)
ax1.grid(True, linestyle='--', alpha=0.5)

# 오른쪽 축: 정확도 (%)
ax2 = ax1.twinx()  
color = '#2E80B8'
ax2.set_ylabel('Exact Match Accuracy (%)', color=color, fontweight='bold', fontsize=12)
line2 = ax2.plot(resolutions, accuracy_em, marker='s', color=color, linewidth=2.5, linestyle='--', label='EM Accuracy (%)')
ax2.tick_params(axis='y', labelcolor=color)

# 축 레이블 합치기
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

# 데이터 라벨링 (어노테이션)
for r, s, a, v in zip(resolutions, speed_sec_per_sample, accuracy_em, vram_usage_gb):
    ax1.annotate(f"{s:.2f}s\n({v:.1f}G)", (r, s), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='#C0392B', weight='bold')
    ax2.annotate(f"{a:.1f}%", (r, a), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='#1F618D', weight='bold')

plt.title("Resolution vs. Inference Speed & Accuracy Trade-off", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()

save_path = 'C:/Users/user/Desktop/서울대/resolution_tradeoff_curve.png'
plt.savefig(save_path, dpi=200)
print(f"\nTrade-off curve plot saved successfully to {save_path}")
