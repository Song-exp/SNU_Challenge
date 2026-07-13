import os
import sys
import matplotlib.pyplot as plt

# Force stdout to UTF-8 to prevent cp949 encode errors in Windows
sys.stdout.reconfigure(encoding='utf-8')

# Windows 한글 폰트 설정 (Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

OUTPUT_PLOT = "C:/Users/user/Desktop/서울대/eda/video_production_types.png"
ARTIFACT_PLOT = "C:/Users/user/.gemini/antigravity-cli/brain/8c0c8c15-ad37-4207-b8c5-0210c0ab1b36/video_production_types.png"

# 실측값 데이터 정의
categories = [
    "유형 3: 장면 전환 3회 (매 프레임 장면 전환)\nType 3: 3 Cuts (4 different scenes)",
    "유형 2: 장면 전환 2회 (3개 씬 분할)\nType 2: 2 Cuts (3 scenes)",
    "과도기 영역: 페이드/디졸브/노이즈\nTransition: Noise & Ambiguous Area",
    "유형 1: 장면 전환 1회 (2개 씬 분할)\nType 1: 1 Cut (2 scenes)",
    "유형 0: 장면 전환 0회 (순수 미세 행동)\nType 0: 0 Cuts (Still Action)"
]

counts = [7508, 1361, 315, 209, 142]
percentages = [78.74, 14.27, 3.31, 2.19, 1.49]
colors = ['#E74C3C', '#E67E22', '#BDC3C7', '#3498DB', '#2ECC71']

def main():
    plt.figure(figsize=(10, 6))
    
    # 가로 막대 그래프 생성
    bars = plt.barh(categories, counts, color=colors, height=0.6)
    
    # 막대 끝에 수치 및 백분율 표시
    for bar, pct in zip(bars, percentages):
        width = bar.get_width()
        plt.text(
            width + 80, 
            bar.get_y() + bar.get_height()/2, 
            f'{width:,}개 ({pct:.2f}%)', 
            va='center', 
            ha='left', 
            fontsize=10, 
            fontweight='bold', 
            color='#2C3E50'
        )
        
    plt.title('비디오 연출 유형별 분포 통계 (전체 9,535개 비디오 전수 조사)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('비디오 샘플 수 (개)', fontsize=12)
    plt.xlim(0, 8800) # 가독성을 위해 오른쪽 마진 확장
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    
    plt.gca().invert_yaxis() # 큰 값이 상단에 오도록 y축 뒤집기
    plt.tight_layout()
    
    # 파일 저장
    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=300)
    
    os.makedirs(os.path.dirname(ARTIFACT_PLOT), exist_ok=True)
    plt.savefig(ARTIFACT_PLOT, dpi=300)
    
    print(f"Production types plot saved successfully to:\n- {OUTPUT_PLOT}\n- {ARTIFACT_PLOT}")

if __name__ == "__main__":
    main()
