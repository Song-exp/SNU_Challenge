import os
import pandas as pd
import sys

# 프로젝트 루트 경로를 Python 경로에 추가
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.features.flag_detector import OrthogonalFlagDetector

def extract_features(input_path, output_path, name):
    if not os.path.exists(input_path):
        print(f"[{name}] 파일을 찾을 수 없습니다: {input_path}")
        return
    
    print(f"\n===== {name} 데이터셋 피처 추출 시작 =====")
    df = pd.read_csv(input_path)
    
    detector = OrthogonalFlagDetector()
    
    results = []
    total = len(df)
    
    for i, row in df.iterrows():
        sentence = row['Sentence']
        features = detector.process_sentence(sentence)
        # 기존 데이터프레임의 모든 열 병합
        row_dict = row.to_dict()
        row_dict.update(features)
        results.append(row_dict)
        
        if (i + 1) % 1000 == 0 or (i + 1) == total:
            print(f"진행 상황: {i + 1}/{total} 완료")
            
    df_featured = pd.DataFrame(results)
    df_featured.to_csv(output_path, index=False)
    print(f"[{name}] 피처가 추가된 데이터셋 저장 완료: {output_path}")
    
    # 간단한 분석 통계 출력
    print(f"\n--- {name} 1차 파티션 분포 ---")
    p_counts = df_featured['Partition'].value_counts()
    p_ratios = df_featured['Partition'].value_counts(normalize=True) * 100
    for idx in p_counts.index:
        print(f"  * {idx}: {p_counts[idx]}개 ({p_ratios[idx]:.2f}%)")
        
    print(f"\n--- {name} 직교 플래그 활성화 분포 ---")
    flag_cols = [col for col in df_featured.columns if col.startswith("N")]
    for col in flag_cols:
        count = df_featured[col].sum()
        ratio = (count / total) * 100
        print(f"  * {col}: {count}개 ({ratio:.2f}%)")

if __name__ == "__main__":
    train_in = os.path.join(base_dir, "train.csv")
    train_out = os.path.join(base_dir, "train_with_flags.csv")
    test_in = os.path.join(base_dir, "test.csv")
    test_out = os.path.join(base_dir, "test_with_flags.csv")
    
    extract_features(train_in, train_out, "Train")
    extract_features(test_in, test_out, "Test")
