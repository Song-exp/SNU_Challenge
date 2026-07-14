import pandas as pd
import os
import re

# 분석 대상 경로 설정 (사용자 로컬 프로젝트 경로)
base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")

def classify_sentence(sentence):
    """
    문장의 통사적 구조(Syntax)를 기반으로 3가지 유형 중 하나로 분류합니다.
    """
    if not isinstance(sentence, str):
        return "Unknown"
    
    # 분석을 위한 소문자 변환
    s = sentence.lower()
    
    # 1. 종속 접속사 및 지시어 패턴 (유형 2 판단용)
    subordinate_patterns = [
        r"\bthen\b", r"\bbefore\b", r"\bafter\b", r"\bwhile\b", r"\bas\b", r"\bwhen\b",
        r"\bfollowed\s+by\b", r"\btransitioning\s+to\b", r"\bshifting\s+(to|from)\b",
        r"\bcutting\s+to\b", r"\bgradually\b"
    ]
    
    has_subordinate = any(re.search(pat, s) for pat in subordinate_patterns)
    
    # 2. 문장 분할 및 등위 부호 카운트
    comma_count = s.count(',')
    semicolon_count = s.count(';')
    and_count = len(re.findall(r"\band\b", s))
    
    total_splits = comma_count + semicolon_count + and_count
    
    # 3. 3대 유형 최종 분류 로직
    # A. 콤마, 세미콜론, and, 종속접속사가 거의 없는 극단적 단일절 구조
    if total_splits == 0 and not has_subordinate:
        return "Type 1: Single-Clause"
    
    if not has_subordinate:
        # B. 종속접속사가 없지만 and만 1개 있는 가벼운 구조는 단일 절로 판정
        if comma_count == 0 and semicolon_count == 0 and and_count <= 1:
            return "Type 1: Single-Clause"
        else:
            # C. 종속접속사는 없지만 콤마나 and가 다수 얽힌 대등 병렬 나열 구조
            return "Type 3: Parallel-Coordinated"
    else:
        # D. 종속접속사가 존재하여 시간적 인과/동시성이 엮인 구조
        return "Type 2: Complex-Subordinate"

def analyze_dataset(file_path, name):
    if not os.path.exists(file_path):
        print(f"[{name}] 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    df['Type'] = df['Sentence'].apply(classify_sentence)
    
    counts = df['Type'].value_counts()
    ratios = df['Type'].value_counts(normalize=True) * 100
    
    print(f"\n===== {name} 데이터셋 분석 결과 (총 {len(df)}개) =====")
    for idx in counts.index:
        print(f"- {idx}: {counts[idx]}개 ({ratios[idx]:.2f}%)")
        
    print(f"\n--- {name} 유형별 샘플 문장 예시 ---")
    for t in ["Type 1: Single-Clause", "Type 2: Complex-Subordinate", "Type 3: Parallel-Coordinated"]:
        subset = df[df['Type'] == t]['Sentence'].head(3).tolist()
        print(f"[{t}]")
        for s in subset:
            print(f"  * {s}")

if __name__ == "__main__":
    print("3단계 문장 모호성 분류 분석을 시작합니다...")
    analyze_dataset(train_path, "Train")
    analyze_dataset(test_path, "Test")
