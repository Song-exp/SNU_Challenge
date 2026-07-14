import pandas as pd
import os
import spacy

# 1. SpaCy 모델 로드 및 없을 시 자동 다운로드
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("en_core_web_sm 모델을 찾을 수 없습니다. 자동으로 다운로드를 시작합니다...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# 분석 대상 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")

def classify_sentence_spacy(sentence):
    """
    SpaCy의 의존 구문 트리(Dependency Parsing Tree)를 기반으로
    문장 사전을 하드코딩하지 않고 100% 일반화하여 3단계로 분류합니다.
    """
    if not isinstance(sentence, str):
        return "Unknown"
    
    doc = nlp(sentence)
    
    # 의존 트리 관계 조사
    has_subordinate_clause = False # advcl (부사절 종속) 유무
    has_parallel_clause = False    # conj (등위절 결합) 유무
    
    for token in doc:
        # 1. advcl (Adverbial Clause) 혹은 ccomp (Clausal Complement) 관계가 있으면 종속절이 존재함
        if token.dep_ in {"advcl", "ccomp"}:
            has_subordinate_clause = True
            
        # 2. conj (Conjunct) 관계이면서 그 대상이 동사(VERB)인 경우 대등절이 존재함
        # (단순 명사 나열을 필터링하기 위해 동사 성격을 띠는 결합만 체크)
        if token.dep_ == "conj" and token.pos_ in {"VERB", "AUX"}:
            has_parallel_clause = True
            
    # 3. 구문 분석 기반 3단계 판정
    if not has_subordinate_clause and not has_parallel_clause:
        # 의존 트리상 동사가 하나이거나 수식 절이 전혀 없는 경우
        return "Type 1: Single-Clause (단일 절)"
    elif has_subordinate_clause:
        # 절과 절이 종속 관계로 수식하고 있는 복합 구조 (사전 하드코딩 불필요)
        return "Type 2: Complex-Subordinate (복합 종속)"
    else:
        # 종속 관계는 없으나, 다수의 사건 동사들이 등위(conj)로 대등하게 엮인 구조
        return "Type 3: Parallel-Coordinated (대등 병렬)"

def analyze_dataset(file_path, name):
    if not os.path.exists(file_path):
        print(f"[{name}] 파일을 찾을 수 없습니다: {file_path}")
        return
    
    df = pd.read_csv(file_path)
    print(f"\n[{name}] 구문 분석 중... 잠시만 기다려주세요.")
    df['Type'] = df['Sentence'].apply(classify_sentence_spacy)
    
    counts = df['Type'].value_counts()
    ratios = df['Type'].value_counts(normalize=True) * 100
    
    print(f"\n===== SpaCy 구문 분석 결과 - {name} ({len(df)}개) =====")
    for idx in counts.index:
        print(f"- {idx}: {counts[idx]}개 ({ratios[idx]:.2f}%)")
        
    print(f"\n--- {name} 유형별 샘플 문장 예시 ---")
    for t in ["Type 1: Single-Clause (단일 절)", "Type 2: Complex-Subordinate (복합 종속)", "Type 3: Parallel-Coordinated (대등 병렬)"]:
        subset = df[df['Type'] == t]['Sentence'].head(3).tolist()
        print(f"[{t}]")
        for s in subset:
            print(f"  * {s}")

if __name__ == "__main__":
    print("SpaCy AI 구문 분석기 기반 3단계 문장 모호성 분류를 시작합니다...")
    analyze_dataset(train_path, "Train")
    analyze_dataset(test_path, "Test")
