import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, mannwhitneyu
import sys

# Force stdout to UTF-8 to prevent encoding errors
sys.stdout.reconfigure(encoding='utf-8')

# Windows 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 경로 설정
BASE_DIR = r"C:\Users\bella\Desktop\대학\공모전\트리플에이치\snu_ai_공모전"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
DATASET_MSE_CACHE = os.path.join(BASE_DIR, "eda", "pairwise_mse_cache.csv")
OUTPUT_LABELS_CSV = os.path.join(BASE_DIR, "eda", "sentence_type_labels.csv")
OUTPUT_MISMATCH_CSV = os.path.join(BASE_DIR, "eda", "mismatch_samples.csv")
OUTPUT_REPORT_MD = os.path.join(BASE_DIR, "eda", "sentence_type_purity_report.md")

# 11장 LEXICON 정의
LEXICON = {
    "nouns": r"\b(man|woman|boy|girl|cyclist|skater|gymnast|person|player|dog|cat|camera|chef|performer|diver|skier|rider|swimmer|dancer|bartender|worker|operator)\b",
    "pronouns": r"\b(he|she|it|they|him|her|them|himself|herself|themselves)\b",
    "stative_verbs": r"\b(seen|sitting|standing|walking|looking|watching|holding|carrying|wearing)\b",
    "temporal_connectives": r"\b(then|before|after|followed|finally|next)\b",
    "simultaneous_connectives": r"\b(while|as|meanwhile|during)\b"
}

def classify_sentence(sentence):
    if not isinstance(sentence, str):
        return "Unknown", "RULE_UNKNOWN", False, False
    
    s = sentence.lower()
    
    # 플래그 추출 (Step 3용)
    has_temporal = any(re.search(pat, s) for pat in [
        r"\bthen\b", r"\bbefore\b", r"\bafter\b", r"\bfollowed\b", r"\bfinally\b", r"\bnext\b", 
        r"\btransitioning\b", r"\bshifting\b", r"\bcutting\b"
    ])
    has_simultaneous = any(re.search(pat, s) for pat in [
        r"\bwhile\b", r"\bas\b", r"\bmeanwhile\b", r"\bduring\b"
    ])
    
    # 종속 지시어 패턴
    subordinate_patterns = [
        r"\bthen\b", r"\bbefore\b", r"\bafter\b", r"\bwhile\b", r"\bas\b", r"\bwhen\b",
        r"\bfollowed\s+by\b", r"\btransitioning\s+to\b", r"\bshifting\s+(to|from)\b",
        r"\bcutting\s+to\b", r"\bending\s+with\b"
    ]
    has_subordinate = any(re.search(pat, s) for pat in subordinate_patterns)
    
    comma_count = s.count(',')
    semicolon_count = s.count(';')
    and_count = len(re.findall(r"\band\b", s))
    total_splits = comma_count + semicolon_count + and_count
    
    # 주체 카운팅 (nouns + pronouns)
    actors = re.findall(LEXICON["nouns"], s) + re.findall(LEXICON["pronouns"], s)
    actor_count = len(actors)
    
    # 3유형 분류
    if total_splits == 0 and not has_subordinate:
        return "Type-1", "RULE_NO_PUNCT_NO_CONN", has_temporal, has_simultaneous
    
    if actor_count <= 1 and not has_subordinate:
        return "Type-1", "RULE_SINGLE_ACTOR_NO_CONN", has_temporal, has_simultaneous
        
    if not has_subordinate:
        if comma_count > 0 or semicolon_count > 0 or and_count > 1:
            return "Type-3", "RULE_PARALLEL_LIST", has_temporal, has_simultaneous
        else:
            return "Type-1", "RULE_SINGLE_AND_NO_CONN", has_temporal, has_simultaneous
    else:
        return "Type-2", "RULE_SUBORDINATE_STRUCTURE", has_temporal, has_simultaneous

def cliffs_delta(x, y):
    n1, n2 = len(x), len(y)
    diff = np.subtract.outer(x, y)
    delta = np.sum(np.sign(diff)) / (n1 * n2)
    return delta

def main():
    print("Starting Sentence Ambiguity & Purity Verification Analysis Pipeline...")
    
    # 1. train.csv 로드
    if not os.path.exists(TRAIN_CSV):
        print(f"train.csv not found at {TRAIN_CSV}")
        return
    df = pd.read_csv(TRAIN_CSV)
    
    # 2. Step 0: 문장 3유형 라벨링
    print("\n--- Step 0: Sentence Labeling ---")
    results = []
    for _, row in df.iterrows():
        s_type, rule, temp, sim = classify_sentence(row['Sentence'])
        results.append((row['Id'], row['Sentence'], s_type, temp, sim, rule))
    
    label_df = pd.DataFrame(results, columns=['Id', 'Sentence', 'Type', 'has_temporal', 'has_simultaneous', 'rule_name'])
    os.makedirs(os.path.dirname(OUTPUT_LABELS_CSV), exist_ok=True)
    label_df.to_csv(OUTPUT_LABELS_CSV, index=False, encoding='utf-8-sig')
    print(f"Saved labeled sentences to {OUTPUT_LABELS_CSV}")
    
    # 분포 비율 출력
    counts = label_df['Type'].value_counts()
    ratios = label_df['Type'].value_counts(normalize=True) * 100
    print("Sentence Type Distribution:")
    for idx in counts.index:
        print(f"- {idx}: {counts[idx]} samples ({ratios[idx]:.2f}%)")
        
    # 유형별 랜덤 10개 출력
    print("\nRandom 10 samples per type for quality check:")
    for t in ["Type-1", "Type-2", "Type-3"]:
        subset = label_df[label_df['Type'] == t]['Sentence'].sample(n=min(10, len(label_df[label_df['Type'] == t])), random_state=42).tolist()
        print(f"[{t}]")
        for i, s in enumerate(subset):
            print(f"  {i+1}. {s}")
            
    # 3. pairwise MSE 결과 로드
    if not os.path.exists(DATASET_MSE_CACHE):
        print(f"Waiting for {DATASET_MSE_CACHE} to be generated...")
        return
        
    mse_df = pd.read_csv(DATASET_MSE_CACHE)
    
    # 데이터 조인
    merged_df = pd.merge(label_df, mse_df, on='Id')
    
    # 4. Step 1: 비디오유형 분류 및 교차표/Lift 산출
    print("\n--- Step 1: Purity & Cross-tabulation Analysis ---")
    # 비디오유형 구간화
    def get_video_type(sim_pairs):
        if sim_pairs == 6:
            return "단일씬 (6쌍)"
        elif sim_pairs == 3:
            return "씬 2개 (3쌍)"
        elif sim_pairs == 1:
            return "씬 3개 (1쌍)"
        elif sim_pairs == 0:
            return "완전전환 (0쌍)"
        else:
            return "과도기 (나머지)"
            
    merged_df['Video_Type'] = merged_df['sim_pairs'].apply(get_video_type)
    
    # 기저 분포 계산 (Marginal)
    base_dist = merged_df['Video_Type'].value_counts(normalize=True)
    
    # 교차표 (Raw Counts)
    crosstab_raw = pd.crosstab(merged_df['Video_Type'], merged_df['Type'])
    # 조건부 분포 P(비디오유형 | 문장유형) -> 열 정규화
    crosstab_pct = pd.crosstab(merged_df['Video_Type'], merged_df['Type'], normalize='columns') * 100
    
    # Lift 계산
    crosstab_lift = pd.crosstab(merged_df['Video_Type'], merged_df['Type'], normalize='columns')
    for col in crosstab_lift.columns:
        crosstab_lift[col] = crosstab_lift[col] / base_dist
        
    print("\nConditional Distribution P(Video_Type | Sentence_Type) (%):")
    print(crosstab_pct)
    
    print("\nLift Matrix (Relative to Marginal):")
    print(crosstab_lift)
    
    # 카이제곱 검정
    chi2, p_val, dof, expected = chi2_contingency(crosstab_raw)
    n = len(merged_df)
    min_dim = min(crosstab_raw.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    print(f"\nChi-square test: chi2 = {chi2:.4f}, p-value = {p_val:.4g}")
    print(f"Cramer's V (Association Strength): {cramers_v:.4f}")
    
    # 사전 판정 기준 검사
    # 저전환 비디오 (유사쌍 3개 이상): sim_pairs >= 3
    merged_df['Low_Transition'] = merged_df['sim_pairs'] >= 3
    low_trans_crosstab = pd.crosstab(merged_df['Low_Transition'], merged_df['Type'], normalize='columns')
    low_trans_base = (merged_df['sim_pairs'] >= 3).mean()
    type1_low_trans_lift = low_trans_crosstab.loc[True, 'Type-1'] / low_trans_base
    
    # Type-3 완전전환형(0쌍) lift
    type3_full_trans_lift = crosstab_lift.loc['완전전환 (0쌍)', 'Type-3']
    
    decision_1 = (type1_low_trans_lift >= 3.0) and (type3_full_trans_lift >= 1.0)
    decision_2 = all(abs(crosstab_lift.loc[vt, st] - 1.0) < 0.15 for vt in crosstab_lift.index for st in crosstab_lift.columns)
    
    final_decision = ""
    decision_reason = ""
    if decision_1:
        final_decision = "1축(텍스트) 라우팅 가능"
        decision_reason = f"Type-1에서 저전환 비디오의 Lift가 {type1_low_trans_lift:.2f} (>= 3.0) 이며, Type-3에서 완전전환형 Lift가 {type3_full_trans_lift:.2f} (>= 1.0)이므로 텍스트 문법 구조가 실제 비디오 전환 레이아웃의 유의미한 상한/하한을 규정합니다."
    elif decision_2:
        final_decision = "2축(텍스트x이미지) 필요"
        decision_reason = "모든 Lift가 1.0 부근에 수렴하여 텍스트 유형이 실제 이미지 씬 구조와 완전히 독립적이며 상호 시너지가 없습니다."
    else:
        final_decision = "유형 재정의 필요"
        decision_reason = f"텍스트 3유형과 비디오 장면 구조 간에 부분적인 상관관계는 관측되나(Type-1 저전환 Lift: {type1_low_trans_lift:.2f}, Type-3 완전전환 Lift: {type3_full_trans_lift:.2f}), 라우팅 분기용 독자적인 축으로 사용하기에는 노이즈와 예외가 많습니다."
        
    print(f"\n[Final Decision] {final_decision}")
    print(f"Reason: {decision_reason}")
    
    # 5. Step 2: 유형별 MSE 분포 비교
    print("\n--- Step 2: MSE Distribution & Hypothesis Testing ---")
    type_groups = {t: merged_df[merged_df['Type'] == t]['median_mse'].values for t in ["Type-1", "Type-2", "Type-3"]}
    
    # Mann-Whitney U 검정
    pairs_to_test = [("Type-1", "Type-3"), ("Type-1", "Type-2"), ("Type-2", "Type-3")]
    mwu_results = {}
    for t1, t2 in pairs_to_test:
        u_stat, p_val_mwu = mannwhitneyu(type_groups[t1], type_groups[t2], alternative='two-sided')
        delta = cliffs_delta(type_groups[t1], type_groups[t2])
        mwu_results[f"{t1} vs {t2}"] = (p_val_mwu, delta)
        print(f"{t1} vs {t2}: MWU p-value = {p_val_mwu:.4g}, Cliff's delta = {delta:.4f}")
        
    # 박스플롯 + 히스토그램 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.boxplot([type_groups[t] for t in ["Type-1", "Type-2", "Type-3"]], tick_labels=["Type-1", "Type-2", "Type-3"])
    plt.title("문장 유형별 MSE 중앙값 박스플롯")
    plt.ylabel("MSE 중앙값 (Median MSE)")
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.subplot(1, 2, 2)
    for t, color in zip(["Type-1", "Type-2", "Type-3"], ['g', 'b', 'r']):
        plt.hist(type_groups[t], bins=30, alpha=0.4, label=t, color=color, density=True)
    plt.title("문장 유형별 MSE 분포 히스토그램")
    plt.xlabel("MSE 중앙값")
    plt.legend()
    plt.grid(linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, "eda", "sentence_type_mse_distribution.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved distribution plot to {plot_path}")
    
    # 6. Step 3: 다중 절 내부 서브 분석 (접속사 부가가치 판정)
    print("\n--- Step 3: Sub-clause Junction Analysis ---")
    multi_clause_df = merged_df[merged_df['Type'].isin(["Type-2", "Type-3"])].copy()
    
    # 동시 연결군 vs 순차 연결군 정의
    # 동시: has_simultaneous=True and has_temporal=False
    # 순차: has_temporal=True and has_simultaneous=False
    multi_clause_df['Junction_Group'] = 'Other'
    multi_clause_df.loc[multi_clause_df['has_simultaneous'] & ~multi_clause_df['has_temporal'], 'Junction_Group'] = 'Simultaneous (동시)'
    multi_clause_df.loc[multi_clause_df['has_temporal'] & ~multi_clause_df['has_simultaneous'], 'Junction_Group'] = 'Sequential (순차)'
    
    junction_counts = multi_clause_df['Junction_Group'].value_counts()
    print(f"Junction Group sample size:\n{junction_counts}")
    
    sim_mse = multi_clause_df[multi_clause_df['Junction_Group'] == 'Simultaneous (동시)']['median_mse'].values
    seq_mse = multi_clause_df[multi_clause_df['Junction_Group'] == 'Sequential (순차)']['median_mse'].values
    
    p_val_junction = None
    delta_junction = None
    step3_decision = ""
    if len(sim_mse) > 0 and len(seq_mse) > 0:
        u_stat_j, p_val_j = mannwhitneyu(sim_mse, seq_mse, alternative='less') # 단측 검정: 동시군이 순차군보다 MSE가 작을 것이다
        p_val_junction = p_val_j
        delta_junction = cliffs_delta(sim_mse, seq_mse)
        print(f"Simultaneous vs Sequential: Mann-Whitney U test (one-sided 'less') p-value = {p_val_j:.4g}, Cliff's delta = {delta_junction:.4f}")
        
        # 유의성 판정 (p < 0.05 이고 Cliff's delta < -0.1)
        if p_val_j < 0.05 and delta_junction < -0.1:
            step3_decision = "동시 연결 하위군의 MSE가 순차 하위군보다 유의하게 낮으므로, 3단계 접속사 분석은 유형 분류 체계에 필수적인 부가가치를 제공합니다. 유형 재정의가 강력히 필요합니다."
        else:
            step3_decision = "두 하위군 간의 MSE 분포 차이가 통계적으로 유의미하지 않거나 효과 크기가 미미하여, 3단계 접속사 분석을 굳이 세부 분류에 추가할 실전적인 필요성이 떨어집니다. 현재의 3유형 분류를 유지하는 근거가 됩니다."
    else:
        step3_decision = "비교를 위한 최소 샘플 수가 부족합니다."
        
    print(f"[Step 3 Decision] {step3_decision}")
    
    # Step 3 시각화
    if len(sim_mse) > 0 and len(seq_mse) > 0:
        plt.figure(figsize=(6, 5))
        plt.boxplot([sim_mse, seq_mse], tick_labels=["Simultaneous (동시)", "Sequential (순차)"])
        plt.title("접속사 하위군별 MSE 중앙값 비교")
        plt.ylabel("MSE 중앙값")
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plot_path_j = os.path.join(BASE_DIR, "eda", "simultaneous_vs_temporal_distribution.png")
        plt.savefig(plot_path_j, dpi=300)
        plt.close()
        print(f"Saved junction plot to {plot_path_j}")
        
    # 7. Step 4: 불일치 셀 샘플링
    print("\n--- Step 4: Mismatch Sampling ---")
    # (A) Type-1 단일 절 × 완전전환형(sim_pairs == 0)
    mismatch_a = merged_df[(merged_df['Type'] == 'Type-1') & (merged_df['sim_pairs'] == 0)]
    # (B) Type-3 병렬 × 저전환(sim_pairs >= 3)
    mismatch_b = merged_df[(merged_df['Type'] == 'Type-3') & (merged_df['sim_pairs'] >= 3)]
    # (C) 동시 연결 플래그 × 완전전환형(sim_pairs == 0)
    mismatch_c = merged_df[(merged_df['has_simultaneous'] == True) & (merged_df['sim_pairs'] == 0)]
    
    sampled_a = mismatch_a.sample(n=min(30, len(mismatch_a)), random_state=42)
    sampled_b = mismatch_b.sample(n=min(30, len(mismatch_b)), random_state=42)
    sampled_c = mismatch_c.sample(n=min(30, len(mismatch_c)), random_state=42)
    
    # 이미지 파일 경로 4개 구성
    def build_img_paths(row):
        sample_id = str(row['Id'])
        ans = ast.literal_eval(row['Answer'])
        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        ordered_files = [None] * 4
        for idx, pos in enumerate(ans):
            ordered_files[pos - 1] = shuffled_files[idx]
        
        # data_train/SampleId/FileName 절대경로 리턴
        paths = [f"file:///C:/Users/bella/Desktop/대학/공모전/트리플에이치/snu_ai_공모전/data_train/{sample_id}/{f}" for f in ordered_files]
        return paths
        
    mismatch_list = []
    for grp_name, grp_df in [("A", sampled_a), ("B", sampled_b), ("C", sampled_c)]:
        for _, row in grp_df.iterrows():
            img_paths = build_img_paths(row)
            mismatch_list.append((
                row['Id'], row['Sentence'], row['Type'], row['has_temporal'], row['has_simultaneous'],
                row['sim_pairs'], row['median_mse'], img_paths[0], img_paths[1], img_paths[2], img_paths[3], grp_name
            ))
            
    mismatch_df = pd.DataFrame(mismatch_list, columns=[
        'Id', 'Sentence', 'Type', 'has_temporal', 'has_simultaneous',
        'sim_pairs', 'median_mse', 'img_path_1', 'img_path_2', 'img_path_3', 'img_path_4', 'mismatch_group'
    ])
    mismatch_df.to_csv(OUTPUT_MISMATCH_CSV, index=False, encoding='utf-8-sig')
    print(f"Saved mismatch samples to {OUTPUT_MISMATCH_CSV} (Total samples: {len(mismatch_df)})")
    
    # 8. 리포트 마크다운 파일 작성 (sentence_type_purity_report.md)
    print("\nGenerating final Markdown report...")
    
    report_content = f"""# 문장 3유형 × 이미지 씬 구조 순도(Purity) 검증 분석 보고서

본 보고서는 SNU AI Challenge 비디오 프레임 순서 예측 경진대회에서 제안된 **"문장 3유형 분류 체계"**가 실제 이미지의 물리적 씬(Scene) 구조와 얼마나 순수하게 정합되는지 정량적으로 검증하고, 이를 모델링 아키텍처의 분기 축으로 사용할 수 있는지 판정한 연구 보고서입니다.

---

## 1. Step 0: 문장 3유형 라벨링 분포 결과
구문 분석 규칙(경량 Regex 기반 절 개수 카운팅)을 전체 학습 데이터셋(9,535개 샘플)에 적용한 결과입니다.

* **Type-1 (단일 절 - Single Clause)**: {counts.get('Type-1', 0):,}개 ({ratios.get('Type-1', 0.0):.2f}%)
* **Type-2 (복합 종속 - Complex Subordinate)**: {counts.get('Type-2', 0):,}개 ({ratios.get('Type-2', 0.0):.2f}%)
* **Type-3 (대등 병렬 - Parallel Coordinated)**: {counts.get('Type-3', 0):,}개 ({ratios.get('Type-3', 0.0):.2f}%)

---

## 2. Step 1: 순도 검증 및 교차 연관성 검정

### 2.1 비디오 연출 유형별 조건부 확률 분포 $P(\\text{{Video\\_Type}} | \\text{{Sentence\\_Type}})$ (%)
문장 유형이 주어졌을 때, 비디오가 실제로 어떤 장면 전환 레이아웃을 가지는지에 대한 비율 테이블입니다.

{crosstab_pct.to_markdown()}

### 2.2 기저 대비 리프트 (Lift) 행렬
전체 비디오 유형의 기저 비율을 기준으로 나눈 Lift 값입니다. (1.0보다 크면 해당 문장 유형이 그 비디오 유형을 강하게 유도함을 의미합니다.)

{crosstab_lift.to_markdown()}

### 2.3 연관성 통계 검정 결과
* **카이제곱 독립성 검정 ($p$-value)**: {p_val:.4g} (유의수준 0.05 기준)
* **Cramér's V (연관성 크기)**: {cramers_v:.4f}
* **판정 해석**: {decision_reason}

---

## 3. Step 2: 유형별 MSE 분포 및 가설 검정
각 문장 유형 그룹의 6쌍 MSE 중앙값(Median MSE) 분포에 대한 비모수 검정 결과입니다.

* **Type-1 vs Type-3**: MWU $p$-value = {mwu_results['Type-1 vs Type-3'][0]:.4g}, Cliff's delta = {mwu_results['Type-1 vs Type-3'][1]:.4f}
* **Type-1 vs Type-2**: MWU $p$-value = {mwu_results['Type-1 vs Type-2'][0]:.4g}, Cliff's delta = {mwu_results['Type-1 vs Type-2'][1]:.4f}
* **Type-2 vs Type-3**: MWU $p$-value = {mwu_results['Type-2 vs Type-3'][0]:.4g}, Cliff's delta = {mwu_results['Type-2 vs Type-3'][1]:.4f}

![MSE Distribution Chart](./sentence_type_mse_distribution.png)

---

## 4. Step 3: 다중 절 내부 서브 분석 (접속사 부가가치 판정)
* **비교 대상**: 동시 연결군(while/as) vs 순차 연결군(then/before 등)
* **통계 검정 (MWU 단측 검정)**: $p$-value = {p_val_junction:.4g if p_val_junction is not None else 'N/A'}, Cliff's delta = {delta_junction:.4f if delta_junction is not None else 'N/A'}
* **최종 판정**: {step3_decision}

{"![Junction Distribution Chart](./simultaneous_vs_temporal_distribution.png)" if len(sim_mse) > 0 and len(seq_mse) > 0 else ""}

---

## 5. Step 4: 불일치(Mismatch) 엣지 케이스 분석 샘플링
정답 레이블과 텍스트 구문 구조가 상충되는 샘플들을 추출했습니다. 구체적인 질적 리뷰(Qualitative Review)를 위한 상세 리스트는 [mismatch_samples.csv](file:///C:/Users/bella/Desktop/대학/공모전/트리플에이치/snu_ai_공모전/eda/mismatch_samples.csv) 파일을 참고하십시오.

* **(A) Type-1 (단일 절) × 완전전환형(0쌍 유사)**: 단일 묘사 문장이지만 물리적 장면 컷이 마구 전환되는 예외 케이스 (총 {len(mismatch_a)}개 중 30개 샘플링)
* **(B) Type-3 (대등 병렬) × 저전환(3쌍 이상 유사)**: 행위가 순차적으로 나열되나 실제로는 한 공간 안에서 미세한 앵글 변화만 있는 케이스 (총 {len(mismatch_b)}개 중 30개 샘플링)
* **(C) 동시성 연결(while/as) × 완전전환형**: "동시에 동작함"을 뜻하나, 비디오 연출상 장면이 아예 바뀌어 있는 케이스 (총 {len(mismatch_c)}개 중 30개 샘플링)

---

## 6. 최종 결론 (1줄 판정)

> [!IMPORTANT]
> **최종 판정: [{final_decision}]**
> - **근거**: {decision_reason}
"""

    with open(OUTPUT_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Final markdown report generated successfully at {OUTPUT_REPORT_MD}")

if __name__ == "__main__":
    main()
