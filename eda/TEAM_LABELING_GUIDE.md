# 👥 SNU AI Challenge - 공동 데이터 검수 및 모호성 라벨링 가이드

이 문서는 비디오 프레임 순서 예측 대회에서 이미지 파트와 텍스트 파트가 협업하여 9,535개 학습 데이터셋의 **문장 모호성(Ambiguity) 및 비디오 연출 유형**을 수작업으로 유형화하기 위한 공동 가이드라인입니다.

---

## 📌 1. 팀원별 담당 영역 지정 (ID 대소문자 무구분 정렬 기준)

컴퓨터와 윈도우 탐색기가 폴더 이름을 정렬하는 표준 대소문자 무구분(Case-insensitive) 정렬 방식을 기준으로, 총 9,535개 샘플을 3개의 구역으로 공평하게 쪼갰습니다. **라벨러 프로그램을 실행하고 본인의 이름을 클릭하면 해당 구역만 자동으로 필터링 및 로드됩니다.**

| 담당자 | 담당 영역 (ID 시작 ~ 끝) | 대상 샘플 수 | 최종 저장되는 파일명 |
| :--- | :--- | :---: | :--- |
| **병철 (Byeong-cheol)** | `00GGp0` ~ `EquxBk` | 3,000개 | `eda/labeled_byeongcheol.csv` |
| **서현 (Seo-hyeon)** | `er2p3e` ~ `oQI68U` | 3,003개 | `eda/labeled_seohyeon.csv` |
| **정현 (Jeong-hyeon)** | `oqImQK` ~ `ZzYxAm` (끝) | 3,532개 | `eda/labeled_jeonghyeon.csv` |

---

## 🛠️ 2. 환경 셋팅 및 실행 방법 (Teammates Setup)

팀원들의 로컬 컴퓨터에서 라벨러 창을 띄우는 순서입니다.

### ① 필수 패키지 설치
라벨러 실행을 위해 파이썬 가상환경 또는 터미널에서 아래 라이브러리를 설치해 줍니다. (Pillow는 이미지 출력용, pandas는 데이터 관리용입니다.)
```bash
pip install pillow pandas
```

### ② 프로그램 실행
프로젝트 루트 디렉토리(`SNU_Challenge/`)에서 터미널을 열고 아래 명령어로 스크립트를 실행합니다.
```bash
python eda/team_image_labeler.py
```
* 실행 후 나타나는 시작 화면에서 본인의 이름을 클릭하면 검수 화면으로 진입합니다.

---

## ✍️ 3. 문장 모호성 5단계 심층 분류 기준 (Taxonomy Guide)

문장을 읽고, 이미지 배열 힌트가 텍스트에 얼마나 녹아있는지 판단하기 위한 **5단계 분류 기준**입니다. 
(※ 분석을 돕기 위해 프로그램 실행 시 입력창에 **AI가 1차 예측한 문법 카테고리와 핵심 동사(Main verb) 정보가 자동으로 입력되어 나타납니다.**)

| 단계 | 카테고리명 | 모호성 수준 | 판단 기준 (동작 및 문법 구조) | 직관적 예시 |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Explicit-Sequential**<br>(명시적 시계열) | **매우 낮음** | 절과 절 사이에 명확한 시간 선후 관계 접속사가 존재함 (`then`, `before`, `after`, `followed by`, `finally`) | A가 ~하고, 그 다음에(`then`) B가 ~한다. |
| **2** | **Implicit-Sequential**<br>(묵시적 시계열) | **낮음** | 시간 접속사는 없으나, 여러 사건이 콤마(`,`)나 `and`로 대등하게 나열되어 순서 관계를 암시함 | A가 ~하고, B가 ~하고, C가 ~한다. |
| **3** | **Implicit-Trajectory**<br>(묵시적 궤적) | **중간** | 단일 동작이지만, 물리적/공간적 이동 전치사(`down`, `up`, `towards`, `into`, `across`)가 궤적의 방향성을 제시함 | 소년이 스케이트보드를 타고 길 아래로(`down`) 내려간다. |
| **4** | **Single-Action**<br>(단일 동작) | **높음** | 다른 시공간 수식어 없이 단일 주어의 단 하나의 동작만 기술함 (포즈 변화 등 디테일 추론 필수) | 여성이 수건으로 얼굴을 닦는다(`wipe`). |
| **5** | **Static-State**<br>(상태/지속) | **매우 높음** | 모든 행동이 동시에 지속되거나(`while`, `as`) 정적인 상태(sitting, holding 등)만 묘사되어 순서 개념이 소실됨 | 남자가 제트스키에 끌려 수상스키를 타는 중이다(`while`). |

---

## 🎮 4. 실전 검수 조작 및 가이드라인

### 🎹 키보드 단축키
마우스 조작을 최소화하여 손목 피로를 줄이고 속도를 높이는 단축키입니다.
* **`Enter (엔터 키)`**: 입력창에 작성된 텍스트를 저장하고 승인 마킹(`Confirmed: True`) 처리한 뒤 **자동으로 다음 문항으로 점프**합니다.
* **`Alt + Left Arrow (←)`**: 이전 문항으로 되돌아갑니다. (오타나 잘못 넘긴 부분 수정)
* **`Alt + Right Arrow (→)`**: 현재 문항을 마킹하지 않고 그냥 넘어갑니다 (Skip).

### 💡 실전 추천 검수 워크플로우
1. 화면에 표시된 이미지 4장과 문장을 확인합니다.
2. 입력창에 AI가 미리 분석해 둔 가이드 텍스트를 눈으로 봅니다.
   * 예: `Category: 4, Main verb: wipe, Background state: No, Anomalies: None`
3. **만약 AI 가이드 분석이 맞다면**: 그냥 바로 **`Enter`**를 쳐서 다음 문제로 넘어갑니다. (가장 빠른 방법)
4. **만약 오분류가 있거나 추가 마킹이 필요한 경우**:
   * 입력창의 내용을 지우거나 뒤에 쉼표(`,`)를 찍고 메모를 추가해 줍니다.
   * **자주 쓰는 권장 태그들**:
     * `Category: 1~5` 단계 변경 기입
     * `이상신호: 오탈자`
     * `이상신호: 문장이 이미지랑 안 맞음` (그림 상 순서와 문장 서술 순서가 매칭 불가능한 경우)
     * `이상신호: 판단불가`
   * 수정한 뒤 **`Enter`**를 누릅니다.

---

## 💾 5. 최종 데이터 취합 (Merge) 방법

세 명이 작업을 모두 마친 후 생성된 3개의 csv 파일(`labeled_byeongcheol.csv`, `labeled_seohyeon.csv`, `labeled_jeonghyeon.csv`)을 하나의 파일(`train_labeled_all.csv`)로 합치기 위한 방법입니다.

프로젝트 폴더 내에서 파이썬을 열고 아래 스크립트를 한 번 실행하면 간단하게 병합이 완료됩니다.

```python
import pandas as pd

# 세 명의 결과 파일 불러오기
df1 = pd.read_csv("./eda/labeled_byeongcheol.csv")
df2 = pd.read_csv("./eda/labeled_seohyeon.csv")
df3 = pd.read_csv("./eda/labeled_jeonghyeon.csv")

# 수직 통합 (Concat)
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# 원본 Id 순서(대소문자 무구분)대로 재정렬
combined_df = combined_df.iloc[combined_df['Id'].str.lower().argsort()].reset_index(drop=True)

# 최종 합본 저장
combined_df.to_csv("./eda/train_labeled_all.csv", index=False, encoding="utf-8-sig")
print("🎉 모든 팀원의 데이터가 성공적으로 병합되었습니다! 생성 파일: eda/train_labeled_all.csv")
```
