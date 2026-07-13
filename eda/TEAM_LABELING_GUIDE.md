# 👥 SNU AI Challenge - 공동 데이터 검수 및 모호성 라벨링 가이드

> 텍스트 담당 팀원의 EDA 결과(5단계 심층 분류 + 모호성 지수 AI)를 요약하고,
> 수작업 라벨링 시 참고할 수 있도록 기준표로 재구성한 문서입니다.

---

## 1. 팀원별 담당 영역 지정 (ID 대소문자 무구분 정렬 기준)

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

## 🔍 3. 판별 순서 (팀원이 설계한 흐름)

문장을 볼 때 아래 순서로 판단하면 카테고리가 정해집니다.

1. **주어-서술어 절 개수부터 센다** (단일 절 vs 다중 절)
   * 어순이 아니라 **"주체+서술어 쌍의 개수"**로 끊어 읽습니다. 도치문(도치 구조)에도 안전한 분류 기준이 됩니다.
2. **단일 절이면** ➡️ 서술어가 **지속성(상태)**인지 **경계성(동작 완료)**인지로 갈립니다.
3. **다중 절이면** ➡️ 절들이 **주절+종속 부사구** 구조인지, **등위/병렬 나열**인지로 갈리고, 접속사 종류(순차 vs 동시)로 최종 카테고리가 정해집니다.

---

## 📊 4. 5단계 분류 기준표

| # | 카테고리 | 모호성 | 핵심 판별 신호 | 통사 구조 | 예시 | Train 비율 | Test 비율 |
|---|---|---|---|---|---|---|---|
| **1** | **Explicit-Sequential**<br>(명시적 시계열) | 매우 낮음 | `then`, `before`, `after`, `followed by`, `finally` 등 순차 접속사가 주성분 사이를 연결 | `[주체A+동작1] + 시간접속사 + [주체B+동작2]` | "she lowers her gaze, **then** a towel is raised, **followed by** a zoom-in" | 69.40% (6,617) | 85.71% (702) |
| **2** | **Implicit-Sequential**<br>(묵시적 시계열) | 낮음 | 순차 접속사는 없지만 콤마/`and`로 여러 사건이 병렬 나열됨 ➡️ 나열 순서 자체가 시간 힌트 | `[주체A+동작1], [주체B+동작2], and [주체C+동작3]` | "the fighter retreats, the opponent advances, the player in yellow moves right" | 8.67% (827) | 5.62% (46) |
| **3** | **Implicit-Trajectory**<br>(묵시적 궤적) | 중간 | 단일 동작이지만 방향/궤적 전치사(`down`,`up`,`towards`,`into`,`across`)가 공간 이동을 서술 | `[주체] + [동작] + [궤적 전치사구]` | "a boy rides **down** a street on a skateboard" | 17.30% (1,650) | 8.67% (71) |
| **4** | **Single-Action**<br>(단일 동작) | 높음 | 단일 주체+단일 경계성(Telic) 동사만 있고, 수식하는 시간/공간 단서가 전혀 없음 | `[단일 주체] + [단일 동사(Telic)]` | "the gymnast mounts the beam" | 4.43% (422) | 0.00% (0) |
| **5** | **Static-State**<br>(상태/지속) | 매우 높음<br>(최대) | 지속성(Stative) 동사(sitting, holding 등)이거나 `while`/`as` 동시접속사로 여러 배경 상태가 동시에 묘사됨 | `[주체]+[지속동사]` 또는 `[핵심사건]+while/as+[동시 배경들]` | "the man is shown waterskiing **while** pulled by the jetski" | 0.20% (19) | 0.00% (0) |

> 💡 **참고**: 각 카테고리별 `No_ordering=True` 비율은 14.5~16.5%로 거의 균일합니다. 즉, 문장의 모호성 수준과 셔플 안 됨(순서 일치) 여부는 서로 무관합니다.

---

## 💡 5. 핵심 구분 개념: 주성분 vs 부속성분

| 구분 | 정의 | 동사 성격 | 라벨링 판단에 미치는 영향 |
|---|---|---|---|
| **주성분 (Foreground Event)** | 문장의 진짜 핵심 행동, 상태 전이를 유발 | 경계성(Telic/Achievement) — 예: `enters`, `mounts`, `wipes` | 프레임 순서를 가르는 결정적 단서 |
| **부속성분 (Background State)** | 핵심 사건을 수식하거나 동시에 일어나는 상황 | 지속성(Stative/Activity) — 예: `filming`, `holding`, `carrying` | 순서 판단에 거의 영향 없음 (4프레임 내내 유지되는 배경일 뿐) |

* **실전 Tip**: 라벨링할 때 문장에서 **"이 부분이 문장에서 빠지거나 사라져도 이미지 순서가 안 바뀌면 부속성분"**이라고 판단하면 빠르게 골라낼 수 있습니다.

---

## 📐 6. 모호성 지수 (Ambiguity Index, AI) 구성 — 참고용

$$AI = 1.0 - \min(1.0,\ w_1 S_{temp} + w_2 S_{aspect} + w_3 S_{motion})$$

* **$S_{temp}$ (시간적 제약)**: 순차 접속사 있으면 ↑, 동시 접속사 있으면 ↓ (`then/before/after` 유무, 콤마+and 나열 개수, `while/as` 유무 반영)
* **$S_{aspect}$ (동작상)**: 경계성 동사 비율이 높을수록 명확함 (전체 동사 중 Telic 동사의 비율)
* **$S_{motion}$ (궤적/물리)**: 방향 전치사가 많을수록 힌트 ↑ (`down/up/towards/into` 등 궤적 전치사 개수)
* **결과 지표**:
  * **`AI = 0.0`**: 완전히 명확 (Explicit-Sequential 극단)
  * **`AI >= 0.8`**: 텍스트 힌트 사실상 없음, 이미지에 전적으로 의존 (Static-State 극단)
  * *실측 통계*: 평균 0.34 / 중앙값 0.22 / 최대 1.15

---

## 🎹 7. 라벨링할 때 이 표를 이렇게 쓰면 됨 (실전 조작)

1. 문장을 읽고 **주체+서술어 쌍이 몇 개인지** 먼저 셉니다 (섹션 3).
2. 5단계 표(섹션 4)에서 통사 구조가 가장 가까운 카테고리(1~5)를 찾습니다.
3. 애매하면 주성분/부속성분 구분법(섹션 5)으로 다시 확인합니다.
4. **5단계 중 어디에도 안 맞는 케이스는 별도로 짧게 태그만 남겨둡니다** (예: "오탈자", "구어체", "복합절3개+" 등) — 나중에 팀원들과 모여서 카테고리를 추가/조정할 때 사용합니다.
5. 입력창에 글을 적을 때 아래 짧은 태그 중심으로 작성하는 것을 권장합니다.
   * `Category: 1~5` (또는 "미해당")
   * `주성분 동사`: (있으면 적기)
   * `부속성분 있음/없음`
   * `이상신호`: 오탈자 / 문장이 이미지랑 안 맞음 / 판단불가
   * *예시 (프로그램이 pre-populate 해놓은 형태)*:
     `Category: 4, Main verb: wipe, Background state: No, Anomalies: None`

---

## ⚠️ 8. 데이터 품질 관련 참고
* 원본 캡션에 오탈자가 섞여 있을 수 있습니다 (예: `he` ➡️ `the` 오타, `row` ➡️ `oar` 오타로 추정 등). 
* 라벨링 중 이런 케이스를 발견하면 입력창 텍스트 뒤에 쉼표를 찍고 **`이상신호: 오탈자`** 혹은 **`이상신호: 판단불가`** 태그를 추가해 주면, 나중에 룰베이스 파서의 커버리지 한계를 확인하고 오염된 데이터를 정제하는 데 활용할 수 있습니다.

---

## 💾 9. 최종 데이터 취합 (Merge) 방법

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
