# SNU AI Challenge - 오답 샘플 분석 및 인간 성능 상한선 검수 가이드 (Wrong Sample Analysis)

본 가이드는 캐글 GPU(T4) 환경에서 200개 샘플을 대상으로 베이스라인 모델(`Qwen2-VL-2B`)을 돌려 획득한 **오답 166개**의 통계 분석 및 대표 오답 샘플에 대한 시각 검수용 리포트입니다.

---

## 1. 베이스라인 오답 패턴 분석 (Error Pattern Analysis)

* **검증 샘플 수**: 200개
* **정답 수**: 34개 (Exact Match Accuracy: **17.00%**)
* **오답 수**: 166개

오답 166개의 모델 예측값(`Model_Pred`) 분포를 분석한 결과, 매우 중요한 편향(Bias) 패턴이 발견되었습니다:

| 모델 예측값 (Model_Pred) | 오답 샘플 수 (Count) | 비율 (%) | 비고 |
| :--- | :--- | :--- | :--- |
| **`[1, 2, 3, 4]`** | **128개** | **77.1%** | **기본(Default) 예측 편향** |
| `[3, 4, 1, 2]` | 12개 | 7.2% | - |
| `[1, 2, 4, 3]` | 7개 | 4.2% | - |
| `[2, 4, 1, 3]` | 6개 | 3.6% | - |
| 기타 (6개 패턴) | 13개 | 7.8% | - |

> [!WARNING]
> * **기본값 편향**: 오답의 **77.1%**에서 모델이 `[1, 2, 3, 4]`(원래 입력된 순서대로 정렬)를 정답으로 예측했습니다.
> * **원인 분석**: `Qwen2-VL-2B`와 같은 소형 VLM은 Zero-shot 상태에서 4장 이미지 간의 복잡한 인과 관계나 시간 흐름을 읽어내지 못할 때, 프롬프트의 예시(`[1, 2, 3, 4]`)를 그대로 따라 쓰거나 이미지의 나열 순서대로 대답하는 경향이 강하게 나타납니다. 즉, 실질적인 의미 이해보다는 **디폴트 값으로 안전하게 찍는 상태**입니다.

---

## 2. 시각 검수용 대표 오답 샘플 (Human Inspection Targets)
오답의 정확한 원인 파악 및 **"사람도 정렬하기 힘든 불가능한 샘플"**을 가려내기 위해 3가지 대표 오답 샘플의 시각 시트를 로컬에 생성했습니다.

아래 이미지 파일 링크를 열어 직접 눈으로 검수해 보시기 바랍니다:

1. **샘플 1: 다중 씬 전환 (Multi-scene transitions) - `u7w0lr`**
   * **이미지 확인**: [wrong_inspect_u7w0lr.png](./assets/wrong_inspect_u7w0lr.png)
   * **텍스트**: *"A girl hula hoops indoors before the scene shifts outdoors to a cheering group on rocks; then, players swim towards the pool's center..."*
   * **정답**: `[3, 1, 2, 4]` (Frame 2 ➡️ Frame 3 ➡️ Frame 1 ➡️ Frame 4)
   * **모델 예측**: `[1, 2, 3, 4]`
   * **검수 가이드**: 실내 훌라후프(Frame 2), 야외 바위(Frame 3), 수영장(Frame 1, 4)이 완전히 분리되어 있어, 사람이 직관적으로 정렬하기는 쉬운 샘플입니다. 모델이 씬 분할과 키워드 매칭에 실패했음을 뜻합니다.

2. **샘플 2: 미세 동작 씬 (Fine-grained action) - `kSE41E`**
   * **이미지 확인**: [wrong_inspect_kSE41E.png](./assets/wrong_inspect_kSE41E.png)
   * **텍스트**: *"The man moves closer to the mirror, tilting his head up while shaving, then a towel is raised to a face as the camera zooms in on a hand reaching down to touch the water surface."*
   * **정답**: `[3, 2, 4, 1]` (Frame 4 ➡️ Frame 2 ➡️ Frame 1 ➡️ Frame 3)
   * **모델 예측**: `[1, 2, 3, 4]`
   * **검수 가이드**: 거울을 보고 면도하는 동작과 손이 물 표면에 닿는 세부 묘사가 섞여 있습니다. 프레임 간 픽셀 차이가 미세하고 행동 묘사의 주체를 헷갈리기 쉽습니다.

3. **샘플 3: 객체 상호작용 (Object interaction) - `Qrt4Ax`**
   * **이미지 확인**: [wrong_inspect_Qrt4Ax.png](./assets/wrong_inspect_Qrt4Ax.png)
   * **텍스트**: *"A person adjusts a bicycle as it shifts right and the camera lowers... then, the view focuses on hands attaching a bike accessory before revealing two similar items being held..."*
   * **정답**: `[4, 2, 1, 3]` (Frame 3 ➡️ Frame 2 ➡️ Frame 4 ➡️ Frame 1)
   * **모델 예측**: `[1, 2, 3, 4]`
   * **검수 가이드**: 자전거를 튜닝하고 부품을 부착하는 미세한 동작들로 구성되어 있어 시간 순서를 정확히 포착해야 합니다.

---

## 3. 인간 성능 상한선 (Human Performance Upper Bound) 추정 가이드

조장형 및 팀원들에게 최종 성능 상한 리포트를 전달하기 위해 아래 절차에 따라 오답 166개 중 일부를 직접 보고 체크해 보세요.

### 3.1 샘플 유형 분류
* **유형 A: 모델 단순 실패 (Model Failure)**
  * 문장에 명확한 지시어("실내에서 실외로", "그 다음 수영을 한다")가 있고 이미지 구분이 쉬우나 모델이 디폴트로 찍어서 틀린 경우. (성능 개선 가능 영역)
* **유형 B: 인간 불가능 샘플 (Ambiguous / Humanly Impossible)**
  * 프레임 간 시각적 변화가 없거나(정지 화면 중복), 텍스트의 설명만으로는 4장 사진의 정확한 1~4위 위치를 도저히 확정할 수 없는 모호한 데이터. (성능의 상한선 결정 요인)

### 3.2 상한선(Upper Bound) 계산식
$$\text{Estimated Human Accuracy Upper Bound} = 100\% - \left( \frac{\text{유형 B (인간 불가능 샘플) 개수}}{\text{총 검사 샘플 수}} \times 100\% \right)$$
지속적으로 오답 이미지를 검수하시면서 이 공식에 따라 상한선을 추정하면 매우 논리적이고 타당한 EDA 보고서를 작성할 수 있습니다.
