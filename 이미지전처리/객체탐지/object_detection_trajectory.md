# 🤖 [최종본] SNU AI Challenge - 객체 탐지 및 궤적 분석 (Object Trajectory Tracking) 설계안

본 보고서는 이미지-문장 순서 정렬 문제에서 사물의 물리적 이동 궤적을 캡션 문장의 사건 흐름과 연동하기 위한 **객체 탐지 및 시공간 궤적 분석**의 최종 설계 및 구현 가이드라인입니다.

---

## 📸 1. 개요 및 학술적 배경

* **선행연구**: *Made to Order (ECCV 2024)* / *Arrow of Time (CVPR 2018)*
* **기본 사상**: 비디오의 시간 흐름은 물리적 법칙(예: 물체의 낙하, 특정 방향으로의 이동)을 따릅니다. 캡션의 `"raises the tool higher(도구를 더 높이 든다)"`, `"moves to the right(오른쪽으로 이동)"` 같은 동작은 이미지 속 핵심 객체의 **y좌표(높이)나 x좌표(좌우)의 흐름과 1:1로 매핑**됩니다.

---

## 🛠️ 2. YOLOv8s의 한계 극복: OWL-ViT (Open-Vocabulary) 채택

* **YOLOv8s의 한계**: COCO 80개 사전정의 클래스만 감지하므로 이발기(`clippers`), 화장 붓(`brush`), 빗(`comb`) 등 이 대회 데이터셋의 핵심 도구들을 아예 감지하지 못합니다.
* **해결책**: Open-Vocabulary 객체 탐지기인 `google/owlvit-base-patch32`를 탑재하여 문장에 명시된 임의의 사물명을 그대로 이미지에서 감지하고 바운딩 박스를 추출합니다.

---

## 🚨 3. 핵심 예외 처리 및 최적화 해결책 (Exception Handling)

### A. 선형 보간의 한계 극복 ➔ 결측치 마스킹(Masking) 처리
* **선형 보간의 문제점**: 인물의 관절 움직임이나 사물의 이동 궤적은 대부분 비선형적(Non-linear)입니다. 탐지가 누락된 중간 프레임을 직선으로 강제 보간하면 오차가 증폭되며, 특히 1번 또는 4번 프레임의 누락 시 외삽(Extrapolation) 오차가 기하급수적으로 커집니다.
* **최적화 해결책**: 가짜 데이터를 생성해 빈 공간을 채우는 대신, 탐지가 실패한 프레임은 **`no detection (skip)`으로 마스킹 처리하여 VLM이 스스로 무시**하도록 유도합니다. VLM은 4장 중 3장의 위치 데이터만으로도 단조성(Monotonicity) 추세를 유추하기에 충분히 강력합니다.

### B. if-else 가중치의 한계 극복 ➔ CLIP Text Embedding 기반 동적 가중치 배분
* **규칙 기반의 문제점**: "zoom", "closer" 등 특정 단어를 if-else 구문으로 직접 코딩할 경우, 평가 환경에서 예상치 못한 동의어가 등장하면 매핑이 실패(Lexical Overfitting)합니다. 또한 "걸어오면서 도구를 집어든다"와 같은 복합 맥락을 처리할 수 없습니다.
* **최적화 해결책 (VRAM 0MB 추가)**: 
  * 새로운 임베딩 모델을 추가하는 대신, 우리가 이미 로컬에 적재해 둔 **CLIP의 Text Encoder**를 재활용합니다.
  * 입력된 캡션 문장 벡터와 아래의 두 기준 앵커 벡터(Anchor Vector) 간의 코사인 유사도를 실시간 연산하여, 두 분석 모듈의 신뢰도 가중치(0.0 ~ 1.0)를 수학적으로 자동 할당합니다.
    * **깊이 축 앵커 (Depth/Zoom)**: `"camera zooming in or out, moving closer or further away"`
    * **평면 이동 축 앵커 (Trajectory)**: `"object moving left, right, up, or down"`

---

## 💻 4. 파이프라인 구현 코드 (Python)

```python
import os
import torch
from PIL import Image

# OpenMP duplicate runtime fix for PyTorch on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from transformers import OwlViTProcessor, OwlViTForObjectDetection

class OWLViTTrajectoryExtractor:
    def __init__(self, model_name="google/owlvit-base-patch32", device="cpu"):
        self.device = device
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def extract_object_trajectory(self, image_paths, query_text, threshold=0.10):
        """
        4장의 이미지 경로와 Open-Vocabulary 쿼리 텍스트를 받아 객체 위치 및 면적 변화를 추적합니다.
        탐지 실패 시 가짜 보간을 배제하고 명시적 결측치(Masking) 플래그를 생성합니다.
        """
        coords = []
        text_queries = [[query_text]]
        
        for idx, img_path in enumerate(image_paths):
            if not os.path.exists(img_path):
                coords.append(f"- Image {idx+1}: file not found (skip)")
                continue
                
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            img_area = w * h
            
            inputs = self.processor(text=text_queries, images=img, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            target_sizes = torch.tensor([img.size[::-1]], dtype=torch.float32).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=threshold
            )[0]
            
            boxes = results["boxes"].cpu().numpy()
            
            if len(boxes) == 0:
                # 결측치 마스킹 처리 (VLM이 스스로 무시하도록 유도)
                coords.append(f"- Image {idx+1}: no '{query_text}' detected (skip this cue)")
                continue
                
            # 다중 객체 Identity 혼선 방지: 면적이 가장 큰 BBox를 주 피사체로 선택
            best_idx = 0
            max_area = 0
            for i, box in enumerate(boxes):
                box_w = box[2] - box[0]
                box_h = box[3] - box[1]
                area = box_w * box_h
                if area > max_area:
                    max_area = area
                    best_idx = i
                    
            best_box = boxes[best_idx]
            x_center = ((best_box[0] + best_box[2]) / 2) / w
            y_center = ((best_box[1] + best_box[3]) / 2) / h
            best_area_ratio = max_area / img_area
            
            coords.append(
                f"- Image {idx+1}: '{query_text}' center=[X={x_center:.3f}, Y={y_center:.3f}], Area={best_area_ratio*100:.1f}%"
            )
            
        return coords
```

---

## 📈 5. VLM 프롬프트 결합 및 매핑 예시

```text
[Visual Object Trajectory Hints]
- Image 1: 'clippers' center=[X=0.450, Y=0.210], Area=12.4%
- Image 2: 'clippers' center=[X=0.480, Y=0.450], Area=18.3%
- Image 3: no 'clippers' detected (skip this cue)  <-- (마스킹 처리된 결측치)
- Image 4: 'clippers' center=[X=0.520, Y=0.880], Area=36.6%

[Dynamic Feature Weighting Cues (via CLIP Text Similarity)]
- Depth/Zoom weight: 0.15
- Spatial Trajectory weight: 0.85 (Focus highly on X/Y Coordinate movements)

[Inference Task]
Using the storyline where the clippers are moved down to the bottom (Y-coordinate increases), match the trajectory center coordinates of Image 1, 2, and 4 (skipping Image 3) to reconstruct the correct chronological sequence.
```
