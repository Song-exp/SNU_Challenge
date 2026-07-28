import os
import torch
from PIL import Image

# OpenMP duplicate runtime fix for PyTorch on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from transformers import OwlViTProcessor, OwlViTForObjectDetection

class OWLViTTrajectoryExtractor:
    def __init__(self, model_name="google/owlvit-base-patch32", device="cpu"):
        self.device = device
        print(f"Loading {model_name} on {device}...")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("OWL-ViT Model Loaded successfully!")
        
    def extract_object_trajectory(self, image_paths, query_text, threshold=0.10):
        """
        4장의 이미지 경로와 Open-Vocabulary 쿼리 텍스트를 받아 객체 위치 및 면적 변화를 추적합니다.
        """
        coords = []
        
        # 쿼리를 리스트 형태로 구성 (OWL-ViT 배치 입력 규격)
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
                
            # 포스트 프로세싱 (바운딩 박스 크기 복원)
            target_sizes = torch.tensor([img.size[::-1]], dtype=torch.float32).to(self.device)
            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=threshold
            )[0]
            
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            
            if len(boxes) == 0:
                # Fallback 수정: 0.5, 0.5 기본값 대신 명시적 관측 실패 선언
                coords.append(f"- Image {idx+1}: no '{query_text}' detected (skip this cue)")
                continue
                
            # Identity 혼선 방지: 면적이 가장 큰 BBox를 주 피사체로 선택
            best_idx = 0
            max_area = 0
            
            for i, box in enumerate(boxes):
                box_w = box[2] - box[0]
                box_h = box[3] - box[1]
                area = box_w * box_h
                if area > max_area:
                    max_area = area
                    best_idx = i
                    
            # 매칭된 가장 큰 객체의 정규화 좌표 계산
            best_box = boxes[best_idx]
            x_center = ((best_box[0] + best_box[2]) / 2) / w
            y_center = ((best_box[1] + best_box[3]) / 2) / h
            best_area_ratio = max_area / img_area
            
            coords.append(
                f"- Image {idx+1}: '{query_text}' center=[X={x_center:.3f}, Y={y_center:.3f}], Area={best_area_ratio*100:.1f}%"
            )
            
        return coords

if __name__ == "__main__":
    # CPU 데모 실행
    extractor = OWLViTTrajectoryExtractor(device="cpu")
    
    # 00GGp0 실제 이미지 경로 매핑
    demo_images = [
        "00GGp0_czj.jpg",
        "00GGp0_fuo.jpg",
        "00GGp0_tuc.jpg",
        "00GGp0_xfq.jpg"
    ]
    
    # 00GGp0에 실제로 존재하는 객체인 "kayak"으로 쿼리 주입
    query = "kayak" 
    
    print(f"\nQuerying: '{query}' on images...")
    
    full_paths = [os.path.join("C:/Users/user/Desktop/서울대/snuaichallenge_data/train/00GGp0", p) for p in demo_images]
    exists = all(os.path.exists(p) for p in full_paths)
    
    if exists:
        trajectory = extractor.extract_object_trajectory(full_paths, query)
        
        print("\n=== OWL-ViT 시공간 궤적 힌트 결과 ===")
        for line in trajectory:
            print(line)
    else:
        print("데모 이미지가 로컬에 존재하지 않습니다.")
