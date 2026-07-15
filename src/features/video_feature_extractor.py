import os
import numpy as np
import torch
from PIL import Image
import pickle

class RealTimeFeatureExtractor:
    """
    훈련(Training) 및 추론(Inference) 단계에서 실시간으로 4장의 이미지로부터
    CLIP 거리, MSE 거리, 그리고 장면 전환 판정값을 추출하여 프롬프트 힌트로 주입할 수 있도록 돕는 헬퍼 클래스입니다.
    """
    def __init__(self, device=None, transformer_path=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[RealTimeFeatureExtractor] Initializing CLIP model on device: {self.device}...")
        try:
            import torch.hub
            self.clip_model, self.clip_preprocess = torch.hub.load("openai/CLIP", "ViT_B_32", trust_repo=True)
            self.clip_model = self.clip_model.to(self.device).eval()
            print("[RealTimeFeatureExtractor] CLIP model loaded successfully.")
        except Exception as e:
            print(f"[RealTimeFeatureExtractor] Failed to load CLIP model: {e}")
            raise e
            
        # Z-Score 정적 스케일링을 위한 학습셋 기준 평균/표준편차 (QuantileTransformer 대용으로 경량 설계)
        # 오프라인 학습셋 통계 기반 표준 스케일링
        self.clip_max_mean, self.clip_max_std = 0.3152, 0.1344
        self.clip_mean_mean, self.clip_mean_std = 0.2264, 0.0961
        self.mse_max_mean, self.mse_max_std = 6200.0, 3100.0  # 예시 기준값
        self.mse_mean_mean, self.mse_mean_std = 4300.0, 2100.0

    def calculate_mse(self, img1, img2):
        # 16:9 종횡비 표준 해상도(320x180)로 리사이즈하여 안정적인 픽셀 오차 계산
        img1_res = img1.resize((320, 180), Image.Resampling.LANCZOS)
        img2_res = img2.resize((320, 180), Image.Resampling.LANCZOS)
        a1 = np.array(img1_res, dtype=np.float32)
        a2 = np.array(img2_res, dtype=np.float32)
        return float(np.mean((a1 - a2) ** 2))

    def extract_features(self, images):
        """
        images: 4개의 PIL Image 객체 리스트 [img1, img2, img3, img4] (정렬 이전 셔플 순서 그대로 입력)
        반환값: 각 피처 수치 및 정규화(Z-Score) 점수를 포함한 딕셔너리
        """
        if len(images) != 4:
            raise ValueError("비디오 세트에는 정확히 4장의 이미지가 필요합니다.")

        # ---------------------------------------------------------------------
        # 1. 픽셀 MSE 물리 오차 계산 (6개 이미지 조합)
        # ---------------------------------------------------------------------
        mse_12 = self.calculate_mse(images[0], images[1])
        mse_13 = self.calculate_mse(images[0], images[2])
        mse_14 = self.calculate_mse(images[0], images[3])
        mse_23 = self.calculate_mse(images[1], images[2])
        mse_24 = self.calculate_mse(images[1], images[3])
        mse_34 = self.calculate_mse(images[2], images[3])
        
        mse_arr = np.array([mse_12, mse_13, mse_14, mse_23, mse_24, mse_34])
        max_mse = np.max(mse_arr)
        mean_mse = np.mean(mse_arr)
        ratio_mse = max_mse / mean_mse if mean_mse > 0 else 1.0

        # ---------------------------------------------------------------------
        # 2. CLIP 코사인 의미 오차 계산 (6개 이미지 조합)
        # ---------------------------------------------------------------------
        preprocessed = [self.clip_preprocess(img) for img in images]
        img_tensor = torch.stack(preprocessed).to(self.device)
        
        with torch.no_grad():
            features = self.clip_model.encode_image(img_tensor)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            feats = features.cpu().numpy()

        def get_clip_dist(f1, f2):
            return float(1.0 - np.dot(f1, f2))

        c12 = get_clip_dist(feats[0], feats[1])
        c13 = get_clip_dist(feats[0], feats[2])
        c14 = get_clip_dist(feats[0], feats[3])
        c23 = get_clip_dist(feats[1], feats[2])
        c24 = get_clip_dist(feats[1], feats[3])
        c34 = get_clip_dist(feats[2], feats[3])

        clip_arr = np.array([c12, c13, c14, c23, c24, c34])
        max_clip = np.max(clip_arr)
        mean_clip = np.mean(clip_arr)
        ratio_clip = max_clip / mean_clip if mean_clip > 0 else 1.0

        # ---------------------------------------------------------------------
        # 3. 안엄격(느슨한) 기준 장면 전환 횟수 계산 (CLIP < 0.20 유사쌍 기준)
        # ---------------------------------------------------------------------
        similar_clip_pairs = sum([1 for c in clip_arr if c < 0.20])
        if similar_clip_pairs >= 5:
            predicted_scene_cuts = 0
        elif 2 <= similar_clip_pairs <= 4:
            predicted_scene_cuts = 1
        elif similar_clip_pairs == 1:
            predicted_scene_cuts = 2
        else:
            predicted_scene_cuts = 3

        # ---------------------------------------------------------------------
        # 4. Z-Score 표준화 적용 (프롬프트 입력용)
        # ---------------------------------------------------------------------
        max_clip_scaled = (max_clip - self.clip_max_mean) / self.clip_max_std
        mean_clip_scaled = (mean_clip - self.clip_mean_mean) / self.clip_mean_std
        
        max_mse_scaled = (max_mse - self.mse_max_mean) / self.mse_max_std
        mean_mse_scaled = (mean_mse - self.mse_mean_mean) / self.mse_mean_std

        return {
            'mse_12': mse_12, 'mse_13': mse_13, 'mse_14': mse_14,
            'mse_23': mse_23, 'mse_24': mse_24, 'mse_34': mse_34,
            'Max_mse': max_mse, 'Mean_mse': mean_mse, 'Ratio_mse': ratio_mse,
            'Max_mse_scaled': max_mse_scaled, 'Mean_mse_scaled': mean_mse_scaled,
            
            'clip_12': c12, 'clip_13': c13, 'clip_14': c14,
            'clip_23': c23, 'clip_24': c24, 'clip_34': c34,
            'Max_clip': max_clip, 'Mean_clip': mean_clip, 'Ratio_clip': ratio_clip,
            'Max_clip_scaled': max_clip_scaled, 'Mean_clip_scaled': mean_clip_scaled,
            
            'predicted_scene_cuts': predicted_scene_cuts
        }
