import os
import sys
import ssl

# Windows SSL 및 OpenMP 우회
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Force stdout to UTF-8 to prevent cp949 encode errors in Windows
sys.stdout.reconfigure(encoding='utf-8')

import ast
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import torch

# =========================================================================
# [경로 및 설정]
# =========================================================================
DATA_DIR = "./snuaichallenge_data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train")

# 왼쪽 봉우리를 거르는 절대 기준선 (이 값보다 작으면 100% 미세행동)
LEFT_PEAK_THRESHOLD = 0.20
TARGET_COUNT = 25 # 찾을 정적 비디오의 개수

class LeftPeakInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 SNU AI Challenge - 왼쪽 봉우리(정적 비디오) 실물 검수기")
        self.root.geometry("1200x700")
        
        # 1. 데이터 로드
        if not os.path.exists(TRAIN_CSV):
            messagebox.showerror("Error", f"train.csv가 없습니다: {TRAIN_CSV}")
            self.root.destroy()
            return
            
        self.df = pd.read_csv(TRAIN_CSV)
        self.sample_pool = self.df.sample(frac=1, random_state=42).reset_index(drop=True) # 셔플
        
        # 2. CLIP 모델 로드
        print("\nLoading CLIP model via Torch Hub (openai/CLIP) on CPU...")
        self.device = "cpu" # 로컬 호환성을 위해 CPU 고정
        try:
            self.clip_model, self.clip_preprocess = torch.hub.load("openai/CLIP", "ViT_B_32", trust_repo=True)
            self.clip_model = self.clip_model.to(self.device).eval()
            print("CLIP loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"CLIP 모델 로드 실패:\n{e}")
            self.root.destroy()
            return
            
        self.static_samples = []
        self.cached_results = {}
        self.current_idx = 0
        
        # 3. 정적 비디오(왼쪽 봉우리) 실시간 탐색 시작
        self.scan_static_videos()
        
        if len(self.static_samples) == 0:
            messagebox.showerror("Error", "정적 비디오를 찾지 못했습니다.")
            self.root.destroy()
            return
            
        self.setup_ui()
        self.load_sample()
        
    def scan_static_videos(self):
        print(f"\nScanning for {TARGET_COUNT} static videos (Max_CLIP < {LEFT_PEAK_THRESHOLD})...")
        print("Please wait...")
        
        checked = 0
        for idx, row in self.sample_pool.iterrows():
            if len(self.static_samples) >= TARGET_COUNT:
                break
                
            sample_id = str(row['Id'])
            shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
            img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in shuffled_files]
            
            # 이미지 4장 존재 여부 체크
            if not all(os.path.exists(p) for p in img_paths):
                continue
                
            checked += 1
            
            # CLIP 피처 추출 및 거리 연산
            try:
                # 4장 한 번에 배치 처리
                imgs = [self.clip_preprocess(Image.open(p).convert("RGB")) for p in img_paths]
                img_tensor = torch.stack(imgs).to(self.device)
                
                with torch.no_grad():
                    features = self.clip_model.encode_image(img_tensor)
                    features = features / features.norm(p=2, dim=-1, keepdim=True)
                    features_np = features.cpu().numpy()
                    
                # 6개 조합 코사인 거리 계산
                clip_dists = []
                for p1, p2 in [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]:
                    cos_sim = float(np.dot(features_np[p1], features_np[p2]))
                    clip_dists.append(1.0 - cos_sim)
                    
                max_clip = max(clip_dists)
                mean_clip = sum(clip_dists) / 6.0
                
                # 왼쪽 봉우리(정적 비디오) 조건 필터링
                if max_clip < LEFT_PEAK_THRESHOLD:
                    self.static_samples.append({
                        'row': row,
                        'max_clip': max_clip,
                        'mean_clip': mean_clip,
                        'dists': clip_dists
                    })
                    print(f"  [Found {len(self.static_samples)}/{TARGET_COUNT}] ID: {sample_id} | Max CLIP: {max_clip:.3f} | Checked: {checked}")
                    
            except Exception as e:
                # 에러 발생 시 건너뜀
                pass
                
        print(f"\nScan completed! Found {len(self.static_samples)} static videos.")
        
    def setup_ui(self):
        # 상단 네비게이션바
        self.info_frame = tk.Frame(self.root, bg="#1ABC9C", pady=5)
        self.info_frame.pack(fill="x")
        
        self.progress_label = tk.Label(
            self.info_frame, text="", fg="white", bg="#1ABC9C",
            font=("Malgun Gothic", 12, "bold")
        )
        self.progress_label.pack()
        
        # 분석 비교 영역
        self.analysis_frame = tk.Frame(self.root, bg="#ECF0F1", pady=10)
        self.analysis_frame.pack(fill="x")
        
        self.label_title = tk.Label(
            self.analysis_frame, text="🟢 정적 비디오(왼쪽 봉우리) 지표 분석", bg="#ECF0F1",
            font=("Malgun Gothic", 11, "bold"), fg="#2C3E50"
        )
        self.label_title.pack()
        
        self.stats_label = tk.Label(
            self.analysis_frame, text="", bg="#ECF0F1",
            font=("Malgun Gothic", 10, "bold"), fg="#27AE60", justify="center"
        )
        self.stats_label.pack(pady=5)
        
        # 문장 영역
        self.sentence_frame = tk.Frame(self.root, pady=10)
        self.sentence_frame.pack(fill="x")
        
        self.id_label = tk.Label(self.sentence_frame, text="", font=("Arial", 11, "bold"), fg="#2980B9")
        self.id_label.pack()
        
        self.sentence_text = tk.Label(
            self.sentence_frame, text="", font=("Malgun Gothic", 12, "bold"),
            wraplength=1100, justify="center", fg="#2C3E50"
        )
        self.sentence_text.pack(pady=5)
        
        # 이미지 배치 영역
        self.img_frame = tk.Frame(self.root)
        self.img_frame.pack(pady=10)
        
        self.img_labels = []
        for i in range(4):
            lbl = tk.Label(self.img_frame, borderwidth=2, relief="solid", bg="gray")
            lbl.pack(side="left", padx=15)
            self.img_labels.append(lbl)
            
        # 하단 조작 버튼
        self.btn_frame = tk.Frame(self.root, pady=10)
        self.btn_frame.pack(fill="x")
        
        tk.Button(
            self.btn_frame, text="◀ 이전 (Left Arrow)", bg="#BDC3C7", fg="black",
            command=self.prev_sample, font=("Malgun Gothic", 10, "bold"), width=18
        ).pack(side="left", padx=50)
        
        tk.Button(
            self.btn_frame, text="다음 (Right Arrow) ▶", bg="#BDC3C7", fg="black",
            command=self.next_sample, font=("Malgun Gothic", 10, "bold"), width=18
        ).pack(side="right", padx=50)
        
        # 키보드 바인딩
        self.root.bind("<Left>", lambda event: self.prev_sample())
        self.root.bind("<Right>", lambda event: self.next_sample())
        
        # 하단 설명 가이드
        self.guide_label = tk.Label(
            self.root, text="💡 이 비디오들은 CLIP Max 오차가 0.20 미만인 극단적으로 고정된 장면(미세 행동) 데이터셋입니다.",
            fg="#7F8C8D", font=("Malgun Gothic", 9)
        )
        self.guide_label.pack(side="bottom", pady=5)
        
    def load_sample(self):
        if self.current_idx < 0 or self.current_idx >= len(self.static_samples):
            return
            
        data = self.static_samples[self.current_idx]
        row = data['row']
        max_clip = data['max_clip']
        mean_clip = data['mean_clip']
        
        sample_id = str(row['Id'])
        sentence = row['Sentence']
        ans = ast.literal_eval(row['Answer'])
        
        # UI 업데이트
        self.progress_label.config(text=f"정적 비디오 검수기 | {self.current_idx + 1} / {len(self.static_samples)}")
        self.id_label.config(text=f"Sample ID: {sample_id} | Correct Order (Answer): {ans}")
        self.sentence_text.config(text=sentence)
        
        stats_text = f"CLIP Max 오차: {max_clip:.3f} (장면 전환 기준인 0.20보다 낮음)  |  CLIP 평균 오차: {mean_clip:.3f}\n" \
                     f"판정 결과: 🟢 앵글 완벽 고정 / 동일 씬 내 미세 행동 비디오"
        self.stats_label.config(text=stats_text)
        
        # 정답 순서대로 이미지 정렬
        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        ordered_files = [None] * 4
        for idx, pos in enumerate(ans):
            ordered_files[pos - 1] = shuffled_files[idx]
            
        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in ordered_files]
        
        # 이미지 바인딩
        for i, img_path in enumerate(img_paths):
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    img = img.resize((240, 135), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.img_labels[i].config(image=photo, text="")
                    self.img_labels[i].image = photo
                except Exception:
                    self.img_labels[i].config(image='', text=f"Error\n{ordered_files[i]}")
            else:
                self.img_labels[i].config(image='', text=f"Not Found\n{ordered_files[i]}")
                
    def prev_sample(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_sample()
            
    def next_sample(self):
        if self.current_idx < len(self.static_samples) - 1:
            self.current_idx += 1
            self.load_sample()

if __name__ == "__main__":
    root = tk.Tk()
    app = LeftPeakInspectorApp(root)
    root.mainloop()
