import os
import sys
import ssl

# Windows SSL 인증서 검증 우회 설정 (학내망/방화벽 대응)
ssl._create_default_https_context = ssl._create_unverified_context

# Windows OpenMP 중복 로딩 충돌 우회 설정
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Force stdout to UTF-8 to prevent cp949 encode errors in Windows
sys.stdout.reconfigure(encoding='utf-8')

import ast
import glob
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

MSE_THRESHOLD = 1200
CLIP_THRESHOLD = 0.20 # Cosine Distance threshold (semantic change)

class ClipSceneCutInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SNU AI Challenge - CLIP + MSE 하이브리드 장면 검수기 (랜덤 100개)")
        self.root.geometry("1200x700")
        
        # 1. 데이터 로드 및 샘플링
        if not os.path.exists(TRAIN_CSV):
            messagebox.showerror("Error", f"train.csv가 없습니다: {TRAIN_CSV}")
            self.root.destroy()
            return
            
        print("Loading train.csv...")
        df = pd.read_csv(TRAIN_CSV)
        self.sample_df = df.sample(n=100, random_state=42).reset_index(drop=True)
        
        # 2. Torch Hub를 통한 CLIP 모델 로드 (transformers 완전 우회)
        print("Loading CLIP model via Torch Hub (openai/CLIP)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.clip_model, self.clip_preprocess = torch.hub.load("openai/CLIP", "ViT_B_32", trust_repo=True)
            self.clip_model = self.clip_model.to(self.device).eval()
            print(f"CLIP loaded successfully on device: {self.device}")
        except Exception as e:
            messagebox.showerror("Error", f"CLIP 모델 로드 중 오류 발생:\n{e}\n인터넷 연결이 필요할 수 있습니다.")
            self.root.destroy()
            return
            
        self.cached_results = {}
        self.current_idx = 0
        
        self.setup_ui()
        self.load_sample()
        
    def setup_ui(self):
        # 상단 네비게이션바
        self.info_frame = tk.Frame(self.root, bg="#2C3E50", pady=5)
        self.info_frame.pack(fill="x")
        
        self.progress_label = tk.Label(
            self.info_frame, text="", fg="white", bg="#2C3E50",
            font=("Malgun Gothic", 12, "bold")
        )
        self.progress_label.pack()
        
        # 분석 비교 영역
        self.analysis_frame = tk.Frame(self.root, bg="#ECF0F1", pady=10)
        self.analysis_frame.pack(fill="x")
        
        self.label_title = tk.Label(
            self.analysis_frame, text="🔍 하이브리드 장면 전환 분석 지표", bg="#ECF0F1",
            font=("Malgun Gothic", 11, "bold"), fg="#2C3E50"
        )
        self.label_title.pack()
        
        self.stats_label = tk.Label(
            self.analysis_frame, text="동작 및 의미 벡터 연산 중...", bg="#ECF0F1",
            font=("Malgun Gothic", 10, "bold"), fg="#D35400", justify="center"
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
            self.root, text="💡 [MSE]가 높고 [CLIP]이 낮으면 구도/앵글 변화(동적 씬)  |  [CLIP]과 [MSE]가 모두 높으면 진짜 장면 전환(Cut) 입니다.",
            fg="#7F8C8D", font=("Malgun Gothic", 9)
        )
        self.guide_label.pack(side="bottom", pady=5)
        
    def compute_image_mse(self, img_path1, img_path2):
        try:
            with Image.open(img_path1).convert('L') as img1, Image.open(img_path2).convert('L') as img2:
                img1_r = img1.resize((64, 64), Image.Resampling.BILINEAR)
                img2_r = img2.resize((64, 64), Image.Resampling.BILINEAR)
                arr1 = np.array(img1_r, dtype=np.float32)
                arr2 = np.array(img2_r, dtype=np.float32)
                return float(np.mean((arr1 - arr2) ** 2))
        except Exception:
            return None
            
    def compute_clip_distance(self, img_path1, img_path2):
        try:
            with Image.open(img_path1).convert("RGB") as img1, Image.open(img_path2).convert("RGB") as img2:
                # preprocess 이미지 텐서 생성
                t1 = self.clip_preprocess(img1).unsqueeze(0).to(self.device)
                t2 = self.clip_preprocess(img2).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # 피처 벡터 추출 및 정규화
                    f1 = self.clip_model.encode_image(t1)
                    f2 = self.clip_model.encode_image(t2)
                    f1 = f1 / f1.norm(p=2, dim=-1, keepdim=True)
                    f2 = f2 / f2.norm(p=2, dim=-1, keepdim=True)
                    
                    # 코사인 유사도 계산
                    cos_sim = torch.clamp(torch.matmul(f1, f2.T), -1.0, 1.0).item()
                    return 1.0 - cos_sim # Cosine Distance
        except Exception as e:
            print(f"CLIP Error: {e}")
            return None
            
    def load_sample(self):
        if self.current_idx < 0 or self.current_idx >= len(self.sample_df):
            return
            
        row = self.sample_df.iloc[self.current_idx]
        sample_id = str(row['Id'])
        sentence = row['Sentence']
        ans = ast.literal_eval(row['Answer'])
        
        # 1. 텍스트 임시 셋업
        self.progress_label.config(text=f"CLIP + MSE 하이브리드 장면 검수기 | {self.current_idx + 1} / {len(self.sample_df)}")
        self.id_label.config(text=f"Sample ID: {sample_id} | Correct Order (Answer): {ans}")
        self.sentence_text.config(text=sentence)
        self.stats_label.config(text="물리적/의미적 인접 오차 연산 중...", fg="#D35400")
        
        # 2. 정답 시간 순서 정렬
        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        ordered_files = [None] * 4
        for idx, pos in enumerate(ans):
            ordered_files[pos - 1] = shuffled_files[idx]
            
        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in ordered_files]
        
        # 3. 오차 연산 (실시간 & 캐싱)
        if sample_id in self.cached_results:
            mse_vals, clip_dists = self.cached_results[sample_id]
        else:
            mse1 = self.compute_image_mse(img_paths[0], img_paths[1])
            mse2 = self.compute_image_mse(img_paths[1], img_paths[2])
            mse3 = self.compute_image_mse(img_paths[2], img_paths[3])
            
            clip1 = self.compute_clip_distance(img_paths[0], img_paths[1])
            clip2 = self.compute_clip_distance(img_paths[1], img_paths[2])
            clip3 = self.compute_clip_distance(img_paths[2], img_paths[3])
            
            mse_vals = (mse1, mse2, mse3)
            clip_dists = (clip1, clip2, clip3)
            self.cached_results[sample_id] = (mse_vals, clip_dists)
            
        # 4. 연산 결과 출력 및 분석 판정
        if None not in mse_vals and None not in clip_dists:
            text_lines = []
            
            # 각 구간 상태 판정
            true_cuts = 0
            for i in range(3):
                m = mse_vals[i]
                c = clip_dists[i]
                
                # 시각적 컷 전환 조건: CLIP 거리가 높음
                is_cut = c >= CLIP_THRESHOLD
                # 구도/움직임이 큰 동일 장면 조건: MSE가 높지만 CLIP 거리는 낮음
                is_dynamic = (m >= MSE_THRESHOLD) and (c < CLIP_THRESHOLD)
                
                if is_cut:
                    status = "🎬 진짜 장면 전환 (True Cut)"
                    true_cuts += 1
                elif is_dynamic:
                    status = "🔄 앵글/구도 변화 (Dynamic Action)"
                else:
                    status = "📌 동일 장면 (Same Scene)"
                    
                text_lines.append(f"{i+1}➡️{i+2}구간: MSE={m:.1f}, CLIP={c:.3f} ➡️ {status}")
                
            stats_text = f"의미적 장면 전환 횟수: {true_cuts}회\n" + "  |  ".join(text_lines)
            self.stats_label.config(text=stats_text, fg="#2C3E50")
        else:
            self.stats_label.config(text="Error: 이미지가 손상되었거나 CLIP 벡터를 계산할 수 없습니다.", fg="#C0392B")
            
        # 5. 이미지 바인딩
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
        if self.current_idx < len(self.sample_df) - 1:
            self.current_idx += 1
            self.load_sample()

if __name__ == "__main__":
    root = tk.Tk()
    app = ClipSceneCutInspectorApp(root)
    root.mainloop()
