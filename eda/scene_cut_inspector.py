import os
import glob
import ast
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
from concurrent.futures import ThreadPoolExecutor

# Force stdout to UTF-8 to prevent cp949 encode errors in Windows
import sys
sys.stdout.reconfigure(encoding='utf-8')

# =========================================================================
# [경로 설정]
# =========================================================================
DATA_DIR = "./snuaichallenge_data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train")

# 동일 장면/장면 전환 임계값 기준
CUT_THRESHOLD = 1200

def compute_image_mse(img_path1, img_path2):
    try:
        with Image.open(img_path1).convert('L') as img1, Image.open(img_path2).convert('L') as img2:
            # 빠른 계산을 위해 64x64로 리사이즈
            img1_r = img1.resize((64, 64), Image.Resampling.BILINEAR)
            img2_r = img2.resize((64, 64), Image.Resampling.BILINEAR)
            arr1 = np.array(img1_r, dtype=np.float32)
            arr2 = np.array(img2_r, dtype=np.float32)
            return float(np.mean((arr1 - arr2) ** 2))
    except Exception:
        return None

class SceneCutInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SNU AI Challenge - 장면 전환 검수기 (랜덤 100개)")
        self.root.geometry("1150x680")
        
        # 1. 데이터 로드 및 랜덤 100개 추출
        if not os.path.exists(TRAIN_CSV):
            messagebox.showerror("Error", f"train.csv가 없습니다: {TRAIN_CSV}")
            self.root.destroy()
            return
            
        print("Loading train.csv...")
        df = pd.read_csv(TRAIN_CSV)
        # 시드 고정을 통해 동일 세트로 반복 검수 가능 (랜덤 100개)
        self.sample_df = df.sample(n=100, random_state=42).reset_index(drop=True)
        self.cached_results = {}
        
        self.current_idx = 0
        
        # UI 레이아웃 구축
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
        
        # 컷 분석 수치 영역
        self.analysis_frame = tk.Frame(self.root, bg="#ECF0F1", pady=8)
        self.analysis_frame.pack(fill="x")
        
        self.stats_label = tk.Label(
            self.analysis_frame, text="동작 오차 계산 중...", bg="#ECF0F1",
            font=("Malgun Gothic", 11, "bold"), fg="#C0392B"
        )
        self.stats_label.pack()
        
        # 문장 영역
        self.sentence_frame = tk.Frame(self.root, pady=10)
        self.sentence_frame.pack(fill="x")
        
        self.id_label = tk.Label(self.sentence_frame, text="", font=("Arial", 11, "bold"), fg="#2980B9")
        self.id_label.pack()
        
        self.sentence_text = tk.Label(
            self.sentence_frame, text="", font=("Malgun Gothic", 12, "bold"),
            wraplength=1050, justify="center", fg="#34495E"
        )
        self.sentence_text.pack(pady=5)
        
        # 이미지 배치 영역
        self.img_frame = tk.Frame(self.root)
        self.img_frame.pack(pady=10)
        
        self.img_labels = []
        for i in range(4):
            lbl = tk.Label(self.img_frame, borderwidth=2, relief="solid", bg="gray")
            lbl.pack(side="left", padx=10)
            self.img_labels.append(lbl)
            
        # 하단 조작계
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
        
        # 하단 가이드 문구
        self.guide_label = tk.Label(
            self.root, text="💡 4장의 이미지는 원래 정답 순서(시간 순서)대로 정렬되어 나타납니다. 키보드 방향키(← / →)로 스와이프 하세요.",
            fg="#7F8C8D", font=("Malgun Gothic", 9)
        )
        self.guide_label.pack(side="bottom", pady=5)
        
    def load_sample(self):
        if self.current_idx < 0 or self.current_idx >= len(self.sample_df):
            return
            
        row = self.sample_df.iloc[self.current_idx]
        sample_id = str(row['Id'])
        sentence = row['Sentence']
        ans = ast.literal_eval(row['Answer'])
        
        # 1. 정보창 일시 업데이트
        self.progress_label.config(text=f"랜덤 장면 전환 검수기 | {self.current_idx + 1} / {len(self.sample_df)}")
        self.id_label.config(text=f"Sample ID: {sample_id} | Correct Order (Answer): {ans}")
        self.sentence_text.config(text=sentence)
        self.stats_label.config(text="픽셀 오차값(MSE) 연산 진행 중...", fg="#E67E22")
        
        # 2. 정답 시간 순서대로 프레임 정렬
        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        ordered_files = [None] * 4
        for idx, pos in enumerate(ans):
            ordered_files[pos - 1] = shuffled_files[idx]
            
        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in ordered_files]
        
        # 3. 3가지 인접 전이 구간 오차 연산 (캐싱 지원)
        if sample_id in self.cached_results:
            mse1, mse2, mse3 = self.cached_results[sample_id]
        else:
            mse1 = compute_image_mse(img_paths[0], img_paths[1])
            mse2 = compute_image_mse(img_paths[1], img_paths[2])
            mse3 = compute_image_mse(img_paths[2], img_paths[3])
            self.cached_results[sample_id] = (mse1, mse2, mse3)
            
        # 4. 연산 결과 판정 및 수치 출력
        if None not in (mse1, mse2, mse3):
            # 전환 판정
            cut1 = "장면 전환" if mse1 >= CUT_THRESHOLD else "동일 장면"
            cut2 = "장면 전환" if mse2 >= CUT_THRESHOLD else "동일 장면"
            cut3 = "장면 전환" if mse3 >= CUT_THRESHOLD else "동일 장면"
            
            cuts_count = sum([1 for m in (mse1, mse2, mse3) if m >= CUT_THRESHOLD])
            
            stats_text = (
                f"총 장면 전환 횟수: {cuts_count}회  |  "
                f"1-2구간: {mse1:.1f} ({cut1})  |  "
                f"2-3구간: {mse2:.1f} ({cut2})  |  "
                f"3-4구간: {mse3:.1f} ({cut3})"
            )
            self.stats_label.config(text=stats_text, fg="#2C3E50")
        else:
            self.stats_label.config(text="Error: 일부 이미지가 존재하지 않거나 연산할 수 없습니다.", fg="#C0392B")
            
        # 5. 시간순 정렬 이미지 로딩 및 바인딩
        for i, img_path in enumerate(img_paths):
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    img = img.resize((240, 135), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.img_labels[i].config(image=photo, text="")
                    self.img_labels[i].image = photo
                except Exception as e:
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
    app = SceneCutInspectorApp(root)
    root.mainloop()
