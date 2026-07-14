import os
import sys
import ast
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

# Force stdout to UTF-8 to prevent cp949 encode errors in Windows
sys.stdout.reconfigure(encoding='utf-8')

# =========================================================================
# [경로 및 설정]
# =========================================================================
DATA_DIR = "./snuaichallenge_data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
FEATURES_CSV = os.path.join(DATA_DIR, "snu_clip_features.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train")

# 왼쪽 봉우리 분기선
LEFT_PEAK_THRESHOLD = 0.20

class FullLeftPeakInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🟢 SNU AI Challenge - 전수 데이터 왼쪽 봉우리(정적 비디오) 검수기")
        self.root.geometry("1200x700")
        
        # 1. 파일 검증 및 로드
        if not os.path.exists(TRAIN_CSV):
            messagebox.showerror("Error", f"train.csv가 없습니다: {TRAIN_CSV}")
            self.root.destroy()
            return
            
        # Search in root first, then in DATA_DIR
        possible_paths = ["snu_clip_features.csv", os.path.join(DATA_DIR, "snu_clip_features.csv")]
        features_path = None
        for path in possible_paths:
            if os.path.exists(path):
                features_path = path
                break
                
        if not features_path:
            messagebox.showerror("Error", "캐글에서 다운받은 snu_clip_features.csv 파일이 프로젝트 루트 폴더나 snuaichallenge_data 폴더에 있어야 합니다!")
            self.root.destroy()
            return
            
        print(f"Loading datasets...")
        self.train_df = pd.read_csv(TRAIN_CSV)
        self.feat_df = pd.read_csv(features_path)
        
        # 병합
        self.merged_df = pd.merge(self.train_df, self.feat_df[['Id', 'Max', 'Mean']], on='Id', how='inner')
        
        # 2. 왼쪽 봉우리 (Max < 0.20) 필터링
        self.static_df = self.merged_df[self.merged_df['Max'] < LEFT_PEAK_THRESHOLD].reset_index(drop=True)
        print(f"Total static videos found: {len(self.static_df)}")
        
        if len(self.static_df) == 0:
            messagebox.showerror("Error", "필터링된 정적 비디오가 0개입니다.")
            self.root.destroy()
            return
            
        self.current_idx = 0
        self.setup_ui()
        self.load_sample()
        
    def setup_ui(self):
        # 상단 네비게이션바
        self.info_frame = tk.Frame(self.root, bg="#27AE60", pady=5)
        self.info_frame.pack(fill="x")
        
        self.progress_label = tk.Label(
            self.info_frame, text="", fg="white", bg="#27AE60",
            font=("Malgun Gothic", 12, "bold")
        )
        self.progress_label.pack()
        
        # 분석 지표 영역
        self.analysis_frame = tk.Frame(self.root, bg="#ECF0F1", pady=10)
        self.analysis_frame.pack(fill="x")
        
        self.label_title = tk.Label(
            self.analysis_frame, text="🔍 전수 데이터 중 정적 비디오(장면 전환 0회) 분석", bg="#ECF0F1",
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
            self.root, text="💡 이 비디오들은 9,535개 전수조사 중 CLIP Max 오차가 0.20 미만인 정적 비디오(총 2,028개) 중 하나입니다.",
            fg="#7F8C8D", font=("Malgun Gothic", 9)
        )
        self.guide_label.pack(side="bottom", pady=5)
        
    def load_sample(self):
        if self.current_idx < 0 or self.current_idx >= len(self.static_df):
            return
            
        row = self.static_df.iloc[self.current_idx]
        sample_id = str(row['Id'])
        sentence = row['Sentence']
        ans = ast.literal_eval(row['Answer'])
        max_clip = row['Max']
        mean_clip = row['Mean']
        
        # UI 업데이트
        self.progress_label.config(text=f"전수 정적 비디오 검수기 | {self.current_idx + 1} / {len(self.static_df)}")
        self.id_label.config(text=f"Sample ID: {sample_id} | Correct Order (Answer): {ans}")
        self.sentence_text.config(text=sentence)
        
        stats_text = f"CLIP Max 오차: {max_clip:.3f} (0.20 미만)  |  CLIP 평균 오차: {mean_clip:.3f}\n" \
                     f"판정 결과: 🟢 동일 장면 내 미세 행동 비디오 (Z-Score 기준 하한값 영역)"
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
        if self.current_idx < len(self.static_df) - 1:
            self.current_idx += 1
            self.load_sample()

if __name__ == "__main__":
    root = tk.Tk()
    app = FullLeftPeakInspectorApp(root)
    root.mainloop()
