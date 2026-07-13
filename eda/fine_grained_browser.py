import os
import glob
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

# =========================================================================
# [설정 및 경로]
# =========================================================================
DATA_DIR = "./snuaichallenge_data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
CACHE_CSV = "./eda/dataset_mse_cache.csv"

class FineGrainedBrowserApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SNU AI Challenge - 미세 행동(Fine-grained) 이미지 브라우저")
        self.root.geometry("1100x600")
        
        # 1. 데이터 로드 및 필터링
        if not os.path.exists(CACHE_CSV):
            messagebox.showerror("Error", f"MSE 캐시 파일이 없습니다: {CACHE_CSV}\n먼저 층화 추출 스크립트를 실행했는지 확인해 주세요.")
            self.root.destroy()
            return
            
        print("Loading MSE cache...")
        df = pd.read_csv(CACHE_CSV)
        # 미세 행동(Fine-grained) 데이터만 필터링 후 MSE 오름차순 정렬
        self.fg_df = df[df['Type'] == 'Fine-grained'].sort_values(by="Avg_MSE").reset_index(drop=True)
        
        if len(self.fg_df) == 0:
            messagebox.showinfo("정보", "미세 행동(Fine-grained)으로 분류된 샘플이 없습니다.")
            self.root.destroy()
            return
            
        self.current_idx = 0
        
        self.setup_ui()
        self.load_sample()
        
    def setup_ui(self):
        # 상단 정보 프레임
        self.info_frame = tk.Frame(self.root, bg="#2C3E50", pady=5)
        self.info_frame.pack(fill="x")
        
        self.progress_label = tk.Label(
            self.info_frame, text="", fg="white", bg="#2C3E50",
            font=("Malgun Gothic", 12, "bold")
        )
        self.progress_label.pack()
        
        # 중단 문장 프레임
        self.sentence_frame = tk.Frame(self.root, pady=10)
        self.sentence_frame.pack(fill="x")
        
        self.id_label = tk.Label(self.sentence_frame, text="", font=("Arial", 11, "bold"), fg="#2980B9")
        self.id_label.pack()
        
        self.sentence_text = tk.Label(
            self.sentence_frame, text="", font=("Malgun Gothic", 12, "bold"),
            wraplength=1000, justify="center", fg="#34495E"
        )
        self.sentence_text.pack(pady=5)
        
        # 이미지 가로 정렬 프레임
        self.img_frame = tk.Frame(self.root)
        self.img_frame.pack(pady=10)
        
        self.img_labels = []
        for i in range(4):
            lbl = tk.Label(self.img_frame, borderwidth=2, relief="solid", bg="gray")
            lbl.pack(side="left", padx=10)
            self.img_labels.append(lbl)
            
        # 하단 조작 버튼
        self.btn_frame = tk.Frame(self.root, pady=10)
        self.btn_frame.pack(fill="x")
        
        self.prev_btn = tk.Button(
            self.btn_frame, text="◀ 이전 (Left Arrow)", bg="#BDC3C7", fg="black",
            command=self.prev_sample, font=("Malgun Gothic", 10, "bold"), width=18
        )
        self.prev_btn.pack(side="left", padx=50)
        
        self.next_btn = tk.Button(
            self.btn_frame, text="다음 (Right Arrow) ▶", bg="#BDC3C7", fg="black",
            command=self.next_sample, font=("Malgun Gothic", 10, "bold"), width=18
        )
        self.next_btn.pack(side="right", padx=50)
        
        # 키보드 단축키 연결
        self.root.bind("<Left>", lambda event: self.prev_sample())
        self.root.bind("<Right>", lambda event: self.next_sample())
        
        # 가이드 라벨
        self.guide_label = tk.Label(
            self.root, text="💡 키보드 왼쪽(←) / 오른쪽(→) 방향키를 눌러 총 152개의 미세 행동 이미지를 넘겨볼 수 있습니다.",
            fg="#7F8C8D", font=("Malgun Gothic", 9)
        )
        self.guide_label.pack(side="bottom", pady=10)
        
    def load_sample(self):
        if self.current_idx < 0 or self.current_idx >= len(self.fg_df):
            return
            
        row = self.fg_df.iloc[self.current_idx]
        sample_id = str(row['Id'])
        sentence = row['Sentence']
        avg_mse = float(row['Avg_MSE'])
        
        # 1. 정보 업데이트
        self.progress_label.config(
            text=f"미세 행동 데이터 브라우저 | {self.current_idx + 1} / {len(self.fg_df)} (순위: {self.current_idx + 1}위)"
        )
        self.id_label.config(text=f"Sample ID: {sample_id} | 평균 인접 MSE: {avg_mse:.2f}")
        self.sentence_text.config(text=sentence)
        
        # 2. 이미지 가로 배열 출력
        img_files = sorted(glob.glob(os.path.join(TRAIN_DIR, sample_id, "*.jpg")))
        for i in range(4):
            if i < len(img_files):
                img_path = img_files[i]
                try:
                    img = Image.open(img_path)
                    img = img.resize((240, 135), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    
                    self.img_labels[i].config(image=photo, text="")
                    self.img_labels[i].image = photo
                except Exception as e:
                    self.img_labels[i].config(image='', text=f"Error\n{os.path.basename(img_path)}")
            else:
                self.img_labels[i].config(image='', text="No Image")
                
    def prev_sample(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_sample()
            
    def next_sample(self):
        if self.current_idx < len(self.fg_df) - 1:
            self.current_idx += 1
            self.load_sample()

if __name__ == "__main__":
    root = tk.Tk()
    app = FineGrainedBrowserApp(root)
    root.mainloop()
