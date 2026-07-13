import os
import ast
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

# =========================================================================
# [설정 및 경로]
# =========================================================================
DATA_DIR = "./snuaichallenge_data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train")
OUTPUT_CSV = "./eda/labeled_1_3001.csv"

class ImageLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SNU AI Challenge - 개념화 검수 라벨러 (1~3001)")
        self.root.geometry("1100x700")
        
        # 1. 데이터 로드 및 정렬
        if not os.path.exists(TRAIN_CSV):
            messagebox.showerror("Error", f"train.csv 파일을 찾을 수 없습니다: {TRAIN_CSV}")
            self.root.destroy()
            return
            
        print("Loading train.csv...")
        df = pd.read_csv(TRAIN_CSV)
        df = df.sort_values(by="Id").reset_index(drop=True)
        self.target_df = df.iloc[0:3001].copy()
        
        # 2. 기존 마킹 및 검수 완료 데이터 불러오기 (이어하기 지원)
        self.labels = {}
        self.confirmed = {}
        
        if os.path.exists(OUTPUT_CSV):
            try:
                progress_df = pd.read_csv(OUTPUT_CSV)
                # Labeled_Concept 불러오기
                for _, row in progress_df.iterrows():
                    sample_id = str(row['Id'])
                    if not pd.isna(row.get('Labeled_Concept')):
                        self.labels[sample_id] = str(row['Labeled_Concept'])
                    # Confirmed 컬럼 불러오기 (True 인 것만 True로 맵핑)
                    if 'Confirmed' in progress_df.columns:
                        if str(row.get('Confirmed')).lower() in ['true', '1', '1.0']:
                            self.confirmed[sample_id] = True
                print(f"로드 성공: 기존 작성 {len(self.labels)}개, 검수 완료 {len(self.confirmed)}개.")
            except Exception as e:
                print(f"이어하기 파일 읽기 오류: {e}")
                
        # 현재 시작할 인덱스 찾기 (검수 완료(Confirmed)가 안 된 첫 번째 행)
        self.current_idx = 0
        for idx, row in self.target_df.iterrows():
            if str(row['Id']) not in self.confirmed:
                self.current_idx = idx
                break
                
        self.current_idx = min(self.current_idx, len(self.target_df) - 1)
        
        # UI 레이아웃 설정
        self.setup_ui()
        self.load_sample()
        
    def setup_ui(self):
        # 상단 진행률 정보
        self.info_frame = tk.Frame(self.root, bg="#2C3E50", pady=5)
        self.info_frame.pack(fill="x")
        
        self.progress_label = tk.Label(
            self.info_frame, text="", fg="white", bg="#2C3E50",
            font=("Malgun Gothic", 12, "bold")
        )
        self.progress_label.pack()
        
        # 중단 문장 출력 프레임
        self.sentence_frame = tk.Frame(self.root, pady=10)
        self.sentence_frame.pack(fill="x")
        
        self.id_label = tk.Label(self.sentence_frame, text="", font=("Arial", 11, "bold"), fg="#2980B9")
        self.id_label.pack()
        
        self.sentence_text = tk.Label(
            self.sentence_frame, text="", font=("Malgun Gothic", 13),
            wraplength=1000, justify="center", fg="#2C3E50"
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
            
        # 하단 입력 및 조작 영역
        self.input_frame = tk.Frame(self.root, pady=15)
        self.input_frame.pack(fill="x")
        
        # 개념화 입력창 라벨
        tk.Label(
            self.input_frame, text="✍️ 이미지/문맥 검수 및 개념화 기술 (Enter 입력 시 승인 후 다음 이동):",
            font=("Malgun Gothic", 11, "bold"), fg="#2C3E50"
        ).pack(anchor="w", padx=40)
        
        # 텍스트 입력 Entry
        self.concept_entry = tk.Entry(
            self.input_frame, font=("Malgun Gothic", 12), width=90,
            borderwidth=2, relief="groove"
        )
        self.concept_entry.pack(pady=5, padx=40, fill="x")
        self.concept_entry.focus_set()
        
        # 조작 버튼 프레임
        self.control_frame = tk.Frame(self.root, pady=5)
        self.control_frame.pack(fill="x")
        
        self.prev_btn = tk.Button(
            self.control_frame, text="◀ 이전 (Alt + Left)", bg="#BDC3C7", fg="black",
            command=self.prev_sample, font=("Malgun Gothic", 10, "bold")
        )
        self.prev_btn.pack(side="left", padx=40)
        
        self.next_btn = tk.Button(
            self.control_frame, text="다음 / Skip (Alt + Right) ▶", bg="#BDC3C7", fg="black",
            command=self.next_sample, font=("Malgun Gothic", 10, "bold")
        )
        self.next_btn.pack(side="right", padx=40)
        
        # 키보드 이벤트 바인딩
        self.concept_entry.bind("<Return>", lambda event: self.save_concept()) # Enter 키 저장 및 승인
        self.root.bind("<Alt-Left>", lambda event: self.prev_sample())
        self.root.bind("<Alt-Right>", lambda event: self.next_sample())
        
        # 가이드 메시지
        self.guide_label = tk.Label(
            self.root, text="💡 힌트 검수 후 [Enter]를 누르면 승인(Confirmed) 후 자동 다음 이동! (이전 수정은 [Alt + ←])",
            fg="#7F8C8D", font=("Malgun Gothic", 9)
        )
        self.guide_label.pack(side="bottom", pady=10)
        
    def load_sample(self):
        if self.current_idx < 0 or self.current_idx >= len(self.target_df):
            return
            
        row = self.target_df.iloc[self.current_idx]
        sample_id = str(row['Id'])
        sentence = row['Sentence']
        
        # 1. 입력창 초기화 및 기존값 채워넣기
        self.concept_entry.delete(0, tk.END)
        if sample_id in self.labels:
            self.concept_entry.insert(0, self.labels[sample_id])
            
        self.concept_entry.focus_set()
        
        # 2. 진행 상황 라벨 업데이트
        confirmed_count = len(self.confirmed)
        total_count = len(self.target_df)
        percentage = (confirmed_count / total_count) * 100
        marked_status = " [검수완료]" if sample_id in self.confirmed else " [미검수]"
        
        self.progress_label.config(
            text=f"검수 상황: {confirmed_count} / {total_count} ({percentage:.1f}%) | 현재 샘플: {self.current_idx + 1}번째 행" + marked_status
        )
        self.id_label.config(text=f"Sample ID: {sample_id} | Answer: {row.get('Answer', 'N/A')}")
        self.sentence_text.config(text=sentence)
        
        # 3. 이미지 로드 및 리사이즈
        img_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        for i, img_file in enumerate(img_files):
            img_path = os.path.join(IMAGE_DIR, sample_id, img_file)
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    img = img.resize((240, 135), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.img_labels[i].config(image=photo)
                    self.img_labels[i].image = photo
                except Exception as e:
                    self.img_labels[i].config(image='', text=f"Error\n{img_file}")
            else:
                self.img_labels[i].config(image='', text=f"Not Found\n{img_file}")
                
    def save_concept(self):
        if self.current_idx < 0 or self.current_idx >= len(self.target_df):
            return
            
        row = self.target_df.iloc[self.current_idx]
        sample_id = str(row['Id'])
        concept_text = self.concept_entry.get().strip()
        
        if not concept_text:
            answer = messagebox.askyesno("경고", "설명이 비어 있습니다. 그대로 저장하고 넘어가시겠습니까?")
            if not answer:
                return
                
        # 1. 딕셔너리에 저장 및 승인(Confirmed) 마킹
        self.labels[sample_id] = concept_text
        self.confirmed[sample_id] = True
        self.save_to_csv()
        
        # 2. 다음 샘플로 이동
        if self.current_idx < len(self.target_df) - 1:
            self.current_idx += 1
            self.load_sample()
        else:
            messagebox.showinfo("완료", "1~3001번째 샘플 검수가 모두 끝났습니다! 고생하셨습니다.")
            
    def prev_sample(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_sample()
            
    def next_sample(self):
        if self.current_idx < len(self.target_df) - 1:
            self.current_idx += 1
            self.load_sample()
            
    def save_to_csv(self):
        df_copy = self.target_df.copy()
        df_copy['Labeled_Concept'] = df_copy['Id'].apply(lambda x: self.labels.get(str(x), None))
        # Confirmed 컬럼도 CSV에 같이 기록
        df_copy['Confirmed'] = df_copy['Id'].apply(lambda x: self.confirmed.get(str(x), False))
        
        # 다른 Labeled 열들도 보전하기 위해 기존 파일 열 복구 루틴 추가
        if os.path.exists(OUTPUT_CSV):
            try:
                progress_df = pd.read_csv(OUTPUT_CSV)
                for col in ['Labeled_Sentence_Level', 'Labeled_Video_Type', 'Labeled_Notes']:
                    if col in progress_df.columns:
                        mapping = progress_df.set_index('Id')[col].to_dict()
                        df_copy[col] = df_copy['Id'].map(mapping)
            except Exception as e:
                print(f"저장 중 기존 컬럼 유지 오류: {e}")
                
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df_copy.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabelerApp(root)
    root.mainloop()
