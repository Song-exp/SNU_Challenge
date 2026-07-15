import os
import sys
import ast
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

# Force stdout to UTF-8 to prevent cp949 encode errors in Windows
sys.stdout.reconfigure(encoding='utf-8')

# =========================================================================
# [경로 설정]
# =========================================================================
DATA_DIR = "./snuaichallenge_data"
SOURCE_CSV = "./train_검토_최종_완료.csv"          # 원본 검토용 CSV (미터치 보존)
TRAIN_CSV = "./train_검토_최종_완료_수정본.csv"    # 복제 및 실제 수정용 CSV
FEATURES_CSV = "./snu_clip_features.csv"
IMAGE_DIR = os.path.join(DATA_DIR, "train")

# ID 기반 담당 범위 인덱스 정의
REVIEWER_RANGES = {
    "병철": (0, 2999),    # 00GGp0 ~ EquxBk
    "서현": (3000, 6002),  # er2p3e ~ oQI68U
    "정현": (6003, 9534)   # oqImQK ~ ZzYxAm
}

class GUISyntaxInspectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 SNU AI Challenge - 문법성분 및 장면전환 통합 검수기 (GUI)")
        self.root.geometry("1450x900")
        self.root.configure(bg="#F5F7FA")
        
        # 1. 복제 파일 존재 여부 확인 및 생성
        if not os.path.exists(TRAIN_CSV):
            if not os.path.exists(SOURCE_CSV):
                messagebox.showerror("Error", f"원본 train_검토_최종_완료.csv 파일이 없습니다: {SOURCE_CSV}")
                self.root.destroy()
                return
            print(f"Creating duplicate copy: {TRAIN_CSV} ...")
            try:
                shutil.copy(SOURCE_CSV, TRAIN_CSV)
                print("Duplicate file generated successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"복제 파일 생성 중 에러 발생: {e}")
                self.root.destroy()
                return
            
        print("Loading datasets...")
        try:
            self.train_df = pd.read_csv(TRAIN_CSV, encoding='cp949')
        except Exception as e:
            messagebox.showerror("Error", f"복제본 CSV 로드 실패: {e}")
            self.root.destroy()
            return
            
        # CLIP 피처 머지
        if os.path.exists(FEATURES_CSV):
            self.feat_df = pd.read_csv(FEATURES_CSV)
            max_col = 'Max_clip' if 'Max_clip' in self.feat_df.columns else 'Max'
            mean_col = 'Mean_clip' if 'Mean_clip' in self.feat_df.columns else 'Mean'
            
            temp_df = self.feat_df[['Id', max_col, mean_col]].copy()
            temp_df.rename(columns={max_col: 'Max_clip', mean_col: 'Mean_clip'}, inplace=True)
            
            self.merged_df = pd.merge(self.train_df, temp_df, on='Id', how='left')
        else:
            self.merged_df = self.train_df.copy()
            self.merged_df['Max_clip'] = 0.0
            self.merged_df['Mean_clip'] = 0.0
            
        # 신규 수정 컬럼 및 모호성 체크용 컬럼 초기화
        for col in ['수정된 장면 전환 횟수', '수정된 고유 주어 개수', '수정된 서술어 개수', '수정된 Partition']:
            if col not in self.merged_df.columns:
                self.merged_df[col] = np.nan
        if '모호_여부' not in self.merged_df.columns:
            self.merged_df['모호_여부'] = False
        if '모호_이유' not in self.merged_df.columns:
            self.merged_df['모호_이유'] = ""
            
        if '검수_자' not in self.merged_df.columns:
            self.merged_df['검수_자'] = ""
        if '검수_완료' not in self.merged_df.columns:
            self.merged_df['검수_완료'] = False
            
        self.reviewer_name = "Unknown"
        self.start_idx = 0
        self.end_idx = len(self.merged_df) - 1
        self.current_idx = 0
        
        self.ask_reviewer()
        
    def ask_reviewer(self):
        self.modal = tk.Toplevel(self.root)
        self.modal.title("검수자 선택")
        self.modal.geometry("400x300")
        self.modal.configure(bg="#2C3E50")
        self.modal.grab_set()
        
        lbl = tk.Label(
            self.modal, text="검수자를 선택해 주세요", fg="#ECF0F1", bg="#2C3E50",
            font=("Malgun Gothic", 12, "bold"), pady=20
        )
        lbl.pack()
        
        btn_frame = tk.Frame(self.modal, bg="#2C3E50")
        btn_frame.pack()
        
        for name, (s, e) in REVIEWER_RANGES.items():
            btn = tk.Button(
                btn_frame, text=f"{name} ({s} ~ {e})", font=("Malgun Gothic", 10, "bold"),
                width=25, bg="#3498DB", fg="white", pady=5,
                command=lambda n=name: self.set_reviewer(n)
            )
            btn.pack(pady=8)
            
    def set_reviewer(self, name):
        self.reviewer_name = name
        self.start_idx, self.end_idx = REVIEWER_RANGES[name]
        self.modal.destroy()
        
        slice_df = self.merged_df.loc[self.start_idx:self.end_idx]
        
        # 🛡️ 껐다 켜기 매칭 수정: True가 아닌 모든 행(False 및 NaN 빈칸)을 미완료로 판단하여 검출
        uncompleted = slice_df[slice_df['검수_완료'] != True]
        
        if not uncompleted.empty:
            self.current_idx = uncompleted.index[0]
            print(f"Resuming at index: {self.current_idx} for reviewer {self.reviewer_name}")
        else:
            self.current_idx = self.start_idx
            
        self.setup_ui()
        self.load_sample()
        
    def setup_ui(self):
        # 1. 상단 타이틀 영역
        self.title_frame = tk.Frame(self.root, bg="#2C3E50", pady=8)
        self.title_frame.pack(fill="x")
        
        self.title_label = tk.Label(
            self.title_frame, text="🔍 전수 데이터 중 정적 비디오(장면 전환 0회) 분석 및 구문 정보 검수 (복제 수정본)", fg="#ECF0F1", bg="#2C3E50",
            font=("Malgun Gothic", 13, "bold")
        )
        self.title_label.pack()
        
        # 2. 이미지 대조 정보 표시 영역
        self.stats_frame = tk.Frame(self.root, bg="#EAEDED", pady=10, relief="groove", bd=2)
        self.stats_frame.pack(fill="x", padx=15, pady=5)
        
        self.clip_label = tk.Label(
            self.stats_frame, text="", bg="#EAEDED", font=("Malgun Gothic", 10, "bold"), fg="#2C3E50"
        )
        self.clip_label.pack()
        
        self.decision_label = tk.Label(
            self.stats_frame, text="", bg="#EAEDED", font=("Malgun Gothic", 10, "bold"), fg="#27AE60"
        )
        self.decision_label.pack(pady=2)
        
        # 3. ID 및 문장 영역
        self.info_frame = tk.Frame(self.root, bg="#F5F7FA", pady=5)
        self.info_frame.pack(fill="x")
        
        self.id_label = tk.Label(self.info_frame, text="", font=("Arial", 11, "bold"), fg="#2980B9", bg="#F5F7FA")
        self.id_label.pack()
        
        self.sentence_label = tk.Label(
            self.info_frame, text="", font=("Malgun Gothic", 11, "bold"),
            wraplength=1300, justify="center", fg="#2C3E50", bg="#F5F7FA"
        )
        self.sentence_label.pack(pady=5)
        
        # 4. 이미지 프레임 영역
        self.img_frame = tk.Frame(self.root, bg="#F5F7FA")
        self.img_frame.pack(pady=10)
        
        self.img_labels = []
        for i in range(4):
            lbl = tk.Label(self.img_frame, borderwidth=3, relief="solid", bg="#BDC3C7")
            lbl.pack(side="left", padx=12)
            self.img_labels.append(lbl)
            
        # 5. 수치 입력 및 수정 영역
        self.edit_frame = tk.LabelFrame(
            self.root, text=" 🛠️ 구문 정보 및 장면 전환 수정 (수정 후 엔터를 누르면 복제본인 'train_검토_최종_완료_수정본.csv'에 기록됩니다) ",
            font=("Malgun Gothic", 10, "bold"), bg="#F5F7FA", fg="#2C3E50", bd=2, relief="groove"
        )
        self.edit_frame.pack(fill="x", padx=80, pady=5)
        
        self.edit_frame.columnconfigure((0, 1, 2, 3, 4, 5, 6, 7), weight=1)
        
        # [Row 0] 수정 값 입력 필드
        tk.Label(self.edit_frame, text="장면 전환 횟수:", font=("Malgun Gothic", 10), bg="#F5F7FA").grid(row=0, column=0, sticky="e", pady=8)
        self.entry_cuts = tk.Entry(self.edit_frame, font=("Arial", 10, "bold"), width=8, justify="center")
        self.entry_cuts.grid(row=0, column=1, sticky="w")
        
        tk.Label(self.edit_frame, text="고유 주어 개수:", font=("Malgun Gothic", 10), bg="#F5F7FA").grid(row=0, column=2, sticky="e", pady=8)
        self.entry_subjs = tk.Entry(self.edit_frame, font=("Arial", 10, "bold"), width=8, justify="center")
        self.entry_subjs.grid(row=0, column=3, sticky="w")
        
        tk.Label(self.edit_frame, text="서술어 개수:", font=("Malgun Gothic", 10), bg="#F5F7FA").grid(row=0, column=4, sticky="e", pady=8)
        self.entry_preds = tk.Entry(self.edit_frame, font=("Arial", 10, "bold"), width=8, justify="center")
        self.entry_preds.grid(row=0, column=5, sticky="w")
        
        tk.Label(self.edit_frame, text="통사 구조 (Type):", font=("Malgun Gothic", 10), bg="#F5F7FA").grid(row=0, column=6, sticky="e", pady=8)
        self.var_partition = tk.StringVar(self.root)
        self.menu_partition = tk.OptionMenu(self.edit_frame, self.var_partition, "Type-1", "Type-2", "Type-3")
        self.menu_partition.config(font=("Arial", 9, "bold"), bg="white", width=10)
        self.menu_partition.grid(row=0, column=7, sticky="w")
        
        # [Row 1] 추출된 원본 주어 / 동사 단어 리스트 표시 영역
        self.var_is_ambiguous = tk.BooleanVar()
        self.chk_ambiguous = tk.Checkbutton(
            self.edit_frame, text="⚠️ 애매함 / 모호한 샘플", 
            variable=self.var_is_ambiguous, font=("Malgun Gothic", 10, "bold"), 
            bg="#F5F7FA", fg="#D35400", selectcolor="white", activebackground="#F5F7FA"
        )
        self.chk_ambiguous.grid(row=1, column=0, columnspan=2, padx=10, pady=6, sticky="w")
        
        # 추출 주어 리스트
        tk.Label(self.edit_frame, text="추출 주어:", font=("Malgun Gothic", 9, "bold"), bg="#F5F7FA", fg="#2980B9").grid(row=1, column=2, sticky="e")
        self.lbl_subj_words = tk.Label(self.edit_frame, text="", font=("Arial", 9, "bold"), bg="#F5F7FA", fg="#2C3E50", anchor="w")
        self.lbl_subj_words.grid(row=1, column=3, columnspan=2, sticky="w")
        
        # 추출 서술어 리스트
        tk.Label(self.edit_frame, text="추출 서술어:", font=("Malgun Gothic", 9, "bold"), bg="#F5F7FA", fg="#27AE60").grid(row=1, column=5, sticky="e")
        self.lbl_pred_words = tk.Label(self.edit_frame, text="", font=("Arial", 9, "bold"), bg="#F5F7FA", fg="#2C3E50", anchor="w")
        self.lbl_pred_words.grid(row=1, column=6, columnspan=2, sticky="w")
        
        # [Row 2] 모호한 이유 입력 필드
        tk.Label(self.edit_frame, text="모호한 이유 / 메모:", font=("Malgun Gothic", 10, "bold"), bg="#F5F7FA", fg="#34495E").grid(row=2, column=0, sticky="e", pady=6)
        self.entry_reason = tk.Entry(self.edit_frame, font=("Malgun Gothic", 10), bg="white", fg="#2C3E50")
        self.entry_reason.grid(row=2, column=1, columnspan=7, padx=5, pady=6, sticky="ew")
        
        # [Row 3] 분류 가이드 안내문
        guide_text = (
            "💡 [통사 구조 분류 기준 및 대표 예시]\n"
            " • Type-1 (단일 절) : 주어-서술어 1쌍 (동사 단락 1개)  ➔  예: \"A boy kicks the ball.\"\n"
            " • Type-2 (복합 종속) : 주절 + 종속절 / 분사구문 (-ing)  ➔  예: \"The camera zooms out, showing a tattoo on the man's arm.\"\n"
            " • Type-3 (대등 병렬) : and 또는 콤마로 동사 대등 연결  ➔  예: \"The chef chops onions and mixes them in a bowl.\""
        )
        self.guide_label = tk.Label(
            self.edit_frame, 
            text=guide_text, 
            font=("Malgun Gothic", 9, "bold"), fg="#5D6D7E", bg="#F5F7FA", 
            justify="left", anchor="w", pady=8
        )
        self.guide_label.grid(row=3, column=0, columnspan=8, padx=20, sticky="w")
        
        # 6. 하단 네비게이션 영역
        self.nav_frame = tk.Frame(self.root, bg="#F5F7FA", pady=10)
        self.nav_frame.pack(fill="x")
        
        self.btn_prev = tk.Button(
            self.nav_frame, text="◀ 이전 (Backspace)", bg="#7F8C8D", fg="white",
            command=self.prev_sample, font=("Malgun Gothic", 10, "bold"), width=18, relief="raised"
        )
        self.btn_prev.pack(side="left", padx=100)
        
        self.progress_label = tk.Label(
            self.nav_frame, text="", font=("Malgun Gothic", 11, "bold"), bg="#F5F7FA", fg="#34495E"
        )
        self.progress_label.pack(side="left", expand=True)
        
        self.btn_next = tk.Button(
            self.nav_frame, text="저장 및 다음 (Enter) ▶", bg="#3498DB", fg="white",
            command=self.next_sample, font=("Malgun Gothic", 10, "bold"), width=24, relief="raised"
        )
        self.btn_next.pack(side="right", padx=100)
        
        self.root.bind("<Return>", lambda event: self.next_sample())
        self.root.bind("<BackSpace>", self.on_backspace)
        
    def on_backspace(self, event):
        focused_widget = self.root.focus_get()
        if isinstance(focused_widget, tk.Entry):
            return
        self.prev_sample()
        
    def load_sample(self):
        if self.current_idx < self.start_idx or self.current_idx > self.end_idx:
            return
            
        row = self.merged_df.iloc[self.current_idx]
        sample_id = str(row['Id'])
        sentence = row['Sentence']
        ans = ast.literal_eval(row['Answer'])
        
        max_clip = float(row['Max_clip']) if not pd.isna(row['Max_clip']) else 0.0
        mean_clip = float(row['Mean_clip']) if not pd.isna(row['Mean_clip']) else 0.0
        
        self.clip_label.config(
            text=f"CLIP Max 오차: {max_clip:.3f} ({'0.20 미만' if max_clip < 0.20 else '0.20 이상'}) | CLIP 평균 오차: {mean_clip:.3f}"
        )
        if max_clip < 0.20:
            self.decision_label.config(
                text="판정 결과: 🟢 동일 장면 내 미세 행동 비디오 (Z-Score 기준 하한값 영역)", fg="#27AE60"
            )
        else:
            self.decision_label.config(
                text="판정 결과: 🎬 장면 전환 비디오 (씬 경계 감지)", fg="#E74C3C"
            )
            
        self.progress_label.config(text=f"검수자: {self.reviewer_name} | 검수 진행률: {self.current_idx + 1} / {self.end_idx + 1}")
        
        show_partition = row['수정된 Partition'] if not pd.isna(row['수정된 Partition']) else row['Partition']
        self.id_label.config(text=f"Sample ID: {sample_id} | Correct Order (Answer): {ans} | 구조: {show_partition}")
        self.sentence_label.config(text=sentence)
        
        def safe_int_str(val, fallback_val):
            if pd.isna(val):
                return str(int(fallback_val))
            try:
                return str(int(float(val)))
            except:
                return str(int(fallback_val))
        
        val_cuts = safe_int_str(row['수정된 장면 전환 횟수'], row['장면 전환 횟수'])
        val_subjs = safe_int_str(row['수정된 고유 주어 개수'], row['고유 주어 개수'])
        val_preds = safe_int_str(row['수정된 서술어 개수'], row['서술어 개수'])
        
        self.entry_cuts.delete(0, tk.END); self.entry_cuts.insert(0, val_cuts)
        self.entry_subjs.delete(0, tk.END); self.entry_subjs.insert(0, val_subjs)
        self.entry_preds.delete(0, tk.END); self.entry_preds.insert(0, val_preds)
        
        self.var_partition.set(show_partition)
        
        is_ambig = bool(row['모호_여부']) if not pd.isna(row['모호_여부']) else False
        reason = str(row['모호_이유']) if not pd.isna(row['모호_이유']) else ""
        
        self.var_is_ambiguous.set(is_ambig)
        self.entry_reason.delete(0, tk.END); self.entry_reason.insert(0, reason)
        
        # 추출된 단어 목록 갱신
        subj_words = str(row['고유 주어']) if not pd.isna(row['고유 주어']) else "[주어 없음]"
        pred_words = str(row['서술어']) if not pd.isna(row['서술어']) else "[서술어 없음]"
        self.lbl_subj_words.config(text=subj_words)
        self.lbl_pred_words.config(text=pred_words)
        
        shuffled_files = [row['Input_1'], row['Input_2'], row['Input_3'], row['Input_4']]
        ordered_files = [None] * 4
        for idx, pos in enumerate(ans):
            ordered_files[pos - 1] = shuffled_files[idx]
            
        img_paths = [os.path.join(IMAGE_DIR, sample_id, f) for f in ordered_files]
        for i, img_path in enumerate(img_paths):
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    img = img.resize((260, 146), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    self.img_labels[i].config(image=photo, text="")
                    self.img_labels[i].image = photo
                except Exception:
                    self.img_labels[i].config(image='', text=f"Error\n{ordered_files[i]}", fg="red")
            else:
                self.img_labels[i].config(image='', text=f"Not Found\n{ordered_files[i]}", fg="red")
                
        self.entry_cuts.focus_set()
        self.entry_cuts.selection_range(0, tk.END)

    def save_current_corrections(self):
        try:
            new_cuts = int(self.entry_cuts.get().strip())
            new_subjs = int(self.entry_subjs.get().strip())
            new_preds = int(self.entry_preds.get().strip())
        except ValueError:
            messagebox.showwarning("입력 오류", "수정 항목에는 반드시 숫자(정수)만 입력해야 합니다!")
            return False
            
        new_part = self.var_partition.get()
        new_ambig = self.var_is_ambiguous.get()
        new_reason = self.entry_reason.get().strip()
        
        # 1. merged_df 메모리 갱신
        self.merged_df.at[self.current_idx, '수정된 장면 전환 횟수'] = new_cuts
        self.merged_df.at[self.current_idx, '수정된 고유 주어 개수'] = new_subjs
        self.merged_df.at[self.current_idx, '수정된 서술어 개수'] = new_preds
        self.merged_df.at[self.current_idx, '수정된 Partition'] = new_part
        self.merged_df.at[self.current_idx, '모호_여부'] = new_ambig
        self.merged_df.at[self.current_idx, '모호_이유'] = new_reason
        self.merged_df.at[self.current_idx, '검수_자'] = self.reviewer_name
        self.merged_df.at[self.current_idx, '검수_완료'] = True
            
        # 2. train_df 데이터프레임 동기화
        for col in ['수정된 장면 전환 횟수', '수정된 고유 주어 개수', '수정된 서술어 개수', '수정된 Partition', '모호_여부', '모호_이유', '검수_자', '검수_완료']:
            self.train_df.at[self.current_idx, col] = self.merged_df.at[self.current_idx, col]
            
        save_cols = ['Id', 'Input_1', 'Input_2', 'Input_3', 'Input_4', 'Sentence', 'Answer', 
                     'No_ordering', 'Partition', '서술어 개수', '서술어', '장면 전환 횟수', 
                     'Unique_Subj_Count', '고유 주어 개수', '고유 주어',
                     '수정된 장면 전환 횟수', '수정된 고유 주어 개수', '수정된 서술어 개수', '수정된 Partition',
                     '모호_여부', '모호_이유', '검수_자', '검수_완료']
        
        try:
            self.train_df[save_cols].to_csv(TRAIN_CSV, index=False, encoding='cp949')
            return True
        except PermissionError:
            messagebox.showerror(
                "저장 실패", 
                "train_검토_최종_완료_수정본.csv 파일이 다른 프로그램에 의해 열려 있어 저장할 수 없습니다!\n"
                "엑셀을 닫고 다시 시도해 주세요."
            )
            return False

    def prev_sample(self):
        if self.current_idx > self.start_idx:
            self.current_idx -= 1
            self.load_sample()
            
    def next_sample(self):
        if self.save_current_corrections():
            if self.current_idx < self.end_idx:
                self.current_idx += 1
                self.load_sample()
            else:
                messagebox.showinfo("완료", f"{self.reviewer_name} 님의 검수 파트({self.start_idx}~{self.end_idx})가 모두 완료되었습니다!")

if __name__ == "__main__":
    root = tk.Tk()
    app = GUISyntaxInspectorApp(root)
    root.mainloop()
