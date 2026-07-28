import os
import sys
import ast
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, filedialog

# Force stdout to UTF-8 to prevent cp949 encode errors in Windows
sys.stdout.reconfigure(encoding='utf-8')

# =========================================================================
# [전역 예외 처리 설정]
# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import traceback

def global_exception_handler(exctype, value, tb):
    log_path = os.path.join(SCRIPT_DIR, "error_log.txt")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            traceback.print_exception(exctype, value, tb, file=f)
    except:
        pass
    
    err_msg = "".join(traceback.format_exception(exctype, value, tb))
    print(err_msg)  # 콘솔에도 출력
    
    try:
        messagebox.showerror("시스템 오류", f"프로그램 실행 중 예외가 발생했습니다:\n{value}\n\n상세 정보가 {log_path} 에 저장되었습니다.")
    except:
        pass

sys.excepthook = global_exception_handler

# =========================================================================
# [경로 설정 - 단독 및 부모 폴더 실행을 모두 지원하는 하이브리드 경로 매핑]
# =========================================================================
CONFIG_PATH = os.path.join(SCRIPT_DIR, "image_path_config.txt")

# 1. 원본 CSV 찾기 (부모 폴더 혹은 현재 폴더 탐색)
possible_sources = [
    os.path.join(os.path.dirname(SCRIPT_DIR), "train_검토_최종_완료.csv"),
    os.path.join(SCRIPT_DIR, "train_검토_최종_완료.csv")
]
SOURCE_CSV = ""
for p in possible_sources:
    if os.path.exists(p):
        SOURCE_CSV = p
        break
if not SOURCE_CSV:
    SOURCE_CSV = os.path.join(SCRIPT_DIR, "train_검토_최종_완료.csv")

# 2. 수정본 CSV 및 피처 CSV 경로 매핑 (찾아낸 원본 CSV 폴더를 기준 삼아 매칭)
CSV_PARENT_DIR = os.path.dirname(SOURCE_CSV)
TRAIN_CSV = os.path.join(CSV_PARENT_DIR, "train_검토_최종_완료_수정본.csv")

possible_features = [
    os.path.join(CSV_PARENT_DIR, "snu_clip_features.csv"),
    os.path.join(SCRIPT_DIR, "snu_clip_features.csv")
]
FEATURES_CSV = ""
for p in possible_features:
    if os.path.exists(p):
        FEATURES_CSV = p
        break
if not FEATURES_CSV:
    FEATURES_CSV = os.path.join(CSV_PARENT_DIR, "snu_clip_features.csv")

# 이미지 폴더 찾기 및 사용자 지정 로직
IMAGE_DIR = ""
candidate_dirs = [
    os.path.join(SCRIPT_DIR, "snuaichallenge_data", "train"),
    os.path.join(os.path.dirname(SCRIPT_DIR), "snuaichallenge_data", "train"),
    os.path.join(SCRIPT_DIR, "train"),
]

# 1. 기존에 선택했던 캐시 경로가 있는지 확인
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cached_path = f.read().strip()
        if os.path.exists(cached_path):
            IMAGE_DIR = cached_path

# 2. 캐시가 없으면 기본 후보 경로 탐색
if not IMAGE_DIR:
    for cdir in candidate_dirs:
        if os.path.exists(cdir):
            IMAGE_DIR = cdir
            break

# 3. 그래도 없으면 사용자에게 대화상자로 선택 요청
if not IMAGE_DIR:
    root_temp = tk.Tk()
    root_temp.withdraw()  # 임시 루트 창 숨기기
    messagebox.showinfo(
        "이미지 경로 설정",
        "대회 이미지(train) 폴더를 찾을 수 없습니다.\n"
        "다음 창에서 다운로드받으신 이미지들이 들어있는 'train' 폴더(예: 00GGp0 등의 하위 폴더가 들어있는 폴더)를 선택해 주세요."
    )
    selected_dir = filedialog.askdirectory(title="대회 이미지(train) 폴더를 선택해 주세요")
    root_temp.destroy()
    
    if selected_dir:
        IMAGE_DIR = os.path.normpath(selected_dir)
        # 선택 경로 저장
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(IMAGE_DIR)
    else:
        # 선택 안 하고 취소 시 프로그램 종료
        sys.exit()

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
            
        # 신규 수정 컬럼 및 모호성 체크용 컬럼 초기화 (타입 오류 방지를 위해 명시적 기본값 지정 및 강제 형변환)
        for col in ['수정된 장면 전환 횟수', '수정된 고유 주어 개수', '수정된 서술어 개수']:
            if col not in self.merged_df.columns:
                self.merged_df[col] = np.nan
            if col not in self.train_df.columns:
                self.train_df[col] = np.nan
            self.merged_df[col] = self.merged_df[col].astype(float)
            self.train_df[col] = self.train_df[col].astype(float)
                
        for col in ['수정된 Partition', '모호_이유', '검수_자']:
            if col not in self.merged_df.columns:
                self.merged_df[col] = ""
            if col not in self.train_df.columns:
                self.train_df[col] = ""
            self.merged_df[col] = self.merged_df[col].fillna("").astype(str)
            self.train_df[col] = self.train_df[col].fillna("").astype(str)
                
        for col in ['모호_여부', '검수_완료']:
            if col not in self.merged_df.columns:
                self.merged_df[col] = False
            if col not in self.train_df.columns:
                self.train_df[col] = False
            self.merged_df[col] = self.merged_df[col].fillna(False).astype(bool)
            self.train_df[col] = self.train_df[col].fillna(False).astype(bool)
            
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
        
        self.translation_label = tk.Label(
            self.info_frame, text="", font=("Malgun Gothic", 11, "italic"),
            wraplength=1300, justify="center", fg="#4A5568", bg="#F5F7FA"
        )
        self.translation_label.pack(pady=3)
        
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
        try:
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
            
            raw_part = row['수정된 Partition']
            show_partition = row['Partition'] if pd.isna(raw_part) or str(raw_part).strip() == "" else raw_part
            self.id_label.config(text=f"Sample ID: {sample_id} | Correct Order (Answer): {ans} | 구조: {show_partition}")
            self.sentence_label.config(text=sentence)
            
            # 실시간 비동기 구글 번역 (무료 API)
            self.translation_label.config(text="🌐 한글 번역 로드 중...")
            import threading
            
            def bg_translate(eng_text):
                import urllib.request
                import urllib.parse
                import json
                try:
                    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=ko&dt=t&q=" + urllib.parse.quote(eng_text)
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=2.0) as res:
                        data = json.loads(res.read().decode('utf-8'))
                        translated = "".join([s[0] for s in data[0] if s[0]])
                        self.root.after(0, lambda: self.translation_label.config(text=f"↳ {translated}"))
                except Exception as e:
                    self.root.after(0, lambda: self.translation_label.config(text="↳ [한글 번역 로드 실패]"))
                    
            threading.Thread(target=bg_translate, args=(sentence,), daemon=True).start()
            
            def safe_int_str(val, fallback_val):
                if pd.isna(val) or str(val).strip() == "":
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
        except Exception as e:
            import traceback
            log_path = os.path.join(SCRIPT_DIR, "error_log.txt")
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    traceback.print_exc(file=f)
            except:
                pass
            messagebox.showerror("로드 오류", f"샘플을 로드하는 중 에러가 발생했습니다:\n{e}\n\n상세 로그가 {log_path} 에 저장되었습니다.")

    def save_current_corrections(self):
        import traceback
        try:
            # 1. 입력 값 추출 (소수점 형태 및 빈 칸 입력 유연화)
            row = self.merged_df.iloc[self.current_idx]
            
            cuts_text = self.entry_cuts.get().strip()
            new_cuts = int(float(cuts_text)) if cuts_text != "" else int(float(row['장면 전환 횟수']))
            
            subjs_text = self.entry_subjs.get().strip()
            new_subjs = int(float(subjs_text)) if subjs_text != "" else int(float(row['고유 주어 개수']))
            
            preds_text = self.entry_preds.get().strip()
            new_preds = int(float(preds_text)) if preds_text != "" else int(float(row['서술어 개수']))
        except ValueError:
            messagebox.showwarning("입력 오류", "수정 항목에는 반드시 숫자(정수) 또는 빈 칸만 입력해야 합니다!")
            return False
            
        new_part = self.var_partition.get()
        new_ambig = self.var_is_ambiguous.get()
        new_reason = self.entry_reason.get().strip()
        
        try:
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
            
            self.train_df[save_cols].to_csv(TRAIN_CSV, index=False, encoding='cp949')
            return True
        except PermissionError:
            messagebox.showerror(
                "저장 실패", 
                "train_검토_최종_완료_수정본.csv 파일이 다른 프로그램에 의해 열려 있어 저장할 수 없습니다!\n"
                "엑셀을 닫고 다시 시도해 주세요."
            )
            return False
        except Exception as e:
            log_path = os.path.join(SCRIPT_DIR, "error_log.txt")
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    traceback.print_exc(file=f)
            except:
                pass
            messagebox.showerror("저장 오류 (디버깅용)", f"데이터 저장 중 에러가 발생했습니다:\n{e}\n\n에러 타입: {type(e)}\n상세 로그가 {log_path} 에 저장되었습니다.")
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
