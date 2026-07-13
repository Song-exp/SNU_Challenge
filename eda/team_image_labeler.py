import os
import re
import pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, ttk

# =========================================================================
# [설정 및 경로]
# =========================================================================
DATA_DIR = "./snuaichallenge_data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train")

# 팀 멤버별 영역 정의 (대소문자 무구분 정렬 인덱스 기준)
MEMBER_CONFIGS = {
    "병철": {
        "start_idx": 0,
        "end_idx": 3000,   # index 0 ~ 2999 (3000개)
        "file_name": "./eda/labeled_byeongcheol.csv",
        "boundary_desc": "00GGp0 ~ EquxBk"
    },
    "서현": {
        "start_idx": 3000,
        "end_idx": 6003,   # index 3000 ~ 6002 (3003개)
        "file_name": "./eda/labeled_seohyeon.csv",
        "boundary_desc": "er2p3e ~ oQI68U"
    },
    "정현": {
        "start_idx": 6003,
        "end_idx": 9535,   # index 6003 ~ 9534 (3532개)
        "file_name": "./eda/labeled_jeonghyeon.csv",
        "boundary_desc": "oqImQK ~ ZzYxAm"
    }
}

# =========================================================================
# [AI 분석 규칙]
# =========================================================================
TEMPORAL_PATTERNS = r"\b(then|before|after|followed|finally|next|first|second|third|afterwards)\b"
SIMULTANEOUS_PATTERNS = r"\b(while|as|meanwhile|during|simultaneously)\b"
MOTION_PREPS = r"\b(down|up|towards|into|across|through|over|under|onto|out|off|to|from)\b"
STATIVE_VERBS = r"\b(seen|sitting|standing|walking|looking|watching|holding|carrying|wearing|is|are|was|were|be|staying|resting|lying|floating)\b"

ACTION_VERBS = [
    "wipe", "wipes", "wiped", "wiping", "mount", "mounts", "mounted", "mounting",
    "ride", "rides", "rode", "riding", "jump", "jumps", "jumped", "jumping",
    "run", "runs", "ran", "running", "shave", "shaves", "shaved", "shaving",
    "cut", "cuts", "cutting", "play", "plays", "played", "playing",
    "hit", "hits", "hitting", "throw", "throws", "threw", "throwing",
    "kick", "kicks", "kicked", "kicking", "spin", "spins", "spun", "spinning",
    "adjust", "adjusts", "adjusted", "adjusting", "turn", "turns", "turned", "turning",
    "move", "moves", "moved", "moving", "swim", "swims", "swam", "swimming",
    "dance", "dances", "danced", "dancing", "sing", "sings", "sang", "singing",
    "talk", "talks", "talked", "talking", "walk", "walks", "walked", "walking",
    "fall", "falls", "fell", "falling"
]

def analyze_sentence(sentence):
    if not isinstance(sentence, str):
        return "Category: Not applicable, Main verb: None, Background state: No, Anomalies: Cannot determine"
    s_lower = sentence.lower()
    has_temporal = bool(re.search(TEMPORAL_PATTERNS, s_lower))
    has_simultaneous = bool(re.search(SIMULTANEOUS_PATTERNS, s_lower))
    has_motion = bool(re.search(MOTION_PREPS, s_lower))
    has_stative = bool(re.search(STATIVE_VERBS, s_lower))
    
    words = re.findall(r"\b[a-zA-Z]+\b", s_lower)
    verbs = []
    for w in words:
        if w in ACTION_VERBS:
            verbs.append(w)
        elif (w.endswith("ed") or w.endswith("ing")) and w not in ["and", "during", "then", "followed"]:
            if not re.search(STATIVE_VERBS, w):
                verbs.append(w)
    verbs = list(dict.fromkeys(verbs))
    
    main_verb = verbs[0] if verbs else "None"
    if main_verb == "None":
        stative_find = re.findall(STATIVE_VERBS, s_lower)
        if stative_find:
            main_verb = stative_find[0]
            
    clause_connectors = len(re.findall(r"\b(and|but|then|when|while|as|before|after|so)\b", s_lower))
    commas = s_lower.count(",")
    is_multi_clause = (clause_connectors >= 1) or (commas >= 1) or (len(verbs) >= 2)
    
    if has_temporal:
        category = "1"
    elif has_simultaneous:
        category = "5"
    elif is_multi_clause and not has_temporal:
        category = "2"
    elif has_motion and not is_multi_clause:
        category = "3"
    elif has_stative and len(verbs) == 0:
        category = "5"
    else:
        category = "4"
        
    bg_state = "Yes" if (has_stative or has_simultaneous) else "No"
    anomaly = "None"
    if len(s_lower) < 15:
        anomaly = "Cannot determine"
    elif "  " in sentence:
        anomaly = "Typo (Double Spaces)"
        
    return f"Category: {category}, Main verb: {main_verb}, Background state: {bg_state}, Anomalies: {anomaly}"

# =========================================================================
# [Tkinter GUI 구현]
# =========================================================================
class TeamLabelerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SNU AI Challenge - 팀 공동 이미지/문장 검수 라벨러")
        self.root.geometry("1150x720")
        
        # 1. 공통 데이터 로드
        if not os.path.exists(TRAIN_CSV):
            messagebox.showerror("Error", f"train.csv 파일을 찾을 수 없습니다: {TRAIN_CSV}")
            self.root.destroy()
            return
            
        print("Loading train.csv...")
        self.full_df = pd.read_csv(TRAIN_CSV)
        # 윈도우 탐색기 정렬 방식과 동일하게 대소문자 무시(case-insensitive) 오름차순 정렬
        self.full_df = self.full_df.iloc[self.full_df['Id'].str.lower().argsort()].reset_index(drop=True)
        
        # 멤버 선택 프레임 띄우기
        self.setup_member_selection()
        
    def setup_member_selection(self):
        self.select_frame = tk.Frame(self.root, bg="#34495E")
        self.select_frame.pack(fill="both", expand=True)
        
        lbl_title = tk.Label(
            self.select_frame, text="👥 SNU AI Challenge - 라벨링 담당자를 선택해 주세요",
            font=("Malgun Gothic", 16, "bold"), fg="white", bg="#34495E"
        )
        lbl_title.pack(pady=60)
        
        btn_container = tk.Frame(self.select_frame, bg="#34495E")
        btn_container.pack()
        
        for member_name, config in MEMBER_CONFIGS.items():
            btn = tk.Button(
                btn_container, text=f"{member_name}님\n({config['boundary_desc']})",
                font=("Malgun Gothic", 13, "bold"), bg="#3498DB", fg="white",
                width=25, height=4, relief="raised", borderwidth=3,
                command=lambda name=member_name: self.start_labeler(name)
            )
            btn.pack(side="left", padx=20)
            
        lbl_guide = tk.Label(
            self.select_frame, 
            text="* 가상환경(.venv)이 켜진 로컬 윈도우 환경에서 실행해 주세요.\n* 본인의 이름을 클릭하면 자동으로 3,000여 개 범위 설정 및 AI 예비 분류가 가동됩니다.",
            font=("Malgun Gothic", 10), fg="#BDC3C7", bg="#34495E", justify="center"
        )
        lbl_guide.pack(pady=50)

    def start_labeler(self, name):
        self.member_name = name
        self.config = MEMBER_CONFIGS[name]
        self.output_csv = self.config["file_name"]
        
        # 슬라이싱
        self.target_df = self.full_df.iloc[self.config["start_idx"]:self.config["end_idx"]].copy()
        
        # UI 전환
        self.select_frame.pack_forget()
        
        # 데이터 초기화 & AI 예비 라벨링
        self.init_data()
        
        # 라벨러 메인 UI 셋업
        self.setup_main_ui()
        self.load_sample()
        
    def init_data(self):
        self.labels = {}
        self.confirmed = {}
        
        # 파일이 이미 존재하는 경우 (이어하기)
        if os.path.exists(self.output_csv):
            try:
                progress_df = pd.read_csv(self.output_csv)
                for _, row in progress_df.iterrows():
                    sample_id = str(row['Id'])
                    if not pd.isna(row.get('Labeled_Concept')):
                        self.labels[sample_id] = str(row['Labeled_Concept'])
                    if 'Confirmed' in progress_df.columns:
                        if str(row.get('Confirmed')).lower() in ['true', '1', '1.0']:
                            self.confirmed[sample_id] = True
                print(f"{self.member_name}님 파일 로드 성공: 완료 {len(self.confirmed)}개.")
            except Exception as e:
                print(f"기존 파일 로드 실패, 새로 생성합니다: {e}")
                
        # 파일이 없거나, 마킹되지 않은 행이 있다면 AI 분석값으로 자동 예비 기입
        prepopulated = 0
        for _, row in self.target_df.iterrows():
            sample_id = str(row['Id'])
            if sample_id not in self.labels:
                self.labels[sample_id] = analyze_sentence(row['Sentence'])
                prepopulated += 1
                
        if prepopulated > 0:
            print(f"AI 가이드 라벨 {prepopulated}개 자동 주입 완료.")
            self.save_to_csv()
            
        # 첫 번째 미검수 인덱스 탐색
        self.current_idx = 0
        for idx, row in self.target_df.reset_index(drop=True).iterrows():
            sample_id = str(row['Id'])
            if sample_id not in self.confirmed:
                self.current_idx = idx
                break
        self.current_idx = min(self.current_idx, len(self.target_df) - 1)

    def setup_main_ui(self):
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
            self.sentence_frame, text="", font=("Malgun Gothic", 13, "bold"),
            wraplength=1000, justify="center", fg="#2C3E50"
        )
        self.sentence_text.pack(pady=5)
        
        # 이미지 가로 정렬 프레임
        self.img_frame = tk.Frame(self.root)
        self.img_frame.pack(pady=5)
        
        self.img_labels = []
        for i in range(4):
            lbl = tk.Label(self.img_frame, borderwidth=2, relief="solid", bg="gray")
            lbl.pack(side="left", padx=10)
            self.img_labels.append(lbl)
            
        # 하단 입력 및 조작 영역
        self.input_frame = tk.Frame(self.root, pady=15)
        self.input_frame.pack(fill="x")
        
        tk.Label(
            self.input_frame, text="✍️ 이미지/문맥 검수 및 개념화 기술 (Enter 입력 시 승인 후 다음 이동):",
            font=("Malgun Gothic", 11, "bold"), fg="#2C3E50"
        ).pack(anchor="w", padx=40)
        
        self.concept_entry = tk.Entry(
            self.input_frame, font=("Malgun Gothic", 12), width=95,
            borderwidth=2, relief="groove"
        )
        self.concept_entry.pack(pady=5, padx=40, fill="x")
        self.concept_entry.focus_set()
        
        # 조작 버튼 프레임
        self.control_frame = tk.Frame(self.root, pady=5)
        self.control_frame.pack(fill="x")
        
        tk.Button(
            self.control_frame, text="◀ 이전 (Alt + Left)", bg="#BDC3C7", fg="black",
            command=self.prev_sample, font=("Malgun Gothic", 10, "bold")
        ).pack(side="left", padx=40)
        
        tk.Button(
            self.control_frame, text="다음 / Skip (Alt + Right) ▶", bg="#BDC3C7", fg="black",
            command=self.next_sample, font=("Malgun Gothic", 10, "bold")
        ).pack(side="right", padx=40)
        
        # 키보드 이벤트 바인딩
        self.concept_entry.bind("<Return>", lambda event: self.save_concept())
        self.root.bind("<Alt-Left>", lambda event: self.prev_sample())
        self.root.bind("<Alt-Right>", lambda event: self.next_sample())
        
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
        
        self.concept_entry.delete(0, tk.END)
        if sample_id in self.labels:
            self.concept_entry.insert(0, self.labels[sample_id])
        self.concept_entry.focus_set()
        
        # 진행도 라벨
        confirmed_count = len(self.confirmed)
        total_count = len(self.target_df)
        percentage = (confirmed_count / total_count) * 100
        marked_status = " [검수완료]" if sample_id in self.confirmed else " [미검수]"
        
        self.progress_label.config(
            text=f"👤 담당자: {self.member_name}님 | 검수 상황: {confirmed_count} / {total_count} ({percentage:.1f}%) | 현재 샘플: {self.current_idx + 1}번째" + marked_status
        )
        self.id_label.config(text=f"Sample ID: {sample_id} | Answer: {row.get('Answer', 'N/A')}")
        self.sentence_text.config(text=sentence)
        
        # 이미지 로딩
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
                
        self.labels[sample_id] = concept_text
        self.confirmed[sample_id] = True
        self.save_to_csv()
        
        if self.current_idx < len(self.target_df) - 1:
            self.current_idx += 1
            self.load_sample()
        else:
            messagebox.showinfo("완료", f"{self.member_name}님의 담당 분량 검수가 모두 완료되었습니다! 고생하셨습니다.")
            
    def prev_sample(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_sample()
            
    def next_sample(self):
        if self.current_idx < len(self.target_df) - 1:
            self.current_idx += 1
            self.load_sample()
            
    def save_to_csv(self):
        # target_df 기준 복제 후 맵핑 저장
        df_copy = self.target_df.copy()
        df_copy['Labeled_Concept'] = df_copy['Id'].apply(lambda x: self.labels.get(str(x), None))
        df_copy['Confirmed'] = df_copy['Id'].apply(lambda x: self.confirmed.get(str(x), False))
        
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        df_copy.to_csv(self.output_csv, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    root = tk.Tk()
    app = TeamLabelerApp(root)
    root.mainloop()
