# 서울대 AI 챌린지 전체 파이프라인 소스 코드 카탈로그 (ALL_PIPELINE_SOURCE_CODES)

본 문서는 서울대학교 AI 챌린지 프로젝트의 **전체 파이프라인 코드 집합**입니다. 아키텍처 다이어그램 생성 사이트(Mermaid, Draw.io, PlantUML 등)나 시각화용 LLM에 복사하여 입력할 수 있도록 **이미지 전처리 ➡️ 자연어 구문 분석 ➡️ QLoRA 모델 학습 ➡️ 우도 K4 TTA 최종 추론**에 이르는 모든 단계의 실전 소스 코드를 단일 파일에 순서대로 통합하여 수록했습니다.

---

## 1. 이미지 전처리 및 장면 전환 (CLIP) 단계 코드
* **위치**: `eda/clip_labeling_model.py` (의미론적 장면 분할 및 Z-Score 매핑 추출 스크립트)

```python
import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import QuantileTransformer

# 1. CLIP ViT-B/32 모델 로드 (오프라인 환경 구동)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device)
model.eval()

def calculate_clip_distances(image_paths):
    """
    4장의 이미지 간의 6쌍의 코사인 거리를 계산합니다.
    """
    embeddings = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
        # L2 정규화
        feat = feat / feat.norm(dim=-1, keepdim=True)
        embeddings.append(feat.cpu().numpy()[0])
    
    # 6쌍의 pairwise 코사인 거리 산출 (1 - CosSim)
    dist_12 = 1.0 - np.dot(embeddings[0], embeddings[1])
    dist_13 = 1.0 - np.dot(embeddings[0], embeddings[2])
    dist_14 = 1.0 - np.dot(embeddings[0], embeddings[3])
    dist_23 = 1.0 - np.dot(embeddings[1], embeddings[2])
    dist_24 = 1.0 - np.dot(embeddings[1], embeddings[3])
    dist_34 = 1.0 - np.dot(embeddings[2], embeddings[3])
    
    return [dist_12, dist_13, dist_14, dist_23, dist_24, dist_34]

def map_similar_pairs_to_cuts(distances, threshold=0.20):
    """
    [안엄격(Loose Cut) 장면 전환 판정 알고리즘]
    유사쌍(CLIP < 0.20)의 개수에 따라 최종 장면 전환 횟수를 판정합니다.
    """
    similar_pairs = sum([1 for d in distances if d < threshold])
    
    if similar_pairs >= 5:
        return 0  # 동일 씬 (장면 전환 0회)
    elif 2 <= similar_pairs <= 4:
        return 1  # 장면 전환 1회
    elif similar_pairs == 1:
        return 2  # 장면 전환 2회
    else:
        return 3  # 전원 다른 장면 (장면 전환 3회)
```

---

## 2. 자연어 구문 분석 및 모호성 모델링 (SpaCy & Flag Detector)
* **위치**: `src/features/flag_detector.py` (문장 구조 분류 및 `ai_score` 산출 코드)

```python
import re
import spacy
from spacy.cli import download

class OrthogonalFlagDetector:
    def __init__(self):
        # SpaCy 영어 모델 로드
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # 오탐율을 최소화하기 위해 경계(\b) 및 굴절어 형태를 정교화한 정규식 사전
        self.patterns = {
            # N1. 카메라/편집 담화
            "N1_camera": re.compile(
                r"\b(camera|scene|zoom(s|ed|ing)?|pan(s|ned|ning)?|shot(s)?|close-up(s)?|cuts\s+to|transition(s|ed|ing)?|fade(s|ed|ing)?|screen|view\s+shift(s)?)\b", 
                re.IGNORECASE
            ),
            # N2. 상적 국면 전이
            "N2_phase": re.compile(
                r"\b(begin|began|starts?|started|continues?|continued|finish(es|ed|ing)?|stops?|stopped|resumes?|resumed|end\s+up|proceeds?\s+to)\b", 
                re.IGNORECASE
            ),
            # N3. 스크립트/절차 지식 (주요 행위 동사 목록)
            "N3_script": re.compile(
                r"\b(bake|mix|pour|chop|fry|serve|stir|knead|slice|adjust|secure|install|remove|assemble|disassemble|insert|attach|detach)\b", 
                re.IGNORECASE
            ),
            # N4. 지시 표현 진행 (부정관사 도입 후 대명사 구정보로의 전환을 문맥 선후로 탐지)
            "N4_referential": re.compile(
                r"\b(a|an)\s+(man|woman|boy|girl|person|player|child|dog|cat|group|gymnast|skater|rider|athlete|fighter|opponent)\b.*\b(he|she|they|his|her|their|himself|herself)\b", 
                re.IGNORECASE | re.DOTALL
            ),
            # N5. 외형/상태 변화 앵커 (결과상태 매핑)
            "N5_state_change": re.compile(
                r"\b(transitions?\s+from|changes?\s+(into|from|to)|switches?\s+to|now\s+wearing|different\s+(outfit|shirt|jacket|clothes))\b", 
                re.IGNORECASE
            ),
            # N6. 반복/순환 동작 (역단서 - 순서 매핑 제한용)
            "N6_iterative": re.compile(
                r"\b(again|repeatedly|multiple\s+times|several\s+times|once\s+more|back\s+and\s+forth|over\s+and\s+over)\b", 
                re.IGNORECASE
            ),
            # N7. 서수 열거
            "N7_ordinal": re.compile(
                r"\b(first|initially|at\s+first|secondly|thirdly|lastly|eventually|in\s+the\s+end|ultimately)\b", 
                re.IGNORECASE
            )
        }

    def classify_syntax_spacy(self, sentence):
        """
        SpaCy의 의존 구문 트리(Dependency Parsing Tree)를 기반으로
        문장을 Type-1(단일 절), Type-2(복합 종속), Type-3(대등 병렬)로 1차 분류합니다.
        """
        if not isinstance(sentence, str):
            return "Type-1"
        
        doc = self.nlp(sentence)
        has_subordinate_clause = False # advcl (부사절 종속) 유무
        has_parallel_clause = False    # conj (등위절 결합) 유무
        
        for token in doc:
            if token.dep_ in {"advcl", "ccomp"}:
                has_subordinate_clause = True
            if token.dep_ == "conj" and token.pos_ in {"VERB", "AUX"}:
                has_parallel_clause = True
                
        if not has_subordinate_clause and not has_parallel_clause:
            return "Type-1"
        elif has_subordinate_clause:
            return "Type-2"
        else:
            return "Type-3"

    def detect_flags(self, sentence):
        """
        문장에 대해 7가지 직교 플래그의 이진 벡터를 리턴합니다.
        """
        if not isinstance(sentence, str):
            return {k: 0 for k in self.patterns.keys()}
        flags = {}
        for flag_name, regex in self.patterns.items():
            flags[flag_name] = 1 if regex.search(sentence) else 0
        return flags

    def calculate_ai_score(self, partition, flags):
        """
        통사적 구조(Base Score)와 의미론적 플래그 가감점 조합으로 모호성 점수를 산출합니다.
        """
        if partition == "Type-1":
            base_score = 0.80
        elif partition == "Type-2":
            base_score = 0.40
        else:
            base_score = 0.50
            
        mod = 0.0
        if flags.get("N6_iterative", 0) == 1:
            mod += 0.30
        if flags.get("N5_state_change", 0) == 1:
            mod -= 0.30
        if flags.get("N1_camera", 0) == 1:
            mod -= 0.20
        if flags.get("N7_ordinal", 0) == 1:
            mod -= 0.20
        if flags.get("N2_phase", 0) == 1:
            mod -= 0.10
            
        final_score = max(0.0, min(1.1, base_score + mod))
        return round(final_score, 2)
```

---

## 3. 최종 모델 학습 파이프라인 (Kaggle 4bit QLoRA & Hard Shuffling)
* **위치**: `kaggle/FINAL_8B_v2.py` (Kaggle T4 GPU 8B 모델 학습 소스 코드)

```python
# ================================================================================
# ★★★ FINAL_8B_v2 — 최종본 (2026-07-23) ★★★
# ================================================================================
# 설정: 8B + v5_reorder + 타깃증강(EDA) + 무작위셔플 + max_pixels 224 + 우도K4 추론
# ================================================================================

# ---- 1. 설치 (Kaggle에 없는 것만) --------------------------------------------
import subprocess, sys
def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs])
pip("transformers==5.13.0", "peft", "bitsandbytes", "accelerate", "qwen-vl-utils")

import os, ast, json, glob, random, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # OOM 단편화 완화
import pandas as pd, torch
from tqdm.auto import tqdm

# ---- 2. 데이터 경로 자동 탐색 -------------------------------------------------
DATA_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train.csv" in files and "test.csv" in files and "train" in dirs:
        DATA_DIR = root
        break
assert DATA_DIR, "❌ 대회 데이터를 못 찾음."

def find_csv(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None

AUG_WEIGHTS = find_csv("aug_weights_exp16_half.csv") or find_csv("aug_weights_exp16.csv")
CLIP_FEATS  = find_csv("snu_clip_features.csv")
HOLDOUT     = find_csv("holdout_300.csv")

# ---- 3. 설정 ------------------------------------------------------------------
CFG = dict(
    model_id="Qwen/Qwen3-VL-8B-Instruct",
    prompt_v5=("Look at the 4 images above labeled Image 1 to Image 4. Determine the "
               "correct chronological order of these images to match the sentence below.\n"
               'Sentence: "{s}"\nProvide the answer ONLY as a Python list of integers. '
               "Example: [1, 2, 3, 4]"),
    aug_mult=1, hard_shuffle=False, lr=1e-4, lora_r=16, lora_alpha=32,
    lora_targets="q_proj,k_proj,v_proj,o_proj", grad_accum=16,
    max_pixels=224*224, warmup_ratio=0.03, seed=42,
    max_steps=0,
    out="/kaggle/working/adapter", ckpt="/kaggle/working/ckpt",
    save_every=50, max_seconds=11.3*3600,
)
os.makedirs(CFG["ckpt"], exist_ok=True)
random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
rng = random.Random(CFG["seed"])

# ---- 2.5 세션 간 재개 자동화 -------------------------------------------------
import shutil
def restore_ckpt_from_input():
    if os.path.exists(os.path.join(CFG["ckpt"], "adapter_model.safetensors")):
        return "working"
    for meta in glob.glob("/kaggle/input/**/meta.json", recursive=True):
        d = os.path.dirname(meta)
        if os.path.exists(os.path.join(d, "adapter_model.safetensors")):
            for f in os.listdir(d):
                shutil.copy(os.path.join(d, f), os.path.join(CFG["ckpt"], f))
            step = json.load(open(meta)).get("step", 0)
            print(f"♻️ 이전 세션 체크포인트 복원: {d} (step {step})")
            return "input"
    return None
restore_ckpt_from_input()

# ---- 4. 데이터 준비 -----------------------------------------------------------
def chrono(ans):
    c=[0]*4
    for i,p in enumerate(ans): c[p-1]=i+1
    return c

PAIR_COLS={(1,2):"dist_12",(1,3):"dist_13",(1,4):"dist_14",(2,3):"dist_23",(2,4):"dist_24",(3,4):"dist_34"}
def load_pairs(path):
    if not path: return {}
    df=pd.read_csv(path); out={}
    for r in df.itertuples():
        out[r.Id]=[p for p,c in PAIR_COLS.items() if getattr(r,c)<0.20]
    return out

def hard_perm(seen, pairs, files, tfiles):
    best,bs=None,-1
    for _ in range(16):
        cand=list(range(4)); rng.shuffle(cand)
        if tuple(cand) in seen or [files[j] for j in cand]==tfiles: continue
        pos={o:s for s,o in enumerate(cand)}
        sc=sum(1 for a,b in pairs if pos[a-1]>pos[b-1])*10+sum(abs(pos[i]-i) for i in range(4))
        if sc>bs: best,bs=cand,sc
    return best

train_df=pd.read_csv(os.path.join(DATA_DIR,"train.csv"))
if HOLDOUT:
    hold=set(pd.read_csv(HOLDOUT)["Id"])
    train_df=train_df[~train_df["Id"].isin(hold)].reset_index(drop=True)

augw={}
if AUG_WEIGHTS:
    w=pd.read_csv(AUG_WEIGHTS); augw=dict(zip(w["Id"],w["aug_mult"].astype(int)))
pairs=load_pairs(CLIP_FEATS)

items=[]
for _,row in train_df.iterrows():
    mult=augw.get(row["Id"],CFG["aug_mult"])
    files=[row["Input_1"],row["Input_2"],row["Input_3"],row["Input_4"]]
    ans=ast.literal_eval(row["Answer"]); ch=chrono(ans)
    tfiles=[files[n-1] for n in ch]; sp=pairs.get(row["Id"],[])
    seen=set()
    for v in range(mult):
        if v==0: perm=list(range(4))
        else:
            perm=hard_perm(seen,sp,files,tfiles) if CFG["hard_shuffle"] else None
            if perm is None:
                perm=list(range(4))
                for _ in range(10):
                    rng.shuffle(perm)
                    if tuple(perm) not in seen: break
        seen.add(tuple(perm))
        shown=[files[j] for j in perm]
        target=[shown.index(f)+1 for f in tfiles]
        items.append(dict(id=row["Id"], sentence=row["Sentence"],
                          paths=[os.path.join(DATA_DIR,"train",row["Id"],f) for f in shown],
                          target=str(target)))
rng.shuffle(items)
print(f"✅ 학습 항목 {len(items)}개")

# ---- 5. 모델 로드 (4bit QLoRA) ------------------------------------------------
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from qwen_vl_utils import process_vision_info

quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
n_gpu = torch.cuda.device_count()
max_mem = {i: "14GiB" for i in range(n_gpu)}

model=AutoModelForImageTextToText.from_pretrained(CFG["model_id"],dtype=torch.bfloat16,
                                                  device_map="auto",max_memory=max_mem,
                                                  quantization_config=quant)
proc=AutoProcessor.from_pretrained(CFG["model_id"],max_pixels=CFG["max_pixels"])
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.config.use_cache=False

resume=0; meta=os.path.join(CFG["ckpt"],"meta.json")
if os.path.exists(os.path.join(CFG["ckpt"],"adapter_model.safetensors")):
    model=PeftModel.from_pretrained(model,CFG["ckpt"],is_trainable=True)
    resume=json.load(open(meta))["step"]
else:
    model=get_peft_model(model,LoraConfig(r=CFG["lora_r"],lora_alpha=CFG["lora_alpha"],
        lora_dropout=0.05,target_modules=CFG["lora_targets"].split(","),bias="none",task_type="CAUSAL_LM"))
model.train()

total=(len(items))//CFG["grad_accum"]
trainable=[p for p in model.parameters() if p.requires_grad]
opt=torch.optim.AdamW(trainable,lr=CFG["lr"],weight_decay=0.01)
sched=get_cosine_schedule_with_warmup(opt,int(total*CFG["warmup_ratio"]),total)
optpt=os.path.join(CFG["ckpt"],"optim.pt")
if resume and os.path.exists(optpt):
    st=torch.load(optpt,map_location="cpu"); opt.load_state_dict(st["o"]); sched.load_state_dict(st["s"])
dev=next(model.parameters()).device

def encode(it):
    content=[]
    for i,p in enumerate(it["paths"]):
        content+=[{"type":"image","image":p},{"type":"text","text":f"\nImage {i+1}\n"}]
    content.append({"type":"text","text":CFG["prompt_v5"].format(s=it["sentence"])})
    msgs=[{"role":"user","content":content}]
    pt=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    fm=msgs+[{"role":"assistant","content":[{"type":"text","text":it["target"]}]}]
    ft=proc.apply_chat_template(fm,tokenize=False)
    img,vid=process_vision_info(msgs)
    full=proc(text=[ft],images=img,videos=vid,padding=True,return_tensors="pt")
    pr=proc(text=[pt],images=img,videos=vid,padding=True,return_tensors="pt")
    lab=full.input_ids.clone(); lab[:,:pr.input_ids.shape[1]]=-100; full["labels"]=lab
    return full.to(dev)

def save_ckpt(step):
    model.save_pretrained(CFG["ckpt"])
    torch.save({"o":opt.state_dict(),"s":sched.state_dict()},optpt)
    json.dump({"step":step},open(meta,"w"))

# ---- 6. 학습 루프 -------------------------------------------------------------
t0=time.time(); step=resume; micro=0; lacc=0.0; skip=0
start=resume*CFG["grad_accum"]
pbar=tqdm(total=len(items),initial=start)
try:
    for idx,it in enumerate(items):
        if idx<start: continue
        try:
            loss=model(**encode(it)).loss/CFG["grad_accum"]; loss.backward()
        except torch.cuda.OutOfMemoryError:
            skip+=1; opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); continue
        lacc+=loss.item(); micro+=1; pbar.update(1)
        if micro%CFG["grad_accum"]==0:
            torch.nn.utils.clip_grad_norm_(trainable,1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step+=1
            if step%10==0: pbar.set_postfix(loss=round(lacc,4),step=step);
            lacc=0.0
            if step%CFG["save_every"]==0: save_ckpt(step)
            if time.time()-t0>CFG["max_seconds"]:
                save_ckpt(step); raise KeyboardInterrupt
except KeyboardInterrupt: pass
model.save_pretrained(CFG["out"]); save_ckpt(step)
```

---

## 4. 최종 모델 추론 파이프라인 (KV-Cache 복제 & 우도 K4 TTA)
* **위치**: `kaggle/INFER_ONLY_K4.py` (Kaggle T4 GPU 초고속 순열 채점 소스 코드)

```python
# ================================================================================
# ★★★ 추론 전용 — 우도 K4 (학습된 어댑터로 submission 생성) ★★★
# ================================================================================

import subprocess, sys
def pip(*p): subprocess.run([sys.executable,"-m","pip","install","-q","-U",*p])
pip("transformers==5.13.0","peft","bitsandbytes","accelerate","qwen-vl-utils")

import os, ast, glob
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import pandas as pd, torch
from tqdm.auto import tqdm
from itertools import permutations

# ---- 설정 --------------------------------------------------------------------
MODEL_ID   = "Qwen/Qwen3-VL-8B-Instruct"
MAX_PIXELS = 384*512
USE_LIKELIHOOD = True
CHUNK      = 6
SAVE_EVERY = 50
PROMPT_V5 = ("Look at the 4 images above labeled Image 1 to Image 4. Determine the "
             "correct chronological order of these images to match the sentence below.\n"
             'Sentence: "{s}"\nProvide the answer ONLY as a Python list of integers. '
             "Example: [1, 2, 3, 4]")

# ---- 데이터·어댑터 자동 탐색 --------------------------------------------------
DATA_DIR=None
for r,d,f in os.walk("/kaggle/input"):
    if "test.csv" in f and "test" in d: DATA_DIR=r; break
assert DATA_DIR, "❌ 대회 데이터 없음"

ADAPTER=None
for p in glob.glob("/kaggle/input/**/adapter_model.safetensors", recursive=True):
    ADAPTER=os.path.dirname(p); break
if not ADAPTER:
    for p in glob.glob("/kaggle/working/**/adapter_model.safetensors", recursive=True):
        ADAPTER=os.path.dirname(p); break
assert ADAPTER, "❌ 어댑터 없음"

# ---- 모델 로드 ----------------------------------------------------------------
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)
n_gpu=torch.cuda.device_count(); max_mem={i:"14GiB" for i in range(n_gpu)}
model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.float16,
        device_map="auto",max_memory=max_mem,quantization_config=quant)
model=PeftModel.from_pretrained(model,ADAPTER)
model.eval()
proc=AutoProcessor.from_pretrained(MODEL_ID,max_pixels=MAX_PIXELS)
dev=next(model.parameters()).device

inner=model
for _ in range(5):
    if hasattr(inner,"rope_deltas"): break
    inner=getattr(inner,"model",None) or getattr(inner,"base_model",None)

test=pd.read_csv(os.path.join(DATA_DIR,"test.csv"))
CANDS=[list(p) for p in permutations([1,2,3,4])]
PERMS=[[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]] if USE_LIKELIHOOD else [[0,1,2,3]]
eos=proc.tokenizer("<|im_end|>",add_special_tokens=False)["input_ids"]

def tgt_str(answer, perm):
    c=[0]*4
    for i,pos in enumerate(answer): c[pos-1]=i+1
    files=[1,2,3,4]; tf=[files[n-1] for n in c]; shown=[files[j] for j in perm]
    return str([shown.index(f)+1 for f in tf])

OUT_PATH="/kaggle/working/submission.csv"

# ---- 우도 채점 (KV-Cache 복제 & 청크 처리) -----------------------------------
import copy
from transformers.cache_utils import DynamicCache

def _clone_cache(cache):
    try:
        return copy.deepcopy(cache)
    except Exception:
        new=DynamicCache()
        if hasattr(cache,"layers") and cache.layers:
            for i,lyr in enumerate(cache.layers):
                new.update(lyr.keys.clone(),lyr.values.clone(),i)
        else:
            for i,(k,v) in enumerate(zip(cache.key_cache,cache.value_cache)):
                new.update(k.clone(),v.clone(),i)
        return new

def score_perm(enc, plen, amat, L, ch):
    with torch.no_grad():
        o1=model(**enc,use_cache=True); base=o1.past_key_values; rd=inner.rope_deltas.item()
        flp=torch.log_softmax(o1.logits[:,-1,:].float(),-1)
        tots=[]
        for cs in range(0, amat.shape[0], ch):
            amc=amat[cs:cs+ch]; b=amc.shape[0]
            pkv=_clone_cache(base); pkv.batch_repeat_interleave(b)
            pos=(torch.arange(plen,plen+L,device=dev)+rd).view(1,1,-1).expand(3,b,-1).contiguous()
            o2=model(input_ids=amc,position_ids=pos,past_key_values=pkv,
                     attention_mask=torch.ones(b,plen+L,device=dev,dtype=torch.long),use_cache=True)
            lp0=flp[0,amc[:,0]]; rest=torch.log_softmax(o2.logits[:,:-1].float(),-1)
            tc=lp0+rest[torch.arange(b)[:,None],torch.arange(L-1)[None,:],amc[:,1:]].sum(1)
            tots.append(tc.detach().cpu()); del pkv,o2,rest
        del o1,base,flp
    return torch.cat(tots)

# ---- 재시작 지원 -------------------------------------------------------------
recs=[]; done=set()
if os.path.exists(OUT_PATH):
    prev=pd.read_csv(OUT_PATH)
    recs=prev.to_dict("records"); done=set(prev["Id"].tolist())

for ridx,(_,row) in enumerate(tqdm(test.iterrows(),total=len(test))):
    if row["Id"] in done: continue
    files=[row["Input_1"],row["Input_2"],row["Input_3"],row["Input_4"]]
    score={tuple(a):0.0 for a in CANDS}
    for perm in PERMS:
        shown=[files[j] for j in perm]
        content=[]
        for i,f in enumerate(shown):
            content+=[{"type":"image","image":os.path.join(DATA_DIR,"test",row["Id"],f)},
                      {"type":"text","text":f"\nImage {i+1}\n"}]
        content.append({"type":"text","text":PROMPT_V5.format(s=row["Sentence"])})
        m=[{"role":"user","content":content}]
        pt=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
        ii,vi=process_vision_info(m); enc=proc(text=[pt],images=ii,videos=vi,return_tensors="pt").to(dev)
        plen=enc["input_ids"].shape[1]
        atok=[proc.tokenizer(tgt_str(a,perm),add_special_tokens=False)["input_ids"]+eos for a in CANDS]
        L=len(atok[0]); amat=torch.tensor(atok,device=dev)
        ch=CHUNK
        while True:
            try:
                tot=score_perm(enc,plen,amat,L,ch); break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if ch<=2: raise
                ch=max(2,ch//2)
        for a,s in zip(CANDS,tot.tolist()): score[tuple(a)]+=s
    best=max(CANDS,key=lambda a:score[tuple(a)])
    recs.append({"Id":row["Id"],"Answer":str(best)})
    if len(recs)%SAVE_EVERY==0:
        pd.DataFrame(recs).to_csv(OUT_PATH,index=False)

out=pd.DataFrame(recs)
out.to_csv(OUT_PATH,index=False)
```
