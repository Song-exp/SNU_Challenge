# ================================================================================
# ★★★ 추론 전용 — greedy (우도 K4 미적용, 빠른 안전판) ★★★
# ================================================================================
# 용도: 학습된 8B 어댑터로 test 추론 → submission.csv. 학습 안 함.
#       INFER_ONLY_K4.py 와 모델·프롬프트·해상도 전부 동일하고 채점 방식만 다름.
#         · K4  : 24후보 우도 × 4배치 = 96 forward/행 → ~5h,  holdout +4.76%p
#         · 이거: 1회 생성            =  1 forward/행 → ~40-70분
#
# 언제 쓰나: 마감 전 점수 먼저 확보 / K4 세션이 터졌을 때 대체 제출 /
#           어댑터가 제대로 학습됐는지 빠른 확인
#
# 준비 (Add Input):
#   1) 대회 데이터 (train.csv/test.csv 있는 것)
#   2) 학습 어댑터 (adapter_model.safetensors 든 폴더)
#   GPU T4×2, Internet On, Commit 실행
# ================================================================================
import subprocess, sys
def pip(*p): subprocess.run([sys.executable,"-m","pip","install","-q","-U",*p])
pip("transformers==5.13.0","peft","bitsandbytes","accelerate","qwen-vl-utils")

import os, ast, glob
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import pandas as pd, torch
from tqdm.auto import tqdm

# ---- 설정 --------------------------------------------------------------------
MODEL_ID   = "Qwen/Qwen3-VL-8B-Instruct"
MAX_PIXELS = 384*512           # ★학습값과 반드시 일치 (kaggle_8b_train.py)
SAVE_EVERY = 50                # ★중간 저장 간격(행). 죽어도 보존+재시작 시 스킵.
SHOW_FIRST = 3                 # 처음 N행의 모델 원문 출력 (형식 눈으로 확인)
PROMPT_V5 = ("Look at the 4 images above labeled Image 1 to Image 4. Determine the "
             "correct chronological order of these images to match the sentence below.\n"
             'Sentence: "{s}"\nProvide the answer ONLY as a Python list of integers. '
             "Example: [1, 2, 3, 4]")

# ---- 데이터·어댑터 자동 탐색 --------------------------------------------------
DATA_DIR=None
for r,d,f in os.walk("/kaggle/input"):
    if "test.csv" in f and "test" in d: DATA_DIR=r; break
assert DATA_DIR, "❌ 대회 데이터 없음 — Add Input"
print("✅ 데이터:", DATA_DIR)

ADAPTER=None
for p in glob.glob("/kaggle/input/**/adapter_model.safetensors", recursive=True):
    ADAPTER=os.path.dirname(p); break
if not ADAPTER:
    for p in glob.glob("/kaggle/working/**/adapter_model.safetensors", recursive=True):
        ADAPTER=os.path.dirname(p); break
assert ADAPTER, "❌ 어댑터 없음 — 학습 Output(ckpt)을 Add Input 하세요"
print("✅ 어댑터:", ADAPTER)

# ---- 모델 로드 (학습과 동일: 4bit nf4 + fp16) ---------------------------------
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True)  # T4=fp16(bf16 텐서코어 없음)
n_gpu=torch.cuda.device_count(); max_mem={i:"14GiB" for i in range(n_gpu)}
model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.float16,
        device_map="auto",max_memory=max_mem,quantization_config=quant)
model=PeftModel.from_pretrained(model,ADAPTER)
model.eval()
proc=AutoProcessor.from_pretrained(MODEL_ID,max_pixels=MAX_PIXELS)
dev=next(model.parameters()).device

# ---- 어댑터 반영 확인 (lora_B는 학습 전 0 → 전부 0이면 베이스 모델로 도는 것) --
_nb=0; _mx=0.0
for _n,_p in model.named_parameters():
    if "lora_B" in _n: _nb+=1; _mx=max(_mx,_p.abs().max().item())
print(f"✅ lora_B {_nb}개, 최대 절대값 {_mx:.6f} → "
      f"{'어댑터 반영됨' if _mx>1e-6 else '❌ 미반영!'}")
assert _mx>1e-6, "어댑터가 로드되지 않았습니다 (lora_B 전부 0) — 경로 확인"

test=pd.read_csv(os.path.join(DATA_DIR,"test.csv"))
OUT_PATH="/kaggle/working/submission.csv"

def parse_greedy(txt):
    """모델 출력 '[3, 1, 2, 4]'(Image 슬롯의 시간순) → 제출 형식. 실패 시 (None)."""
    s=txt.rfind("["); e=txt.find("]",s)
    if s<0 or e<0: return None
    try:
        r=ast.literal_eval(txt[s:e+1])
    except Exception:
        return None
    if not (isinstance(r,list) and sorted(r)==[1,2,3,4]): return None
    sub=[0]*4
    for i,n in enumerate(r): sub[n-1]=i+1
    return sub

# ---- 재시작: 이미 처리한 Id는 스킵 (죽어도 이어감) ----------------------------
recs=[]; done=set()
if os.path.exists(OUT_PATH):
    try:
        prev=pd.read_csv(OUT_PATH)
        recs=prev.to_dict("records"); done=set(prev["Id"].tolist())
        print(f"↻ 재시작: {len(done)}행 이미 완료 → 스킵")
    except Exception:
        print("⚠️ 기존 파일 로딩 실패 → 처음부터")

n_fail=0
for ridx,(_,row) in enumerate(tqdm(test.iterrows(),total=len(test))):
    if row["Id"] in done: continue
    files=[row["Input_1"],row["Input_2"],row["Input_3"],row["Input_4"]]
    content=[]
    for i,f in enumerate(files):
        content+=[{"type":"image","image":os.path.join(DATA_DIR,"test",row["Id"],f)},
                  {"type":"text","text":f"\nImage {i+1}\n"}]
    content.append({"type":"text","text":PROMPT_V5.format(s=row["Sentence"])})
    m=[{"role":"user","content":content}]
    pt=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
    ii,vi=process_vision_info(m)
    enc=proc(text=[pt],images=ii,videos=vi,return_tensors="pt").to(dev)
    with torch.no_grad():
        out=model.generate(**enc,max_new_tokens=32,do_sample=False)
    txt=proc.batch_decode(out[:,enc.input_ids.shape[1]:],skip_special_tokens=True)[0]
    sub=parse_greedy(txt)
    if sub is None:                       # 파싱 실패 → 항등순서로 대체
        n_fail+=1; sub=[1,2,3,4]
        if n_fail<=5: print(f"  ⚠️ 파싱실패 {row['Id']}: {txt!r}")
    if ridx<SHOW_FIRST: print(f"  [{ridx}] 모델출력={txt.strip()!r} → 제출={sub}")
    recs.append({"Id":row["Id"],"Answer":str(sub)})
    if len(recs)%SAVE_EVERY==0:           # ★중간 저장
        pd.DataFrame(recs).to_csv(OUT_PATH,index=False)

out=pd.DataFrame(recs)
out.to_csv(OUT_PATH,index=False)

# ---- 요약: 파싱실패율·항등답 비율로 이상 조기 감지 ---------------------------
ident=(out["Answer"]=="[1, 2, 3, 4]").mean()
print(f"\n✅ submission.csv 생성 ({len(out)}행, greedy)")
print(f"   파싱 실패 {n_fail}행 ({n_fail/max(len(out),1):.1%})  ← 5% 넘으면 프롬프트/어댑터 점검")
print(f"   항등답 [1,2,3,4] 비율 {ident:.1%}                ← 정상 4~10%. 20% 넘으면 이상")
print("   Output에서 다운로드 → 제출!")
