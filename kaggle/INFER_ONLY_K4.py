# ================================================================================
# ★★★ 추론 전용 — 우도 K4 (학습된 어댑터로 submission 생성) ★★★
# ================================================================================
# 용도: 팀원/본인이 학습한 8B 어댑터(체크포인트)로 test 추론 → submission.csv
#       학습 안 함. 어댑터만 로드해서 우도 K=4 순열 TTA (holdout +4.76%p).
#
# 준비 (Add Input):
#   1) 대회 데이터 (train.csv/test.csv 있는 것)
#   2) 학습 어댑터가 든 것 — 두 경우:
#      · 팀원 학습 Output (notebook Output, ckpt 폴더에 adapter_model.safetensors)
#      · 또는 본인 학습 노트북의 Output
#   GPU T4×2, Internet On, Commit 실행
#
# ⚠️ 우도 K4 = test 819 × 4배치라 ~5-7h (T4). 세션 시간 확인.
#    빠른 확인용이면 아래 USE_LIKELIHOOD=False (greedy, ~30분)로 먼저 제출.
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
MAX_PIXELS = 384*512           # ★팀원 8B 학습값과 일치(kaggle_8b_train.py). 반드시 학습과 동일해야 함.
USE_LIKELIHOOD = True          # True=우도K4(+4.76%p, 느림) / False=greedy(빠름, 점수확인용)
CHUNK      = 6                 # ★OOM방지: 후보24개를 6개씩(512×384는 프리픽스가 커서 6부터). 또 터지면 자동 반감.
SAVE_EVERY = 50                # ★중간 저장 간격(행). 죽어도 여기까지는 보존+재시작 시 스킵.
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

# 어댑터 탐색: adapter_model.safetensors 있는 폴더 (ckpt 또는 adapter)
ADAPTER=None
for p in glob.glob("/kaggle/input/**/adapter_model.safetensors", recursive=True):
    ADAPTER=os.path.dirname(p); break
# working에도 있으면 (같은 노트북서 학습했으면)
if not ADAPTER:
    for p in glob.glob("/kaggle/working/**/adapter_model.safetensors", recursive=True):
        ADAPTER=os.path.dirname(p); break
assert ADAPTER, "❌ 어댑터 없음 — 학습 Output(ckpt)을 Add Input 하세요"
print("✅ 어댑터:", ADAPTER)

# ---- 모델 로드 (학습과 동일 방식) ---------------------------------------------
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
n_gpu=torch.cuda.device_count(); max_mem={i:"14GiB" for i in range(n_gpu)}
model=AutoModelForImageTextToText.from_pretrained(MODEL_ID,dtype=torch.bfloat16,
        device_map="auto",max_memory=max_mem,quantization_config=quant)
model=PeftModel.from_pretrained(model,ADAPTER)
model.eval()
proc=AutoProcessor.from_pretrained(MODEL_ID,max_pixels=MAX_PIXELS)
dev=next(model.parameters()).device
# rope_deltas 모듈 탐색 (M-RoPE 우도용)
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

def parse_greedy(txt):
    s=txt.rfind("["); e=txt.find("]",s)
    try:
        r=ast.literal_eval(txt[s:e+1])
        if isinstance(r,list) and sorted(r)==[1,2,3,4]:
            sub=[0]*4
            for i,n in enumerate(r): sub[n-1]=i+1
            return sub
    except: pass
    return [1,2,3,4]

OUT_PATH="/kaggle/working/submission.csv"

# ---- 우도 채점 (청크로 나눠 OOM 방지) ----------------------------------------
import copy
from transformers.cache_utils import DynamicCache
def _clone_cache(cache):
    """DynamicCache를 원본 훼손 없이 복제. deepcopy 우선, 실패 시 버전별 수동복제."""
    try:
        return copy.deepcopy(cache)
    except Exception:
        new=DynamicCache()
        if hasattr(cache,"layers") and cache.layers:            # transformers 5.x
            for i,lyr in enumerate(cache.layers):
                new.update(lyr.keys.clone(),lyr.values.clone(),i)
        else:                                                    # transformers 4.x
            for i,(k,v) in enumerate(zip(cache.key_cache,cache.value_cache)):
                new.update(k.clone(),v.clone(),i)
        return new
def score_perm(enc, plen, amat, L, ch):
    """프리픽스는 perm당 1회만 forward(속도 유지). KV캐시를 deepcopy로 복제해
       후보를 ch개씩만 배치→ 메모리 스파이크를 24배→ch배로 제한(OOM 방지)."""
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

# ---- 재시작: 이미 처리한 Id는 스킵 (죽어도 이어감) ----------------------------
recs=[]; done=set()
if os.path.exists(OUT_PATH):
    prev=pd.read_csv(OUT_PATH)
    recs=prev.to_dict("records"); done=set(prev["Id"].tolist())
    print(f"↻ 재시작: {len(done)}행 이미 완료 → 스킵")

for ridx,(_,row) in enumerate(tqdm(test.iterrows(),total=len(test))):
    if row["Id"] in done: continue
    files=[row["Input_1"],row["Input_2"],row["Input_3"],row["Input_4"]]
    if USE_LIKELIHOOD:
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
            while True:  # OOM 시 청크를 반으로 줄여 재시도
                try:
                    tot=score_perm(enc,plen,amat,L,ch); break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if ch<=2: raise
                    ch=max(2,ch//2); print(f"  ⚠️OOM→CHUNK {ch}로 재시도")
            for a,s in zip(CANDS,tot.tolist()): score[tuple(a)]+=s
        best=max(CANDS,key=lambda a:score[tuple(a)])
        sub=best   # ★best는 이미 정답형식 — 추가 역변환 금지(팀원 0.44 버그 방지)
    else:  # greedy (빠른 확인용)
        content=[]
        for i,f in enumerate(files):
            content+=[{"type":"image","image":os.path.join(DATA_DIR,"test",row["Id"],f)},
                      {"type":"text","text":f"\nImage {i+1}\n"}]
        content.append({"type":"text","text":PROMPT_V5.format(s=row["Sentence"])})
        m=[{"role":"user","content":content}]
        pt=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
        ii,vi=process_vision_info(m); enc=proc(text=[pt],images=ii,videos=vi,return_tensors="pt").to(dev)
        with torch.no_grad():
            out=model.generate(**enc,max_new_tokens=32,do_sample=False)
        txt=proc.batch_decode(out[:,enc.input_ids.shape[1]:],skip_special_tokens=True)[0]
        sub=parse_greedy(txt)
    recs.append({"Id":row["Id"],"Answer":str(sub)})
    if len(recs)%SAVE_EVERY==0:            # ★중간 저장 — 죽어도 여기까지 보존
        pd.DataFrame(recs).to_csv(OUT_PATH,index=False)

out=pd.DataFrame(recs)
out.to_csv(OUT_PATH,index=False)
mode="우도K4" if USE_LIKELIHOOD else "greedy"
print(f"✅ submission.csv 생성 ({len(out)}행, {mode}). Output에서 다운로드 → 제출!")
