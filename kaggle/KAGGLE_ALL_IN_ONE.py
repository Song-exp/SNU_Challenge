# ================================================================================
# SNU AI Challenge — Kaggle 8B 학습 + 제출 (초보용 올인원)
# ================================================================================
# 사용법: 이 파일 전체를 Kaggle 노트북 셀 하나에 복사 → 실행. 끝.
# 데이터 경로는 자동으로 찾습니다 (수정 불필요).
# 세션이 12시간에 끊기면: 노트북을 그냥 다시 실행하세요. 이어서 학습합니다.
# ================================================================================

# ---- 1. 설치 (Kaggle에 없는 것만) --------------------------------------------
import subprocess, sys
def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs])
pip("transformers==5.13.0", "peft", "bitsandbytes", "accelerate", "qwen-vl-utils")

import os, ast, json, glob, random, time
import pandas as pd, torch
from tqdm.auto import tqdm

# ---- 2. 데이터 경로 자동 탐색 -------------------------------------------------
def find_path(target_name, must_have=None):
    """/kaggle/input 아래에서 파일/폴더를 이름으로 자동 탐색."""
    for root, dirs, files in os.walk("/kaggle/input"):
        names = dirs + files
        if target_name in names:
            p = os.path.join(root, target_name)
            if must_have is None or os.path.exists(os.path.join(p, must_have)):
                return p
    return None

DATA_DIR = None
# train.csv가 있는 폴더를 대회 데이터로 인식
for root, dirs, files in os.walk("/kaggle/input"):
    if "train.csv" in files and "test.csv" in files and "train" in dirs:
        DATA_DIR = root
        break
assert DATA_DIR, "❌ 대회 데이터를 못 찾음. 오른쪽 'Add Input'에서 대회 데이터를 추가하세요."
print(f"✅ 대회 데이터: {DATA_DIR}")

def find_csv(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    return hits[0] if hits else None

AUG_WEIGHTS = find_csv("aug_weights_exp16.csv")   # snu-ai-aux 데이터셋에서
CLIP_FEATS  = find_csv("snu_clip_features.csv")
HOLDOUT     = find_csv("holdout_300.csv")
print(f"aug_weights: {AUG_WEIGHTS}\nclip: {CLIP_FEATS}\nholdout: {HOLDOUT}")
if not AUG_WEIGHTS:
    print("⚠️ aug_weights 없음 → 균일 증강으로 진행 (snu-ai-aux 데이터셋 추가 권장)")

# ---- 3. 설정 (건드릴 필요 없음) -----------------------------------------------
CFG = dict(
    model_id="Qwen/Qwen3-VL-8B-Instruct",
    prompt_v5=("Look at the 4 images above labeled Image 1 to Image 4. Determine the "
               "correct chronological order of these images to match the sentence below.\n"
               'Sentence: "{s}"\nProvide the answer ONLY as a Python list of integers. '
               "Example: [1, 2, 3, 4]"),
    aug_mult=2, hard_shuffle=True, lr=1e-4, lora_r=16, lora_alpha=32,
    lora_targets="q_proj,k_proj,v_proj,o_proj", grad_accum=16,
    max_pixels=512*384, warmup_ratio=0.03, seed=42,
    out="/kaggle/working/adapter", ckpt="/kaggle/working/ckpt",
    save_every=100, max_seconds=11.3*3600,
)
os.makedirs(CFG["ckpt"], exist_ok=True)
random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
rng = random.Random(CFG["seed"])

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
    print(f"holdout {len(hold)}개 제외 → train {len(train_df)}")
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

# ---- 5. 모델 로드 (4bit QLoRA, 체크포인트 재개) -------------------------------
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from qwen_vl_utils import process_vision_info

quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
model=AutoModelForImageTextToText.from_pretrained(CFG["model_id"],dtype=torch.bfloat16,
                                                  device_map="auto",quantization_config=quant)
proc=AutoProcessor.from_pretrained(CFG["model_id"],max_pixels=CFG["max_pixels"])
model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
model.config.use_cache=False

resume=0; meta=os.path.join(CFG["ckpt"],"meta.json")
if os.path.exists(os.path.join(CFG["ckpt"],"adapter_model.safetensors")):
    model=PeftModel.from_pretrained(model,CFG["ckpt"],is_trainable=True)
    resume=json.load(open(meta))["step"]; print(f"⏩ 재개: step {resume}")
else:
    model=get_peft_model(model,LoraConfig(r=CFG["lora_r"],lora_alpha=CFG["lora_alpha"],
        lora_dropout=0.05,target_modules=CFG["lora_targets"].split(","),bias="none",task_type="CAUSAL_LM"))
model.print_trainable_parameters(); model.train()

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
    json.dump({"step":step},open(meta,"w")); print(f"💾 저장 step {step}",flush=True)

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
                print("⏰ 시간한도 — 저장 후 종료. 노트북 재실행하면 이어감."); save_ckpt(step); raise KeyboardInterrupt
except KeyboardInterrupt: pass
model.save_pretrained(CFG["out"]); save_ckpt(step)
pct=step/total*100
print(f"\n{'='*50}\n종료: {step}/{total} 스텝 ({pct:.0f}%), {(time.time()-t0)/3600:.1f}h, OOM스킵 {skip}")
if pct<99: print("⚠️ 미완주 — 노트북을 다시 실행(Run All)하면 체크포인트에서 이어집니다.")
else: print("✅ 완주! 아래 추론 코드로 submission.csv를 만드세요.")
print(f"어댑터: {CFG['out']}")

# ================================================================================
# ↓↓↓ 학습이 100% 완주한 뒤에만 실행하세요 (추론·제출 생성) ↓↓↓
# ================================================================================
RUN_INFERENCE = False   # ← 학습 완주 후 True로 바꾸고 이 아래 셀 실행

if RUN_INFERENCE:
    from itertools import permutations
    CANDS=[list(p) for p in permutations([1,2,3,4])]
    def tgt_str(a):
        c=[0]*4
        for i,pos in enumerate(a): c[pos-1]=i+1
        f=[1,2,3,4]; tf=[f[n-1] for n in c]
        return str([f.index(x)+1 for x in tf])
    model.eval()
    inner=model
    for _ in range(4):
        if hasattr(inner,"rope_deltas"): break
        inner=getattr(inner,"model",None) or getattr(inner,"base_model",None)
    eos=proc.tokenizer("<|im_end|>",add_special_tokens=False)["input_ids"]
    test=pd.read_csv(os.path.join(DATA_DIR,"test.csv"))
    recs=[]
    for _,row in tqdm(test.iterrows(),total=len(test)):
        files=[row["Input_1"],row["Input_2"],row["Input_3"],row["Input_4"]]
        content=[]
        for i,f in enumerate(files):
            content+=[{"type":"image","image":os.path.join(DATA_DIR,"test",row["Id"],f)},
                      {"type":"text","text":f"\nImage {i+1}\n"}]
        content.append({"type":"text","text":CFG["prompt_v5"].format(s=row["Sentence"])})
        m=[{"role":"user","content":content}]
        pt=proc.apply_chat_template(m,tokenize=False,add_generation_prompt=True)
        ii,vi=process_vision_info(m); enc=proc(text=[pt],images=ii,videos=vi,return_tensors="pt").to(dev)
        plen=enc["input_ids"].shape[1]
        atok=[proc.tokenizer(tgt_str(a),add_special_tokens=False)["input_ids"]+eos for a in CANDS]
        L=len(atok[0]); amat=torch.tensor(atok,device=dev)
        with torch.no_grad():
            o1=model(**enc,use_cache=True); pkv=o1.past_key_values; rd=inner.rope_deltas.item()
            flp=torch.log_softmax(o1.logits[:,-1,:].float(),-1)
            pos=(torch.arange(plen,plen+L,device=dev)+rd).view(1,1,-1).expand(3,24,-1).contiguous()
            pkv.batch_repeat_interleave(24)
            o2=model(input_ids=amat,position_ids=pos,past_key_values=pkv,
                     attention_mask=torch.ones(24,plen+L,device=dev,dtype=torch.long),use_cache=True)
            lp0=flp[0,amat[:,0]]; rest=torch.log_softmax(o2.logits[:,:-1].float(),-1)
            tot=lp0+rest[torch.arange(24)[:,None],torch.arange(L-1)[None,:],amat[:,1:]].sum(1)
            best=CANDS[tot.argmax()]
        sub=[0]*4
        for i,n in enumerate(best): sub[n-1]=i+1
        recs.append({"Id":row["Id"],"Answer":str(sub)})
    out=pd.DataFrame(recs)
    out.to_csv("/kaggle/working/submission.csv",index=False)
    print(f"✅ submission.csv 생성 ({len(out)}행). 오른쪽 Output에서 다운로드 → 리더보드 제출!")
