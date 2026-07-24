# ================================================================================
# ★ 본 추론(5시간) 돌리기 전 검증 — 좌표 뒤집힘(0.44 버그) / 어댑터 미로드 조기 탐지
# ================================================================================
# 셀 A: 좌표 로직만 검증. GPU 불필요, 3초. 어디서든 단독 실행 가능.
# 셀 B: 실제 모델로 train 20행 리허설. INFER_ONLY_K4.py 안에 끼워넣어 실행 (약 5분).
# ================================================================================


# ================================================================================
# 【셀 A】 좌표 로직 검증 — 단독 실행 (GPU 불필요, 3초)
# ================================================================================
import ast
import sys
from itertools import permutations

try:  # 윈도우 cp949 콘솔에서 이모지 출력 시 죽는 것 방지 (Kaggle은 영향 없음)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CANDS = [list(p) for p in permutations([1, 2, 3, 4])]
PERMS = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]


def tgt_str(answer, perm):
    """INFER_ONLY_K4.py 81행에서 그대로 복사"""
    c = [0] * 4
    for i, pos in enumerate(answer):
        c[pos - 1] = i + 1
    files = [1, 2, 3, 4]
    tf = [files[n - 1] for n in c]
    shown = [files[j] for j in perm]
    return str([shown.index(f) + 1 for f in tf])


def train_target(answer, perm):
    """kaggle_8b_train.py 118-138행 학습 타깃 생성식 (파일명 대신 번호로 치환)"""
    files = [1, 2, 3, 4]
    chrono = [0] * 4
    for i, pos in enumerate(answer):
        chrono[pos - 1] = i + 1
    time_files = [files[n - 1] for n in chrono]
    shown = [files[j] for j in perm]
    return str([shown.index(f) + 1 for f in time_files])


def decode_back(txt, perm):
    """모델 출력(보여준 Image 슬롯의 시간순) → 제출 형식으로 되돌리기"""
    r = ast.literal_eval(txt)
    shown = [[1, 2, 3, 4][j] for j in perm]  # Image i+1 이 실제로 보여주는 Input 번호
    order = [shown[n - 1] for n in r]  # 시간순 Input 번호
    sub = [0] * 4
    for rank, inp in enumerate(order):
        sub[inp - 1] = rank + 1
    return sub


def run_cell_a():
    print("=" * 70)
    print("【셀 A】 좌표 로직 검증")
    print("=" * 70)

    # A-1. 학습 타깃식 == 추론 타깃식 인가 (학습/추론 좌표계 일치)
    n1 = sum(
        1
        for p in PERMS
        for a in CANDS
        if tgt_str(a, p) != train_target(a, p)
    )
    print(f"A-1 학습식 vs 추론식 일치      : 불일치 {n1}건 / 96건")

    # A-2. 왕복 검증 — 타깃 문자열을 되돌리면 원래 후보가 나오는가
    bad = [
        (p, a, tgt_str(a, p), decode_back(tgt_str(a, p), p))
        for p in PERMS
        for a in CANDS
        if decode_back(tgt_str(a, p), p) != a
    ]
    print(f"A-2 좌표 왕복 (4perm × 24후보) : 불일치 {len(bad)}건 / 96건")
    for p, a, t, b in bad[:5]:
        print(f"    ✗ perm={p} 정답={a} 타깃={t} 복원={b}")

    # A-3. greedy 경로와 우도 경로가 같은 좌표계를 쓰는가
    def parse_greedy(txt):  # INFER_ONLY_K4.py 87행 그대로
        s = txt.rfind("[")
        e = txt.find("]", s)
        r = ast.literal_eval(txt[s : e + 1])
        sub = [0] * 4
        for i, n in enumerate(r):
            sub[n - 1] = i + 1
        return sub

    n3 = sum(1 for a in CANDS if parse_greedy(tgt_str(a, [0, 1, 2, 3])) != a)
    print(f"A-3 greedy 경로 좌표계 일치    : 불일치 {n3}건 / 24건")

    # A-4. 24개 후보의 토큰 길이가 같은가 (L=len(atok[0]) 가정 검증)
    lens = {len(tgt_str(a, [0, 1, 2, 3])) for a in CANDS}
    print(f"A-4 후보 문자열 길이 균일       : {'OK' if len(lens)==1 else '✗ '+str(lens)}")

    ok = (n1 == 0) and (not bad) and (n3 == 0) and (len(lens) == 1)
    print("-" * 70)
    print("✅ 좌표 로직 정상 — sub=best 그대로 제출 가능" if ok else "❌ 좌표 문제 발견")
    print("=" * 70)
    return ok


# ================================================================================
# 【셀 B】 실전 리허설 — INFER_ONLY_K4.py 의 133행(score_perm 정의 끝) 바로 뒤,
#                        135행(재시작 블록) 앞에 아래를 통째로 끼워넣고 실행.
#         모델이 이미 로드된 상태를 재사용하므로 추가 로딩 없음. 약 5분.
#         정상이면 그대로 본 추론이 이어서 돌아감. 문제면 즉시 멈춤.
# ================================================================================
CELL_B = r'''
# ---- 【검증】 train 20행 리허설 (본 추론 전 5분 안전장치) --------------------
VERIFY_N = 20          # 0으로 두면 검증 건너뜀
if VERIFY_N:
    import ast as _ast
    print("\n" + "="*70)
    print(f"【검증】 어댑터 확인 + train {VERIFY_N}행 리허설")
    print("="*70)

    # B-1. 어댑터가 실제로 붙었는가 (lora_B는 학습 전 0 → 0이면 학습분 미반영)
    _nb = _mx = 0
    for _n, _p in model.named_parameters():
        if "lora_B" in _n:
            _nb += 1; _mx = max(_mx, _p.abs().max().item())
    print(f"B-1 lora_B 텐서 {_nb}개, 최대 절대값 {_mx:.6f}"
          f"  → {'✅ 어댑터 반영됨' if _mx > 1e-6 else '❌ 어댑터 미반영! 경로 확인'}")
    assert _mx > 1e-6, "어댑터가 로드되지 않았습니다 (lora_B 전부 0)"

    # B-2. 실제 파이프라인으로 정답 아는 train 행 채점
    _tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv")).head(VERIFY_N)
    _ok = _okinv = 0
    for _, _row in tqdm(_tr.iterrows(), total=len(_tr), desc="리허설"):
        _files = [_row["Input_1"], _row["Input_2"], _row["Input_3"], _row["Input_4"]]
        _score = {tuple(a): 0.0 for a in CANDS}
        for _perm in PERMS:
            _shown = [_files[j] for j in _perm]
            _content = []
            for _i, _f in enumerate(_shown):
                _content += [{"type":"image","image":os.path.join(DATA_DIR,"train",_row["Id"],_f)},
                             {"type":"text","text":f"\nImage {_i+1}\n"}]
            _content.append({"type":"text","text":PROMPT_V5.format(s=_row["Sentence"])})
            _m = [{"role":"user","content":_content}]
            _pt = proc.apply_chat_template(_m, tokenize=False, add_generation_prompt=True)
            _ii, _vi = process_vision_info(_m)
            _enc = proc(text=[_pt], images=_ii, videos=_vi, return_tensors="pt").to(dev)
            _plen = _enc["input_ids"].shape[1]
            _atok = [proc.tokenizer(tgt_str(a,_perm), add_special_tokens=False)["input_ids"]+eos
                     for a in CANDS]
            _L = len(_atok[0]); _amat = torch.tensor(_atok, device=dev)
            _tot = score_perm(_enc, _plen, _amat, _L, CHUNK)
            for a, s in zip(CANDS, _tot.tolist()):
                _score[tuple(a)] += s
        _best = max(CANDS, key=lambda a: _score[tuple(a)])
        _gt = _ast.literal_eval(_row["Answer"])
        _inv = [0]*4                                   # 어제 0.44 버그 재현본
        for _i, _n in enumerate(_best): _inv[_n-1] = _i+1
        _ok += int(_best == _gt); _okinv += int(_inv == _gt)

    _a1 = _ok/len(_tr); _a2 = _okinv/len(_tr)
    print("-"*70)
    print(f"B-2 sub=best (현재 코드)  정확도 {_a1:6.1%}  ({_ok}/{len(_tr)})")
    print(f"    한번 더 역변환 (버그본) 정확도 {_a2:6.1%}  ({_okinv}/{len(_tr)})")
    print(f"    무작위 기대값 4.2%")
    print("-"*70)
    if _a1 >= 0.5 and _a1 > _a2:
        print("✅ 정상 — 좌표 맞음. 본 추론 진행합니다.")
        print("   (train은 학습에 쓴 데이터라 점수가 높게 나오는 게 정상. 실제 성능 아님)")
    elif _a2 > _a1:
        raise SystemExit("❌ 좌표 뒤집힘! sub=best 를 역변환본으로 바꿔야 합니다.")
    else:
        raise SystemExit(f"❌ 둘 다 낮음({_a1:.1%}/{_a2:.1%}) — 어댑터·해상도·프롬프트 점검 필요")
    print("="*70 + "\n")
    del _tr
    torch.cuda.empty_cache()
# ---- 【검증】 끝 -------------------------------------------------------------
'''

if __name__ == "__main__":
    run_cell_a()
    print("\n[셀 B] 는 아래 블록을 INFER_ONLY_K4.py 133행 뒤에 붙여넣어 실행하세요:")
    print(CELL_B)
