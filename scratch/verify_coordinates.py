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
        print(f"    x perm={p} 정답={a} 타깃={t} 복원={b}")

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

    # A-4. 24개 후보의 토큰 길이가 같은가 (L=len(atok[0]) assumption)
    lens = {len(tgt_str(a, [0, 1, 2, 3])) for a in CANDS}
    print(f"A-4 후보 문자열 길이 균일       : {'OK' if len(lens)==1 else 'x '+str(lens)}")

    ok = (n1 == 0) and (not bad) and (n3 == 0) and (len(lens) == 1)
    print("-" * 70)
    print("✅ 좌표 로직 정상 — sub=best 그대로 제출 가능" if ok else "❌ 좌표 문제 발견")
    print("=" * 70)
    return ok

if __name__ == "__main__":
    run_cell_a()
