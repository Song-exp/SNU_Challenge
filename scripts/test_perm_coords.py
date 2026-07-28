# -*- coding: utf-8 -*-
"""우도 스코어링 0단계 — 좌표 규약 단위 테스트 (GPU 불필요).

목적: "제시 순열 π 하에서 정답 후보 a를 채점할 문자열"을 만드는 변환이
train.py의 타깃 생성·eval_zero_shot의 파서와 자소 단위로 정합함을 증명.
이 테스트 통과 전 채점 하니스 착수 금지 (OWL-ViT 누수와 같은 부류의 방향 혼동 위험).

핵심 정의 (train.py에서 역추출):
- 대회 정답 answer: answer[i] = 원본 슬롯 i(Input_i, 0-based) 이미지의 시간상 등수(1-based).
- chrono_image_numbers(answer): chrono[k] = 시간상 k+1번째인 원본 이미지 번호(1-based).
- 제시 순열 perm: 새 슬롯 j에 원본 이미지 perm[j] 배치 (0-based).
- train.py 타깃: shown=[files[perm[j]]], target[k]=shown.index(time_files[k])+1
  즉 target = "각 시간 등수 k에 대응하는 '새 슬롯 번호'" (시간순 슬롯 나열).
"""
import ast
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import chrono_image_numbers  # noqa: E402
from eval_zero_shot import parse_model_output  # noqa: E402


def target_string_for_candidate(answer, perm):
    """정답 후보 answer(대회 형식)를 제시 순열 perm 하에서 채점할 문자열 생성.

    train.py build_training_items의 타깃 생성 로직을 문자열 좌표로 그대로 복제
    (파일 객체 대신 원본 이미지 번호 1..4를 심볼로 사용)."""
    files = [1, 2, 3, 4]                                  # 원본 이미지 번호 (심볼)
    chrono = chrono_image_numbers(answer)                # 시간순 원본 이미지 번호
    time_files = [files[n - 1] for n in chrono]          # 시간순 파일(=번호)
    shown = [files[j] for j in perm]                     # 새 슬롯 1..4에 오는 원본 번호
    target = [shown.index(f) + 1 for f in time_files]    # train.py:133과 동일
    return str(target)                                    # "[3, 1, 4, 2]" 형식


def submission_from_model_string(model_list, perm):
    """제시 순열 perm에서 모델이 낸 문자열 리스트 -> 대회 제출 형식(원본 좌표).

    eval의 parse_model_output은 perm=identity 가정이라 여기선 perm 역매핑을 명시.
    model_list[k] = 시간 등수 k+1에 대응하는 '새 슬롯 번호'.
    새 슬롯 s의 원본 번호 = perm[s-1]+1. 그 원본 이미지의 시간 등수 = k+1.
    제출 answer[orig-1] = 등수."""
    sub = [0] * 4
    for k, slot in enumerate(model_list):
        orig = perm[slot - 1] + 1        # 새 슬롯 -> 원본 이미지 번호
        sub[orig - 1] = k + 1            # 원본 이미지의 시간 등수
    return sub


PERMS = {
    "e0": [0, 1, 2, 3], "r1": [1, 2, 3, 0], "r2": [2, 3, 0, 1], "r3": [3, 0, 1, 2],
}


def all_answers():
    from itertools import permutations
    return [list(p) for p in permutations([1, 2, 3, 4])]


def test_roundtrip():
    """① 후보->문자열->제출 왕복이 원래 정답으로 복귀 (24후보 x 4배치 전수)."""
    fails = 0
    for a in all_answers():
        for name, perm in PERMS.items():
            s = target_string_for_candidate(a, perm)
            model_list = ast.literal_eval(s)
            back = submission_from_model_string(model_list, perm)
            if back != a:
                fails += 1
                if fails <= 5:
                    print(f"  FAIL a={a} {name} perm={perm} str={s} back={back}")
    total = 24 * 4
    print(f"[왕복 테스트] {total - fails}/{total} 통과")
    return fails == 0


def test_e0_inverse():
    """② perm=e0에서 target 문자열은 answer의 '역순열'이어야 한다.

    answer[i]=원본 이미지 i의 시간 등수 (i->rank),
    target[k]=시간 k번째 이미지의 슬롯번호 (rank->slot).
    e0에서 슬롯=원본이므로 target = answer의 역함수. (계획서 §3.1 방향 혼동 지점)"""
    fails = 0
    for a in all_answers():
        s = ast.literal_eval(target_string_for_candidate(a, PERMS["e0"]))
        inv = [0] * 4                      # answer의 역순열
        for i, rank in enumerate(a):
            inv[rank - 1] = i + 1
        if s != inv:
            fails += 1
            if fails <= 5:
                print(f"  FAIL a={a} e0 str={s} inv={inv}")
    print(f"[e0 역순열] {24 - fails}/24 통과")
    return fails == 0


def test_eval_parser_consistency():
    """③ e0에서 생성한 문자열을 eval 파서에 넣으면 대회 제출형식으로 정확 변환."""
    fails = 0
    for a in all_answers():
        s = target_string_for_candidate(a, PERMS["e0"])
        sub, ok = parse_model_output(s)          # eval의 실제 파서
        # 우리 변환과 eval 파서가 같은 제출값을 내는가
        ours = submission_from_model_string(ast.literal_eval(s), PERMS["e0"])
        if not ok or sub != ours:
            fails += 1
            if fails <= 5:
                print(f"  FAIL a={a} str={s} eval={sub} ours={ours} ok={ok}")
    print(f"[eval 파서 정합] {24 - fails}/24 통과")
    return fails == 0


def test_manual_cases():
    """④ 손계산 케이스 — 정답을 알 때 채점 문자열이 상식과 일치."""
    ok = True
    # 케이스: answer=[1,2,3,4] (이미 시간순), perm=r1(슬롯에 원본 2,3,4,1)
    # 시간순 원본=[1,2,3,4]. r1에서 원본1은 새슬롯4, 원본2는 새슬롯1, 원본3은 슬롯2, 원본4는 슬롯3.
    # target[k]=시간 k번째 원본의 슬롯: [4,1,2,3]
    s = target_string_for_candidate([1, 2, 3, 4], PERMS["r1"])
    exp = "[4, 1, 2, 3]"
    print(f"  case1 r1 identity: {s} (기대 {exp}) {'OK' if s == exp else 'FAIL'}")
    ok &= (s == exp)
    return ok


if __name__ == "__main__":
    results = [
        test_roundtrip(),
        test_e0_inverse(),
        test_eval_parser_consistency(),
        test_manual_cases(),
    ]
    print("\n" + ("=== 전체 통과 — 하니스 착수 가능 ===" if all(results)
                  else "### 실패 — 좌표 규약 재검토, 하니스 착수 금지 ###"))
    sys.exit(0 if all(results) else 1)
