# -*- coding: utf-8 -*-
r"""학습 로그 실시간 감시 — GPU를 전혀 건드리지 않는다 (파일 읽기 + nvidia-smi 조회만).

⚠️ 파일명에 'train'을 넣지 말 것 — train_queue의 학습 감지 정규식(train(_cot)?\.py)에
   걸려 큐가 학습 중으로 오인한다 (HANDOVER 함정 16번 유형).

사용:
    python scripts/watch_log.py                     # 기본: exp16_sparsecam_aug
    python scripts/watch_log.py --run mini_v1_aug1  # 다른 run
    python scripts/watch_log.py --once              # 1회 출력 후 종료
"""
import argparse
import os
import subprocess
import time

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def gpu_line():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,clocks.sm,power.draw,temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return f"GPU: {out}"
    except Exception as e:
        return f"GPU 조회 실패: {e}"


def show(run_name, n_tail=3):
    log = os.path.join("outputs", "runs", run_name, "train_log.csv")
    print(f"\n===== {time.strftime('%H:%M:%S')} | {run_name} =====")
    if not os.path.exists(log):
        print("train_log.csv 아직 없음 (모델 로드 중이거나 시작 전)")
    else:
        age = int(time.time() - os.path.getmtime(log))
        with open(log, encoding="utf-8") as f:
            lines = f.read().splitlines()
        print(f"[{log}] 갱신 {age}초 전 | 총 {max(len(lines) - 1, 0)}개 기록")
        for line in lines[:1] + lines[-n_tail:] if len(lines) > 1 else lines:
            print("  " + line)
    print(gpu_line())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="exp16_sparsecam_aug")
    p.add_argument("--interval", type=int, default=60, help="갱신 주기(초)")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    while True:
        show(args.run)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
