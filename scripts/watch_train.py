# -*- coding: utf-8 -*-
"""학습 진행률 감시기 — train.py가 쓰는 train_log.csv를 읽어 게이지/ETA를 보여준다.

학습 프로세스와 완전히 분리되어 있어 GPU/학습에 영향 없음. 노트북 커널이
학습 셀로 막혀 있을 때 PowerShell이나 두 번째 노트북에서 실행한다.

사용 예:
    python scripts/watch_train.py --run exp01_aug2_lr1e4          # 30초마다 갱신
    python scripts/watch_train.py --run exp01_aug2_lr1e4 --once   # 한 번만 출력
"""
import argparse
import json
import os
import time
from datetime import datetime, timedelta

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def read_status(run_dir):
    """(설정, 마지막 로그 행) 반환. 로그가 아직 없으면 (설정, None)."""
    with open(os.path.join(run_dir, "run_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    log_path = os.path.join(run_dir, "train_log.csv")
    if not os.path.exists(log_path):
        return cfg, None
    import pandas as pd
    log = pd.read_csv(log_path)
    return cfg, (log.iloc[-1] if len(log) else None)


def render(cfg, last):
    grad_accum = cfg["grad_accum"]
    total_items = cfg["total_opt_steps"] * grad_accum  # max_steps 상한 반영된 실제 목표량
    if last is None:
        return f"로그 대기 중... (첫 기록은 시작 후 약 {10 * grad_accum * 2 // 60}분 뒤)"

    done_items = int(last.opt_step * grad_accum)
    frac = min(done_items / total_items, 1.0)
    sec_per_item = float(last.sec_per_item)
    remain_sec = (total_items - done_items) * sec_per_item
    eta = datetime.now() + timedelta(seconds=remain_sec)

    bar_len = 30
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)

    return (
        f"[{bar}] {frac * 100:5.1f}%  "
        f"{done_items}/{total_items} 항목 | loss {last.loss:.3f} | {sec_per_item:.2f}초/항목 | "
        f"경과 {last.elapsed_min:.0f}분, 남은 {remain_sec / 3600:.1f}시간 | "
        f"완료 예상 {eta.strftime('%H:%M')} ({'내일 ' if eta.date() != datetime.now().date() else ''}기준) | "
        f"VRAM {last.peak_vram_gb}GB"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="run 이름 (outputs/runs/<이름>)")
    parser.add_argument("--once", action="store_true", help="한 번만 출력하고 종료")
    parser.add_argument("--interval", type=int, default=30, help="갱신 주기(초)")
    args = parser.parse_args()

    run_dir = os.path.join("./outputs/runs", args.run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run 폴더 없음: {run_dir} (학습이 시작됐는지, 이름이 맞는지 확인)")

    while True:
        cfg, last = read_status(run_dir)
        print(f"{datetime.now().strftime('%H:%M:%S')}  {render(cfg, last)}", flush=True)
        if args.once:
            break
        if last is not None and last.opt_step >= cfg["total_opt_steps"]:
            print("학습 목표 스텝 도달 — 종료", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
