# -*- coding: utf-8 -*-
"""프롬프트 스크리닝 큐 — 학습(train.py) 종료를 기다렸다가 zero-shot 평가를 순차 실행한다.

- 학습과 GPU를 다투지 않도록 train.py 프로세스가 모두 끝난 뒤에만 시작
- 커널과 무관한 독립 프로세스 (Train_Experiments.ipynb ⑩ 셀 또는 터미널에서 실행)
- 결과는 eval_zero_shot.py가 outputs/experiments.csv에 누적 (prompt 컬럼으로 구분)
- 실행 중 절전 진입 차단

사용 예:
    python scripts/prompt_screen_queue.py --prompts v2_temporal v3_cot:256
    ("이름:N" -> --max-new-tokens N, 생략 시 32. CoT 계열은 256 권장)
"""
import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG_PATH = PROJECT / "outputs" / "prompt_queue.log"


def log(msg):
    """백그라운드 실행이므로 진행 기록을 반드시 프로젝트 파일로 남긴다."""
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def keep_system_awake():
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def train_running() -> bool:
    """train.py 프로세스가 하나라도 살아 있으면 True. 확인 실패 시 True(안전측)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
             "| Where-Object { $_.CommandLine -like '*train.py*' }).Count"],
            capture_output=True, text=True, timeout=60,
        )
        return int(out.stdout.strip()) > 0
    except Exception:
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--prompts", nargs="+", required=True,
                        help="프롬프트 이름 또는 이름:max_new_tokens (예: v3_cot:256)")
    parser.add_argument("--adapter", default=None, help="지정 시 어댑터를 얹어 평가")
    parser.add_argument("--max-wait-hours", type=float, default=24.0)
    args = parser.parse_args()

    keep_system_awake()
    log(f"큐 가동: prompts={args.prompts}, model={args.model}, adapter={args.adapter}")

    deadline = time.time() + args.max_wait_hours * 3600
    while train_running():
        if time.time() > deadline:
            log("학습 종료 대기 시간 초과 - 큐 포기")
            sys.exit(1)
        log("학습 진행 중 - 5분 뒤 재확인")
        time.sleep(300)

    # 노트북의 자동 평가 셀(⑥)이 먼저 GPU를 잡도록 잠시 양보 (이후는 VRAM 대기로 자연 직렬화)
    log("학습 종료 감지 - 120초 뒤 평가 시작")
    time.sleep(120)

    failed = []
    for spec in args.prompts:
        name, _, tokens = spec.partition(":")
        cmd = [PY, str(PROJECT / "scripts" / "eval_zero_shot.py"),
               "--model", args.model, "--prompt", name,
               "--max-new-tokens", tokens or "32"]
        if args.adapter:
            cmd += ["--adapter", args.adapter]
        for attempt in (1, 2):
            log(f"[{name}] 평가 시작 (시도 {attempt})")
            with open(LOG_PATH, "a", encoding="utf-8") as lf:
                rc = subprocess.run(cmd, cwd=PROJECT, stdout=lf, stderr=subprocess.STDOUT).returncode
            if rc == 0:
                log(f"[{name}] 성공")
                break
            log(f"[{name}] 실패 (exit {rc}) - 60초 뒤 재시도")
            time.sleep(60)
        else:
            failed.append(name)

    log("큐 완료" + (f" (실패: {failed})" if failed else " - 전 항목 성공, 결과: outputs/experiments.csv, outputs/preds/"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
