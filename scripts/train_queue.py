# -*- coding: utf-8 -*-
"""학습 체인 큐 — 진행 중인 학습이 끝나기를 기다렸다가 다음 학습들을 순차 실행하고,
각 학습이 끝나면 holdout 평가까지 자동으로 잇는다.

- 노트북이 소유한 학습(train.py)과 GPU를 다투지 않음 (종료 감지 후에만 시작)
- 각 run의 콘솔은 outputs/runs/<이름>/console.log 로 — 노트북 게이지 셀(⑤)로 관전 가능
- 큐 진행 기록: outputs/train_queue.log
- 실행 중 절전 차단. 한 번에 하나만 실행할 것.

사용 예:
    python scripts/train_queue.py --queue "exp07_aug2_full|--aug-mult 2 --lr 1e-4 --epochs 1 --max-hours 0 --snapshot-steps 150"
    ("이름|train.py 인자들" 형식, --queue 반복 가능)
"""
import argparse
import ctypes
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG_PATH = PROJECT / "outputs" / "train_queue.log"


def log(msg):
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def keep_system_awake():
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def train_running() -> bool:
    """다른 학습 프로세스(train.py/train_cot.py)가 살아 있으면 True. 확인 실패 시 True(안전측)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
             "| Where-Object { $_.CommandLine -match 'train(_cot)?\\.py' }).Count"],
            capture_output=True, text=True, timeout=60,
        )
        return int(out.stdout.strip()) > 0
    except Exception:
        return True


def wait_until_gpu_free():
    """앞선 학습 종료를 기다린 뒤, 노트북의 자동 평가(⑥)가 지나가도록 10분 양보."""
    while train_running():
        log("앞선 학습 진행 중 - 5분 뒤 재확인")
        time.sleep(300)
    log("앞선 학습 종료 감지 - 10분 양보 후 시작 (노트북 평가 셀 우선)")
    time.sleep(600)


def parse_opt(args_str: str, flag: str, default: str) -> str:
    toks = args_str.split()
    return toks[toks.index(flag) + 1] if flag in toks else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", action="append", required=True,
                        help='"run이름|train.py 인자들" (반복 지정 가능, 순서대로 실행)')
    parser.add_argument("--after", default=None,
                        help="이 run의 console.log에 종료 마커가 찍힌 뒤에만 시작 (아직 시작 전이어도 안전)")
    parser.add_argument("--pre-eval", type=int, default=0, metavar="MAX_NEW_TOKENS",
                        help="--after run의 어댑터를 큐 시작 전에 평가 (run_config.json에서 model/prompt 자동, "
                             "CoT 계열은 512 지정)")
    args = parser.parse_args()
    if args.pre_eval and not args.after:
        parser.error("--pre-eval은 --after와 함께 써야 함 (평가 대상 run 지정)")

    keep_system_awake()
    log(f"체인 큐 가동: {len(args.queue)}개 예약 - {[q.split('|')[0] for q in args.queue]}")

    if args.after:
        marker_file = PROJECT / "outputs" / "runs" / args.after / "console.log"
        log(f"선행 run '{args.after}' 완료 대기 (종료 마커 감시: {marker_file})")
        while True:
            try:
                if "종료(" in open(marker_file, encoding="utf-8", errors="replace").read():
                    break
            except OSError:
                pass  # 아직 시작 전이면 파일이 없음 - 계속 대기
            time.sleep(300)
        log(f"선행 run '{args.after}' 종료 확인")

    failed = []

    # ---- 선행 run 평가 (--pre-eval) — 학습 프롬프트 세트 원칙에 따라 run_config.json의 prompt 사용 ----
    if args.pre_eval:
        import json
        cfg = json.load(open(PROJECT / "outputs" / "runs" / args.after / "run_config.json",
                             encoding="utf-8"))
        wait_until_gpu_free()
        eval_cmd = [PY, str(PROJECT / "scripts" / "eval_zero_shot.py"),
                    "--model", cfg["model"], "--adapter", f"./outputs/runs/{args.after}/adapter",
                    "--prompt", cfg["prompt"], "--max-new-tokens", str(args.pre_eval)]
        if cfg.get("load_4bit"):
            eval_cmd.append("--load-4bit")
        log(f"[{args.after}] 선행 평가 시작 (prompt={cfg['prompt']}, tokens={args.pre_eval})")
        with open(LOG_PATH, "a", encoding="utf-8") as lf:
            rc = subprocess.run(eval_cmd, cwd=PROJECT, stdout=lf, stderr=subprocess.STDOUT).returncode
        log(f"[{args.after}] 선행 평가 {'완료' if rc == 0 else f'실패 (exit {rc}) - 큐는 계속 진행'}")
        if rc != 0:
            failed.append(f"{args.after}(선행 평가)")
    for entry in args.queue:
        name, _, train_args = entry.partition("|")
        run_dir = PROJECT / "outputs" / "runs" / name
        run_dir.mkdir(parents=True, exist_ok=True)

        # 의사 플래그 추출 (학습 스크립트에는 전달하지 않음):
        #   --script train_cot.py        학습 엔진 교체 (기본 train.py)
        #   --eval-max-new-tokens 512    평가 생성 길이 (CoT 계열 필수)
        toks = train_args.split()
        script, eval_tokens = "train.py", None
        if "--script" in toks:
            i = toks.index("--script")
            script = toks[i + 1]
            del toks[i:i + 2]
        if "--eval-max-new-tokens" in toks:
            i = toks.index("--eval-max-new-tokens")
            eval_tokens = toks[i + 1]
            del toks[i:i + 2]

        wait_until_gpu_free()

        # ---- 학습 (콘솔은 run 폴더로 -> 노트북 게이지 셀에서 관전 가능) ----
        cmd = [PY, str(PROJECT / "scripts" / script), "--run-name", name] + toks
        log(f"[{name}] 학습 시작: {' '.join(cmd)}")
        with open(run_dir / "console.log", "w", encoding="utf-8") as cf:
            rc = subprocess.run(cmd, cwd=PROJECT, stdout=cf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            log(f"[{name}] 학습 실패 (exit {rc}) - 이 항목 중단, 다음 항목으로")
            failed.append(name)
            continue
        log(f"[{name}] 학습 완료")

        # ---- 평가 (학습과 같은 model/prompt 세트로) ----
        model = parse_opt(train_args, "--model", "./models/Qwen3-VL-2B-Instruct")
        prompt = parse_opt(train_args, "--prompt", "v1_list")
        eval_cmd = [PY, str(PROJECT / "scripts" / "eval_zero_shot.py"),
                    "--model", model, "--adapter", f"./outputs/runs/{name}/adapter",
                    "--prompt", prompt]
        if eval_tokens:
            eval_cmd += ["--max-new-tokens", eval_tokens]
        if "--load-4bit" in train_args:
            eval_cmd.append("--load-4bit")
        log(f"[{name}] 평가 시작")
        with open(LOG_PATH, "a", encoding="utf-8") as lf:
            rc = subprocess.run(eval_cmd, cwd=PROJECT, stdout=lf, stderr=subprocess.STDOUT).returncode
        log(f"[{name}] 평가 {'완료' if rc == 0 else f'실패 (exit {rc})'}")
        if rc != 0:
            failed.append(f"{name}(평가)")

    log("체인 큐 종료" + (f" - 실패: {failed}" if failed else " - 전 항목 성공, 결과: outputs/experiments.csv"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
