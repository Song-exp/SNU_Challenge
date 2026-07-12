# -*- coding: utf-8 -*-
"""밤샘 오케스트레이터: 모델 다운로드 완료를 기다렸다가 zero-shot 평가를 순차 실행한다.

동작:
1. 각 모델이 완전히 받아졌는지 확인 (HF 서버의 파일 크기 목록과 대조)
2. 다운로더가 죽어 있으면 다시 살린다 (download_models.py는 받은 만큼 건너뛰므로 안전)
3. 모두 받아지면 아직 평가 안 된 모델을 하나씩 평가 (experiments.csv에 누적)
4. 실행 중 절전 진입 차단

사용법: python scripts/overnight_run.py
"""
import ctypes
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PY = sys.executable
RESULTS = PROJECT / "outputs" / "experiments.csv"

# (모델 HF id, 4bit 여부) — 8GB VRAM에 들어가는 구동 방식 기준
MODELS = [
    ("Qwen/Qwen2-VL-2B-Instruct", False),
    ("Qwen/Qwen2.5-VL-3B-Instruct", True),   # fp16(7.5GB)은 8GB에 못 올림 -> 4bit
    ("Qwen/Qwen2.5-VL-7B-Instruct", True),
    ("Qwen/Qwen3-VL-2B-Instruct", False),    # 최신 세대 2B — fp16 가능
    ("Qwen/Qwen3-VL-4B-Instruct", True),     # 최신 세대 4B — 4bit
]


def keep_system_awake():
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def local_dir(model_id: str) -> Path:
    return PROJECT / "models" / model_id.split("/")[-1]


def expected_sizes(model_id: str, cache={}):
    """HF 서버에 물어본 파일 크기 목록 (모델당 1회만 조회)."""
    if model_id not in cache:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_id, files_metadata=True)
        cache[model_id] = {s.rfilename: s.size for s in info.siblings}
    return cache[model_id]


def is_downloaded(model_id: str) -> bool:
    try:
        expected = expected_sizes(model_id)
    except Exception as e:
        print(f"크기 조회 실패({e}) — 다음 회차에 재시도", flush=True)
        return False
    d = local_dir(model_id)
    for name, size in expected.items():
        f = d / name
        if size is not None and (not f.exists() or f.stat().st_size != size):
            return False
    return True


# 다운로더는 이 오케스트레이터가 직접 낳은 자식 프로세스 하나만 사용한다.
# (시스템 전체를 스캔하는 방식은 중복 실행을 못 막았던 전례가 있음)


def already_evaluated() -> set:
    if not RESULTS.exists():
        return set()
    with open(RESULTS, newline="", encoding="utf-8") as f:
        return {row["model_id"].rstrip("/").split("/")[-1] for row in csv.DictReader(f)}


def main():
    keep_system_awake()

    # 1단계: 다운로드 완료 대기 (다운로더는 내 자식 프로세스 하나만, 죽으면 부활)
    dl_proc = None
    while True:
        pending = [m for m, _ in MODELS if not is_downloaded(m)]
        if not pending:
            print("모든 모델 다운로드 완료", flush=True)
            break
        print(f"다운로드 대기 중: {pending}", flush=True)
        if dl_proc is None or dl_proc.poll() is not None:
            print("다운로더 시작", flush=True)
            dl_proc = subprocess.Popen([PY, str(PROJECT / "scripts" / "download_models.py")])
        time.sleep(120)
    if dl_proc is not None and dl_proc.poll() is None:
        dl_proc.wait()  # 다운로더가 마무리 검증 중이면 끝날 때까지 대기

    # 2단계: 미평가 모델 순차 평가 (모델별 별도 프로세스 = VRAM 완전 해제 보장)
    done = already_evaluated()
    for model_id, use_4bit in MODELS:
        name = model_id.split("/")[-1]
        if name in done:
            print(f"[{name}] 이미 평가됨 - 건너뜀", flush=True)
            continue
        cmd = [PY, str(PROJECT / "scripts" / "eval_zero_shot.py"), "--model", f"./models/{name}"]
        if use_4bit:
            cmd.append("--load-4bit")
        print(f"[{name}] 평가 시작", flush=True)
        r = subprocess.run(cmd, cwd=PROJECT)
        print(f"[{name}] 평가 종료 (exit {r.returncode})", flush=True)

    print("\n밤샘 작업 완료 -> outputs/experiments.csv 확인", flush=True)


if __name__ == "__main__":
    main()
