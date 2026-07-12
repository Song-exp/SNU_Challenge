# -*- coding: utf-8 -*-
"""실험 후보 모델 일괄 다운로드 (curl 이어받기 방식).

연결이 끊겨도 받은 만큼 누적되는 resume 방식이라 불안정한 네트워크에서도 반드시 완료된다.
- 파일별로 curl -C - (이어받기) + 60초간 속도 없음 시 자동 재접속
- 완료 파일은 크기 검증 후 건너뜀 (재실행해도 안전)
- 실행 중 Windows 절전 진입 차단 (종료 시 자동 해제)

저장 위치: 프로젝트의 models/<모델명>/ (노트북에서 MODEL_ID = "./models/<모델명>" 으로 로드)

사용법: python scripts/download_models.py
※ 한 번에 하나만 실행할 것 (중복 실행 금지)
"""
import ctypes
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

MODELS = [
    "Qwen/Qwen2-VL-2B-Instruct",     # 베이스라인 — 최우선 (실험 노트북이 바로 사용)
    "Qwen/Qwen2.5-VL-3B-Instruct",   # 체급 상향 1순위 (8GB에서는 4-bit)
    "Qwen/Qwen2.5-VL-7B-Instruct",   # 최종 후보 체급 (8GB에서는 4-bit 필수)
    "Qwen/Qwen3-VL-2B-Instruct",     # 최신 세대 2B (2025.10 공개, 규정 기한 내)
    "Qwen/Qwen3-VL-4B-Instruct",     # 최신 세대 4B — 파인튜닝 스위트스팟 후보 (4-bit)
]

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
MAX_ATTEMPTS_PER_FILE = 40


def keep_system_awake():
    """스크립트가 도는 동안 절전 진입을 차단한다 (프로세스 종료 시 자동 해제)."""
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def download_file(url: str, dest: Path, expected_size) -> bool:
    """curl 이어받기로 파일 하나를 받는다. 끊겨도 받은 만큼 유지되어 재시도가 누적된다."""
    for attempt in range(1, MAX_ATTEMPTS_PER_FILE + 1):
        if expected_size is not None and dest.exists() and dest.stat().st_size == expected_size:
            return True
        got = dest.stat().st_size if dest.exists() else 0
        print(f"    시도 {attempt}: {got / 1e6:,.0f} / {(expected_size or 0) / 1e6:,.0f} MB 지점부터", flush=True)
        subprocess.run([
            "curl.exe", "-L", "-sS",
            "-C", "-",                    # 이어받기
            "--retry", "20", "--retry-delay", "2", "--retry-all-errors",
            "--connect-timeout", "15",
            "--speed-limit", "10240",     # 60초간 10KB/s 미만이면(멈춤)
            "--speed-time", "60",         # 연결을 끊고 --retry로 재접속
            "-o", str(dest), url,
        ])
        time.sleep(2)
    return expected_size is None or (dest.exists() and dest.stat().st_size == expected_size)


def download_model(model_id: str) -> bool:
    target = MODELS_DIR / model_id.split("/")[-1]
    target.mkdir(parents=True, exist_ok=True)

    info = HfApi().model_info(model_id, files_metadata=True)
    files = [(s.rfilename, s.size) for s in info.siblings]
    print(f"[{model_id}] 파일 {len(files)}개 -> {target}", flush=True)

    for name, size in files:
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if size is not None and dest.exists() and dest.stat().st_size == size:
            print(f"  OK(스킵) {name}", flush=True)
            continue
        print(f"  받는 중: {name}", flush=True)
        if not download_file(f"https://huggingface.co/{model_id}/resolve/main/{name}", dest, size):
            print(f"  실패: {name}", flush=True)
            return False
    print(f"[{model_id}] 완료", flush=True)
    return True


if __name__ == "__main__":
    keep_system_awake()
    results = {m: download_model(m) for m in MODELS}
    print("\n===== 결과 =====")
    for m, ok in results.items():
        print(f"{'OK ' if ok else 'FAIL'} {m}")
    sys.exit(0 if all(results.values()) else 1)
