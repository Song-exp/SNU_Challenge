# -*- coding: utf-8 -*-
"""977/1488 중단 학습에서 '학습에 포함되지 않은' 증강 항목을 유형별로 정리.

배경: 최종 세팅(8B + 타깃증강 + hard_shuffle)을 977 opt-step까지만 학습한 체크포인트
(submission_restored_correct)의 미학습 데이터가 all_untrained_items_raw.csv.
이 파일은 '증강 항목' 단위이므로 한 Id가 여러 번 등장할 수 있다.

핵심 개념:
- total_copies[Id]   = 그 Id가 만들어낸 증강 사본 수 (= aug_weights의 aug_mult, 기본 2)
- untrained[Id]      = 미학습 CSV에 등장한 사본 수
- trained[Id]        = total - untrained
- status: fully_untrained(trained==0) / partial / fully_trained(untrained==0)

유형(stype) = structure_features.assign_types (sparse/dense x camO/camX) — 단일 진실.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import structure_features as sf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNTRAINED = os.path.join(ROOT, "all_untrained_items_raw.csv")
TRAIN_CSV = os.path.join(ROOT, "snuaichallenge_data", "train.csv")
HOLDOUT = os.path.join(ROOT, "splits", "holdout_300.csv")
AUGW = os.path.join(ROOT, "outputs", "aug_weights_exp16.csv")
OUT_PERID = os.path.join(ROOT, "outputs", "untrained_analysis_per_id.csv")
OUT_TYPE = os.path.join(ROOT, "outputs", "untrained_analysis_by_type.csv")

os.chdir(ROOT)  # structure_features의 상대경로 상수 대응


def main():
    # 1) 학습 유니버스 = train.csv - holdout
    train = pd.read_csv(TRAIN_CSV)
    hold = set(pd.read_csv(HOLDOUT)["Id"])
    universe = train[~train["Id"].isin(hold)].copy()
    print(f"학습 유니버스: train {len(train)} - holdout {len(hold)} = {len(universe)}")

    # 2) 유형 부여 (gemma 라벨 -> assign_types)
    gem = sf.load_gemma_labels()
    types = sf.assign_types(gem)[
        ["Id", "stype", "camera", "n_markers",
         "tag_multi_subj", "tag_no_marker", "tag_viewer"]
    ]
    df = universe[["Id"]].merge(types, on="Id", how="left")
    n_no_type = df["stype"].isna().sum()
    df["stype"] = df["stype"].fillna("UNKNOWN(gemma라벨없음)")
    print(f"유형 매칭: {len(df) - n_no_type}/{len(df)} (미매칭 {n_no_type})")

    # 3) total_copies = aug_mult (기본 2)
    augw = pd.read_csv(AUGW)
    augw_map = dict(zip(augw["Id"], augw["aug_mult"].astype(int)))
    df["total_copies"] = df["Id"].map(lambda i: augw_map.get(i, 2))

    # 4) 미학습 사본 수
    un = pd.read_csv(UNTRAINED)
    un_counts = un["id"].value_counts()
    df["untrained_copies"] = df["Id"].map(lambda i: int(un_counts.get(i, 0)))
    # 방어: 미학습이 total보다 많을 수 없음
    df["untrained_copies"] = df[["untrained_copies", "total_copies"]].min(axis=1)
    df["trained_copies"] = df["total_copies"] - df["untrained_copies"]

    def status(r):
        if r["trained_copies"] == 0:
            return "fully_untrained"
        if r["untrained_copies"] == 0:
            return "fully_trained"
        return "partial"

    df["status"] = df.apply(status, axis=1)

    # ---- 전체 요약 ----
    tot_copies = df["total_copies"].sum()
    tot_untr = df["untrained_copies"].sum()
    print("\n" + "=" * 64)
    print("[전체]")
    print(f"  증강 사본 총 {tot_copies}개 중 미학습 {tot_untr}개 "
          f"({tot_untr / tot_copies:.1%})  |  CSV 행수 {len(un)}")
    vc = df["status"].value_counts()
    for s in ["fully_trained", "partial", "fully_untrained"]:
        print(f"  {s:16s}: {int(vc.get(s, 0)):5d} ids")

    # ---- 유형별 요약 ----
    def agg(g):
        return pd.Series({
            "n_ids": len(g),
            "fully_untrained_ids": int((g["status"] == "fully_untrained").sum()),
            "partial_ids": int((g["status"] == "partial").sum()),
            "total_copies": int(g["total_copies"].sum()),
            "untrained_copies": int(g["untrained_copies"].sum()),
        })

    for col, label in [("stype", "유형(stype)"),
                       ("camera", "카메라 축"),
                       ("tag_no_marker", "시간표지 없음")]:
        t = df.groupby(col, dropna=False).apply(agg, include_groups=False)
        t["untrained_%"] = (t["untrained_copies"] / t["total_copies"] * 100).round(1)
        t["fully_untrained_%_of_ids"] = (
            t["fully_untrained_ids"] / t["n_ids"] * 100).round(1)
        t = t.sort_values("untrained_%", ascending=False)
        print("\n" + "=" * 64)
        print(f"[{label}]  — untrained_% = 그 유형 증강 사본 중 미학습 비율")
        print(t.to_string())
        if col == "stype":
            t.to_csv(OUT_TYPE, encoding="utf-8-sig")

    # ---- 저장 ----
    df.sort_values(["status", "stype", "untrained_copies"], ascending=[True, True, False]) \
      .to_csv(OUT_PERID, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 64)
    print(f"저장: {OUT_PERID}")
    print(f"저장: {OUT_TYPE}")

    # ---- 완전 미학습 id 목록(유형별 상위) ----
    fu = df[df["status"] == "fully_untrained"]
    if len(fu):
        print(f"\n[완전 미학습 id {len(fu)}개] 유형 분포:")
        print(fu["stype"].value_counts().to_string())


if __name__ == "__main__":
    main()
