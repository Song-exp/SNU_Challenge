# -*- coding: utf-8 -*-
"""977 중단 학습의 미학습 데이터를 gemma 문장성분 기반 다축으로 검증.

가설(사용자): 4유형(sparse/dense x cam) 밖의 세분 유형이 통째로 학습에서 빠졌을 수 있다.
검증: 셔플은 rng.shuffle(무작위)이므로 어떤 축이든 미학습률 ~= 전역률(34.4%)이어야 한다.
     각 버킷의 편차가 이항분포 노이즈(±3σ)를 넘는지 z-score로 판정 → 넘으면 '실제 편향'.

미학습 = 증강 사본 단위. copy 단위 무작위 추출이므로
  expected_untrained = N_copies * p,  std = sqrt(N*p*(1-p)),  z=(obs-exp)/std.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import structure_features as sf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
UNTRAINED = "all_untrained_items_raw.csv"
TRAIN_CSV = "snuaichallenge_data/train.csv"
HOLDOUT = "splits/holdout_300.csv"
AUGW = "outputs/aug_weights_exp16.csv"

SEQ_MARKERS = ("then", "after", "before", "next", "finally", "first",
               "later", "once", "subsequently")
SIM_MARKERS = ("while", "as", "during", "simultaneously", "meanwhile")


def bucket_events(n):
    if n <= 1:
        return "ev_0-1"
    if n == 2:
        return "ev_2"
    if n == 3:
        return "ev_3"
    return "ev_4+"


def bucket_subj(n):
    return {0: "subj_0", 1: "subj_1", 2: "subj_2"}.get(n, "subj_3+")


def bucket_mark(n):
    return {0: "mark_0", 1: "mark_1"}.get(n, "mark_2+")


def bucket_wc(n):
    if n <= 10:
        return "wc_<=10"
    if n <= 18:
        return "wc_11-18"
    if n <= 28:
        return "wc_19-28"
    return "wc_29+"


def main():
    train = pd.read_csv(TRAIN_CSV)
    hold = set(pd.read_csv(HOLDOUT)["Id"])
    universe = set(train[~train["Id"].isin(hold)]["Id"])

    gem = sf.load_gemma_labels()
    gem = gem[gem["Id"].isin(universe)].copy()

    # ---- gemma 성분 기반 파생 축 ----
    gem["ev_bucket"] = gem["n_events_noncam"].map(bucket_events)
    gem["subj_bucket"] = gem["n_subj_noncam"].map(bucket_subj)
    gem["mark_bucket"] = gem["n_markers"].map(bucket_mark)
    gem["wc_bucket"] = gem["sentence"].map(lambda s: bucket_wc(len(str(s).split())))
    gem["axis_camera"] = gem["camera"].map({True: "camO", False: "camX"})
    gem["axis_viewer"] = gem["viewer"].map({True: "viewerY", False: "viewerN"})
    gem["axis_multisubj"] = (gem["n_subj_noncam"] >= 2).map({True: "multi", False: "single"})

    def marker_kind(mlist):
        ml = [str(m).lower() for m in (mlist or [])]
        txt = " ".join(ml)
        has_seq = any(k in txt for k in SEQ_MARKERS)
        has_sim = any(k in txt for k in SIM_MARKERS)
        if has_seq and has_sim:
            return "mk_mixed"
        if has_seq:
            return "mk_seq"
        if has_sim:
            return "mk_sim"
        return "mk_none"
    gem["marker_kind"] = gem["markers"].map(marker_kind)

    # 첫 사건 동사(러프): 첫 event의 첫 동사 후보 = 두번째 토큰 근방
    def first_verb(evlist):
        if not evlist:
            return "noevent"
        toks = re.findall(r"[a-z]+", str(evlist[0]).lower())
        # 주어(the/a X) 건너뛴 대략적 동사 위치
        for t in toks[1:4]:
            if t not in ("the", "a", "an", "of", "in", "on", "to", "and"):
                return t
        return toks[-1] if toks else "noevent"
    gem["first_verb"] = gem["events"].map(first_verb)

    # ---- copy 수 / 미학습 수 ----
    augw = pd.read_csv(AUGW)
    augw_map = dict(zip(augw["Id"], augw["aug_mult"].astype(int)))
    gem["total_copies"] = gem["Id"].map(lambda i: augw_map.get(i, 2))
    un = pd.read_csv(UNTRAINED)
    un_counts = un["id"].value_counts()
    gem["untrained_copies"] = gem["Id"].map(lambda i: int(un_counts.get(i, 0)))
    gem["untrained_copies"] = gem[["untrained_copies", "total_copies"]].min(axis=1)
    gem["trained_copies"] = gem["total_copies"] - gem["untrained_copies"]

    p = gem["untrained_copies"].sum() / gem["total_copies"].sum()
    print(f"전역 미학습률 p = {p:.4f}  (사본 {int(gem['total_copies'].sum())}개)")
    print("각 버킷: untrained_% 가 34.4%에서 얼마나 벗어나는지 z로 판정 (|z|>3 = 노이즈 초과)\n")

    def axis_report(col):
        rows = []
        for name, g in gem.groupby(col):
            N = int(g["total_copies"].sum())
            u = int(g["untrained_copies"].sum())
            exp = N * p
            std = np.sqrt(N * p * (1 - p))
            z = (u - exp) / std if std > 0 else 0.0
            # 완전미학습(×2 한정, 순수 무작위 기대 p^2)
            g2 = g[g["total_copies"] == 2]
            fu2 = int((g2["trained_copies"] == 0).sum())
            n2 = len(g2)
            rows.append(dict(bucket=name, n_ids=len(g), copies=N,
                             untrained_pct=round(u / N * 100, 1), z=round(z, 1),
                             fully_untr_x2=f"{fu2}/{n2}" if n2 else "-",
                             fu2_pct=round(fu2 / n2 * 100, 1) if n2 else None))
        t = pd.DataFrame(rows).sort_values("z", key=abs, ascending=False)
        flag = t[t["z"].abs() > 3]
        print(f"[{col}]")
        print(t.to_string(index=False))
        print(f"  → |z|>3 (실제 편향 의심) 버킷: {len(flag)}개"
              + (f" ⚠️ {list(flag['bucket'])}" if len(flag) else " ✅ 없음") + "\n")
        return t

    axes = ["ev_bucket", "subj_bucket", "mark_bucket", "wc_bucket",
            "axis_camera", "axis_viewer", "axis_multisubj", "marker_kind"]
    for a in axes:
        axis_report(a)

    # first_verb: 희소 버킷 많음 → n_ids>=30만
    vt = gem.groupby("first_verb").agg(
        n_ids=("Id", "size"), copies=("total_copies", "sum"),
        untr=("untrained_copies", "sum")).reset_index()
    vt = vt[vt["n_ids"] >= 30].copy()
    vt["untrained_pct"] = (vt["untr"] / vt["copies"] * 100).round(1)
    vt["z"] = ((vt["untr"] - vt["copies"] * p) /
               np.sqrt(vt["copies"] * p * (1 - p))).round(1)
    vt = vt.sort_values("z", key=abs, ascending=False)
    print("[first_verb] (n_ids>=30 동사만, 상위 편차)")
    print(vt.head(15).to_string(index=False))
    n_flag = (vt["z"].abs() > 3).sum()
    print(f"  → |z|>3: {n_flag}개" + (" ✅ 없음" if n_flag == 0 else " ⚠️ 확인요망"))

    out = "outputs/untrained_extended_firstverb.csv"
    vt.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
