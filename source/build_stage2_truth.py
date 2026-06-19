#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from datetime import date as Date
import re
import polars as pl
from tqdm import tqdm

ROOT = Path("/scratch/bkr3as/wikidata_scope_benchmark")
STAGE1 = ROOT / "stage1_claims_full"
OUT = ROOT / "stage2_truth"
OUT.mkdir(parents=True, exist_ok=True)

DROP_SELF_LOOPS = True  # drop subject_qid == value_qid (recommended)

_time_re = re.compile(r"^([+-]?\d{1,})(?:-(\d{2})(?:-(\d{2}))?)?")

def same_year(date_str: str, anchor_iso: str) -> bool:
    """
    Returns True if date_str and anchor_iso are in the same year.
    """
    if date_str is None:
        return False
    try:
        return int(date_str[:4]) == int(anchor_iso[:4])
    except Exception:
        return False


def parse_wikidata_time_like(s: str):
    """
    Accepts ISO-like 'YYYY-MM-DD' (what you stored) or Wikidata '+YYYY-MM-DDT..' style.
    Returns Date or None.
    """
    if s is None:
        return None
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    if "T" in s:
        s = s.split("T", 1)[0]
    if s.startswith("+"):
        s = s[1:]
    m = _time_re.match(s)
    if not m:
        return None
    y, mm, dd = m.group(1), m.group(2), m.group(3)
    try:
        year = int(y)
        month = int(mm) if mm and mm != "00" else 1
        day = int(dd) if dd and dd != "00" else 1
        if not (1 <= month <= 12):
            month = 1
        if not (1 <= day <= 31):
            day = 1
        return Date(year, month, day)
    except Exception:
        return None

def first_date_from_json_list(js: str):
    """
    Your stage1 qualifier columns are JSON strings like '["1957-03-25"]' or None.
    This returns the minimum date in the list (as ISO string) or None.
    """
    if js is None:
        return None
    try:
        arr = json.loads(js)
        if not isinstance(arr, list) or not arr:
            return None
        # pick earliest date in the list for stability
        dates = []
        for x in arr:
            d = parse_wikidata_time_like(x)
            if d is not None:
                dates.append(d)
        if not dates:
            return None
        return min(dates).isoformat()
    except Exception:
        return None

def last_date_from_json_list(js: str):
    if js is None:
        return None
    try:
        arr = json.loads(js)
        if not isinstance(arr, list) or not arr:
            return None
        dates = []
        for x in arr:
            d = parse_wikidata_time_like(x)
            if d is not None:
                dates.append(d)
        if not dates:
            return None
        return max(dates).isoformat()
    except Exception:
        return None

def build_truth_for_year(year: int, anchors: list[str], threads: int):
    in_dir = STAGE1 / f"snapshot_year={year}"
    assert in_dir.exists(), f"Missing stage1: {in_dir}"

    out_dir = OUT / f"snapshot_year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pl.Config.set_tbl_rows(8)
    os.environ["POLARS_MAX_THREADS"] = str(max(1, threads))

    # Scan all parquet parts
    lf = pl.scan_parquet(str(in_dir / "*.parquet"))

    # Basic filters
    lf = lf.filter(pl.col("rank") != "deprecated")

    if DROP_SELF_LOOPS:
        lf = lf.filter(pl.col("subject_qid") != pl.col("value_qid"))

    # Parse qualifier JSON lists into canonical single dates (strings)
    # We keep:
    # - start_date = earliest start
    # - end_date = latest end
    # - pit_date = earliest point-in-time
    # This is stable and deterministic.
    def map_first(expr):
        return expr.map_elements(first_date_from_json_list, return_dtype=pl.Utf8)

    def map_last(expr):
        return expr.map_elements(last_date_from_json_list, return_dtype=pl.Utf8)

    lf = lf.with_columns([
        map_first(pl.col("q_start_time")).alias("start_date"),
        map_last(pl.col("q_end_time")).alias("end_date"),
        map_first(pl.col("q_point_in_time")).alias("pit_date"),
    ])

    # Temporal flag
    lf = lf.with_columns(
        (pl.col("start_date").is_not_null() | pl.col("end_date").is_not_null() | pl.col("pit_date").is_not_null()).alias("has_time")
    )

    # Rank priority: preferred > normal
    lf = lf.with_columns(
        pl.when(pl.col("rank") == "preferred").then(2)
          .when(pl.col("rank") == "normal").then(1)
          .otherwise(0).alias("rank_score")
    )

    temporal_cols = [
        "snapshot_year","anchor_date",
        "subject_qid","pid","value_qid","rank","rank_score",
        "start_date","end_date","pit_date",
        "q_follows","q_followed_by","q_replaces","q_replaced_by","q_series_ordinal",
        "q_of_applies_to","q_part_of","q_start_cause","q_end_cause",
        "q_significant_event","q_participant_in_qual",
    ]
    timeless_cols = [
        "snapshot_year","anchor_date",
        "subject_qid","pid","value_qid","rank","rank_score",
    ]

    # For each anchor, compute active rows and then deduplicate by (subject,pid) picking top rank_score.
    manifest = {"year": year, "anchors": anchors, "outputs": []}

    for a in tqdm(anchors, desc=f"year {year} anchors"):
        anchor = parse_wikidata_time_like(a)
        assert anchor is not None, f"Bad anchor: {a}"
        anchor_iso = anchor.isoformat()

        # Active logic:
        # - If pit_date exists: require pit_date == anchor (strict)
        # - Else interval: start_date <= anchor and (end_date is null or end_date >= anchor)
        # - If no time info: timeless, handled separately
        active_temporal = lf.filter(pl.col("has_time")).filter(
            pl.when(pl.col("pit_date").is_not_null())
              .then(
                  pl.col("pit_date").map_elements(
                      lambda d: same_year(d, anchor_iso),
                      return_dtype=pl.Boolean
                  )
              )
              .otherwise(
                  (pl.col("start_date").is_null() | (pl.col("start_date") <= pl.lit(anchor_iso))) &
                  (pl.col("end_date").is_null() | (pl.col("end_date") >= pl.lit(anchor_iso)))
              )
        )


        temporal_active_all = (
            active_temporal
            .with_columns([
                pl.lit(year).alias("snapshot_year"),
                pl.lit(anchor_iso).alias("anchor_date"),
            ])
            .select(temporal_cols)
        )

        # Pick best statement per (subject,pid) at this anchor
        temporal_best = (
            active_temporal
            .sort(["subject_qid","pid","rank_score"], descending=[False,False,True])
            .unique(subset=["subject_qid","pid"], keep="first")
            .with_columns([
                pl.lit(year).alias("snapshot_year"),
                pl.lit(anchor_iso).alias("anchor_date"),
            ])
            .select(temporal_cols)
        )

        temporal_all_path = out_dir / f"truth_temporal_all_active_anchor={anchor_iso}.parquet"
        temporal_path = out_dir / f"truth_temporal_anchor={anchor_iso}.parquet"
        temporal_active_all.collect(streaming=True).write_parquet(temporal_all_path)
        temporal_best.collect(streaming=True).write_parquet(temporal_path)

        # Timeless: no time qualifiers at all
        timeless_active = lf.filter(~pl.col("has_time"))

        timeless_all = (
            timeless_active
            .with_columns([
                pl.lit(year).alias("snapshot_year"),
                pl.lit(anchor_iso).alias("anchor_date"),
            ])
            .select(timeless_cols)
        )

        timeless_best = (
            timeless_active
            .sort(["subject_qid","pid","rank_score"], descending=[False,False,True])
            .unique(subset=["subject_qid","pid"], keep="first")
            .with_columns([
                pl.lit(year).alias("snapshot_year"),
                pl.lit(anchor_iso).alias("anchor_date"),
            ])
            .select(timeless_cols)
        )

        timeless_all_path = out_dir / f"truth_timeless_all_active_anchor={anchor_iso}.parquet"
        timeless_path = out_dir / f"truth_timeless_anchor={anchor_iso}.parquet"
        timeless_all.collect(streaming=True).write_parquet(timeless_all_path)
        timeless_best.collect(streaming=True).write_parquet(timeless_path)

        manifest["outputs"].append({
            "anchor": anchor_iso,
            "truth_temporal_all_active": str(temporal_all_path),
            "truth_temporal": str(temporal_path),
            "truth_timeless_all_active": str(timeless_all_path),
            "truth_timeless": str(timeless_path),
        })

    # Write manifest
    manifest_path = out_dir / "stage2_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[done] year={year} wrote={len(anchors)} anchors manifest={manifest_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=str, default="2022,2023,2024,2025")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "16")))
    ap.add_argument("--anchors", type=str, default="01-01,04-01,07-01,10-01")
    args = ap.parse_args()

    years = [int(x) for x in args.years.split(",") if x.strip()]
    anchors_mmdd = [x.strip() for x in args.anchors.split(",") if x.strip()]

    # Build anchors per year in ISO
    for y in years:
        anchors = [f"{y}-{mmdd}" for mmdd in anchors_mmdd]
        build_truth_for_year(y, anchors, threads=args.threads)

if __name__ == "__main__":
    main()
