#!/usr/bin/env python3
import os
import io
import json
import time
import re
import bz2
import gzip
import shutil
import subprocess
from pathlib import Path
from collections import Counter

import zstandard as zstd
import polars as pl
import duckdb
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date as Date

# ---------------------------
# Config (edit if needed)
# ---------------------------
ROOT = Path("/scratch/bkr3as/wikidata_scope_benchmark")
STAGE0 = ROOT / "stage0_jsonl"
OUT_STAGE1 = ROOT / "stage1_claims_full"
LABEL_DB = ROOT / "label_maps" / "labels_en.duckdb"
RAW_DUMPS = Path("/standard/hartvigsen_lab/yash/raw_dumps/wikidata_yearly")

FILTER_TO_EN_LABELS = True
DROP_DEPRECATED = True
ONLY_QID_MAIN_VALUE = True

FLUSH_ROWS = 400_000

def as_json_list_str(x):
    """
    Normalize to JSON string for list-like qualifier columns.
    Returns None or a JSON string like '["Q1","Q2"]'.
    """
    if x is None:
        return None
    if isinstance(x, list):
        return json.dumps(x, ensure_ascii=False)
    # Sometimes you may get a scalar due to edge cases; wrap it
    return json.dumps([x], ensure_ascii=False)


# Curated PID set (variety)
PIDS = sorted(set([
    # Temporal-relational core
    "P6","P35","P39","P108","P463","P102",
    "P169","P488","P127","P54","P286",
    "P131","P17","P276","P69","P551","P27",
    # Multi-hop / hierarchy
    "P31","P279","P361","P355","P749","P155","P156","P106","P101",
    # Optional expansion
    "P50","P161","P175","P1344","P166",
]))

# Qualifiers
QUAL_TIME = {"P580":"q_start_time", "P582":"q_end_time", "P585":"q_point_in_time"}
QUAL_ENTITY = {
    "P155":"q_follows", "P156":"q_followed_by",
    "P1365":"q_replaces", "P1366":"q_replaced_by",
    "P642":"q_of_applies_to", "P361":"q_part_of",
    "P1535":"q_start_cause", "P1534":"q_end_cause",
    "P793":"q_significant_event", "P1344":"q_participant_in_qual",
}
QUAL_ORDINAL = {"P1545":"q_series_ordinal"}

_time_re = re.compile(r"^([+-]?\d{1,})(?:-(\d{2})(?:-(\d{2}))?)?")

def parse_wikidata_time(time_str: str):
    if not time_str or not isinstance(time_str, str):
        return None
    m = _time_re.match(time_str.strip())
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

def mainsnak_value_qid(mainsnak):
    if not mainsnak or mainsnak.get("snaktype") != "value":
        return None
    dv = mainsnak.get("datavalue", {})
    val = dv.get("value")
    if isinstance(val, dict) and val.get("entity-type") == "item":
        nid = val.get("numeric-id")
        if isinstance(nid, int):
            return f"Q{nid}"
    return None

def qualifier_values(claim, pid: str):
    quals = claim.get("qualifiers") or {}
    return quals.get(pid) or []

def qual_time_list(claim, pid: str):
    out = []
    for s in qualifier_values(claim, pid):
        if s.get("snaktype") != "value":
            continue
        dv = s.get("datavalue", {})
        val = dv.get("value")
        if isinstance(val, dict) and "time" in val:
            d = parse_wikidata_time(val.get("time"))
            if d is not None:
                out.append(d.isoformat())
    return out or None

def qual_entity_list(claim, pid: str):
    out = []
    for s in qualifier_values(claim, pid):
        if s.get("snaktype") != "value":
            continue
        dv = s.get("datavalue", {})
        val = dv.get("value")
        if isinstance(val, dict) and val.get("entity-type") == "item":
            nid = val.get("numeric-id")
            if isinstance(nid, int):
                out.append(f"Q{nid}")
    return out or None

def qual_ordinal_list(claim, pid: str):
    out = []
    for s in qualifier_values(claim, pid):
        if s.get("snaktype") != "value":
            continue
        dv = s.get("datavalue", {})
        val = dv.get("value")
        if isinstance(val, str):
            out.append(val)
    return out or None

def open_label_db():
    con = duckdb.connect(str(LABEL_DB), read_only=True)
    return con

def filter_rows_by_labels(con, rows):
    if not rows:
        return rows

    subj = list({r["subject_qid"] for r in rows})
    val = list({r["value_qid"] for r in rows})

    def fetch_present(qids, chunk=40000):
        present = set()
        for i in range(0, len(qids), chunk):
            c = qids[i:i+chunk]
            if not c:
                continue
            vals = ",".join([f"('{q}')" for q in c])
            got = con.execute(
                f"SELECT v.id FROM (VALUES {vals}) v(id) JOIN labels USING(id)"
            ).fetchall()
            present.update(x[0] for x in got)
        return present

    subj_ok = fetch_present(subj)
    val_ok = fetch_present(val)
    return [r for r in rows if (r["subject_qid"] in subj_ok and r["value_qid"] in val_ok)]

def list_shards(year: int):
    d = STAGE0 / f"snapshot_year={year}"
    files = sorted(d.glob(f"entities-{year}-part-*.jsonl.zst"))
    if not files:
        raise FileNotFoundError(f"No shards found: {d}")
    return files


def find_raw_dump(year: int) -> Path:
    year_dir = RAW_DUMPS / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"No raw dump directory found for year={year}: {year_dir}")
    candidates = sorted(year_dir.glob("*/wikidata-*-all.json.*"))
    candidates = [p for p in candidates if p.suffix in {".bz2", ".gz"}]
    if not candidates:
        raise FileNotFoundError(f"No raw dump found under {year_dir}")
    # Prefer the most recent snapshot folder if multiple exist.
    return candidates[-1]


def iter_raw_dump_entities(raw_path: Path):
    suffix = raw_path.suffix.lower()
    proc = None

    def builtin_reader():
        if suffix == ".bz2":
            return bz2.open(raw_path, "rt", encoding="utf-8", errors="ignore")
        if suffix == ".gz":
            return gzip.open(raw_path, "rt", encoding="utf-8", errors="ignore")
        raise ValueError(f"Unsupported raw dump type: {raw_path}")

    if suffix == ".bz2":
        pbzip2 = shutil.which("pbzip2")
        bzip2 = shutil.which("bzip2")
        if pbzip2:
            proc = subprocess.Popen([pbzip2, "-dc", str(raw_path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        elif bzip2:
            proc = subprocess.Popen([bzip2, "-dc", str(raw_path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    elif suffix == ".gz":
        pigz = shutil.which("pigz")
        gzip_bin = shutil.which("gzip")
        if pigz:
            proc = subprocess.Popen([pigz, "-dc", str(raw_path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        elif gzip_bin:
            proc = subprocess.Popen([gzip_bin, "-dc", str(raw_path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    fh = io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="ignore") if proc is not None else builtin_reader()

    try:
        for line in fh:
            line = line.strip()
            if not line or line == "[" or line == "]":
                continue
            if line.endswith(","):
                line = line[:-1]
            if not line or line in {"[", "]"}:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue
    finally:
        fh.close()
        if proc is not None:
            proc.wait()

def shard_out_dir(year: int):
    d = OUT_STAGE1 / f"snapshot_year={year}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def process_one_shard(year: int, shard_path: Path, part_prefix: str, flush_rows: int, filter_to_en_labels: bool):
    t0 = time.time()
    dctx = zstd.ZstdDecompressor()
    wanted = set(PIDS)

    outdir = shard_out_dir(year)
    rows = []
    wrote_parts = 0
    entities_seen = 0
    claims_seen = 0
    kept_rows = 0

    con = open_label_db() if filter_to_en_labels and LABEL_DB.exists() else None

    def flush():
        QUAL_COLS = list(QUAL_TIME.values()) + list(QUAL_ENTITY.values()) + list(QUAL_ORDINAL.values())
        nonlocal rows, wrote_parts, kept_rows
        if not rows:
            return
        rows2 = filter_rows_by_labels(con, rows) if con is not None else rows
        if rows2:
            schema_overrides = {c: pl.Utf8 for c in QUAL_COLS}
            df = pl.from_dicts(rows2, schema_overrides=schema_overrides, strict=False)
            # df = pl.from_dicts(rows2, strict=False)
            outp = outdir / f"{part_prefix}.part-{wrote_parts:05d}.parquet"
            df.write_parquet(outp)
            kept_rows += df.height
            wrote_parts += 1
        rows = []

    with open(shard_path, "rb") as f:
        with dctx.stream_reader(f) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text:
                line = line.strip()
                if not line:
                    continue
                try:
                    ent = json.loads(line)
                except Exception:
                    continue

                subject = ent.get("id")
                if not (isinstance(subject, str) and subject.startswith("Q")):
                    continue

                claims = ent.get("claims") or {}
                entities_seen += 1

                for pid in wanted:
                    clist = claims.get(pid) or []
                    if not clist:
                        continue
                    for claim in clist:
                        rank = claim.get("rank", "normal")
                        if DROP_DEPRECATED and rank == "deprecated":
                            continue
                        value_qid = mainsnak_value_qid(claim.get("mainsnak"))
                        if ONLY_QID_MAIN_VALUE and value_qid is None:
                            continue
                        claims_seen += 1

                        rec = {
                            "snapshot_year": year,
                            "subject_qid": subject,
                            "pid": pid,
                            "value_qid": value_qid,
                            "rank": rank,
                        }
                        for qp, outk in QUAL_TIME.items():
                            rec[outk] = as_json_list_str(qual_time_list(claim, qp))
                        for qp, outk in QUAL_ENTITY.items():
                            rec[outk] = as_json_list_str(qual_entity_list(claim, qp))
                        for qp, outk in QUAL_ORDINAL.items():
                            rec[outk] = as_json_list_str(qual_ordinal_list(claim, qp))


                        rows.append(rec)
                        if len(rows) >= flush_rows:
                            flush()

    flush()
    if con is not None:
        con.close()

    return {
        "year": year,
        "shard": shard_path.name,
        "entities_seen": entities_seen,
        "claims_seen": claims_seen,
        "kept_rows": kept_rows,
        "parts_written": wrote_parts,
        "elapsed_sec": round(time.time() - t0, 2),
    }


def process_raw_dump(year: int, raw_path: Path, flush_rows: int, filter_to_en_labels: bool):
    t0 = time.time()
    wanted = set(PIDS)

    outdir = shard_out_dir(year)
    rows = []
    wrote_parts = 0
    entities_seen = 0
    claims_seen = 0
    kept_rows = 0

    con = open_label_db() if filter_to_en_labels and LABEL_DB.exists() else None

    def flush():
        QUAL_COLS = list(QUAL_TIME.values()) + list(QUAL_ENTITY.values()) + list(QUAL_ORDINAL.values())
        nonlocal rows, wrote_parts, kept_rows
        if not rows:
            return
        rows2 = filter_rows_by_labels(con, rows) if con is not None else rows
        if rows2:
            schema_overrides = {c: pl.Utf8 for c in QUAL_COLS}
            df = pl.from_dicts(rows2, schema_overrides=schema_overrides, strict=False)
            outp = outdir / f"raw-{year}.part-{wrote_parts:05d}.parquet"
            df.write_parquet(outp)
            kept_rows += df.height
            wrote_parts += 1
        rows = []

    for ent in tqdm(iter_raw_dump_entities(raw_path), desc=f"raw {year}", unit="entities"):
        subject = ent.get("id")
        if not (isinstance(subject, str) and subject.startswith("Q")):
            continue

        claims = ent.get("claims") or {}
        entities_seen += 1

        for pid in wanted:
            clist = claims.get(pid) or []
            if not clist:
                continue
            for claim in clist:
                rank = claim.get("rank", "normal")
                if DROP_DEPRECATED and rank == "deprecated":
                    continue
                value_qid = mainsnak_value_qid(claim.get("mainsnak"))
                if ONLY_QID_MAIN_VALUE and value_qid is None:
                    continue
                claims_seen += 1

                rec = {
                    "snapshot_year": year,
                    "subject_qid": subject,
                    "pid": pid,
                    "value_qid": value_qid,
                    "rank": rank,
                }
                for qp, outk in QUAL_TIME.items():
                    rec[outk] = as_json_list_str(qual_time_list(claim, qp))
                for qp, outk in QUAL_ENTITY.items():
                    rec[outk] = as_json_list_str(qual_entity_list(claim, qp))
                for qp, outk in QUAL_ORDINAL.items():
                    rec[outk] = as_json_list_str(qual_ordinal_list(claim, qp))

                rows.append(rec)
                if len(rows) >= flush_rows:
                    flush()

    flush()
    if con is not None:
        con.close()

    return {
        "year": year,
        "shard": raw_path.name,
        "source": "raw_dump",
        "raw_path": str(raw_path),
        "entities_seen": entities_seen,
        "claims_seen": claims_seen,
        "kept_rows": kept_rows,
        "parts_written": wrote_parts,
        "elapsed_sec": round(time.time() - t0, 2),
    }

def run_year(year: int, workers: int, source: str = "auto", filter_to_en_labels: bool = FILTER_TO_EN_LABELS):
    OUT_STAGE1.mkdir(parents=True, exist_ok=True)
    outdir = shard_out_dir(year)
    print("[run] outdir:", outdir)

    if filter_to_en_labels and not LABEL_DB.exists():
        print(f"[warn] label DB missing at {LABEL_DB}; continuing without Stage1 label filtering")
        filter_to_en_labels = False

    stats = []
    stage0_available = (STAGE0 / f"snapshot_year={year}").exists()
    use_stage0 = source == "stage0" or (source == "auto" and stage0_available)

    if use_stage0:
        shards = list_shards(year)
        print(f"[run] year={year} source=stage0 shards={len(shards)} workers={workers} flush_rows={FLUSH_ROWS}")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = []
            for sp in shards:
                prefix = sp.name.replace(".jsonl.zst", "")
                futures.append(ex.submit(process_one_shard, year, sp, prefix, FLUSH_ROWS, filter_to_en_labels))

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"year {year}"):
                stats.append(fut.result())
        source_used = "stage0"
        shard_count = len(shards)
    else:
        raw_path = find_raw_dump(year)
        print(f"[run] year={year} source=raw_dump path={raw_path} flush_rows={FLUSH_ROWS}")
        stats.append(process_raw_dump(year, raw_path, FLUSH_ROWS, filter_to_en_labels=filter_to_en_labels))
        source_used = "raw_dump"
        shard_count = 1

    # Write manifest per year
    total_kept = sum(s["kept_rows"] for s in stats)
    total_parts = sum(s["parts_written"] for s in stats)
    manifest = outdir / "stage1_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({
            "year": year,
            "pids": PIDS,
            "filter_to_en_labels": filter_to_en_labels,
            "drop_deprecated": DROP_DEPRECATED,
            "only_qid_main_value": ONLY_QID_MAIN_VALUE,
            "flush_rows": FLUSH_ROWS,
            "workers": workers,
            "source_used": source_used,
            "total_rows": total_kept,
            "total_parts": total_parts,
            "shards": shard_count,
            "jobs": stats,
        }, f, ensure_ascii=False, indent=2)

    print(f"[done] year={year} rows={total_kept:,} parts={total_parts:,} manifest={manifest}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "32")))
    ap.add_argument("--source", choices=["auto", "stage0", "raw"], default="auto")
    ap.add_argument("--no_filter_to_en_labels", action="store_true")
    args = ap.parse_args()
    run_year(
        args.year,
        args.workers,
        source=("raw_dump" if args.source == "raw" else args.source),
        filter_to_en_labels=(not args.no_filter_to_en_labels),
    )
