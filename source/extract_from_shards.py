
import argparse, json, io
from pathlib import Path
import zstandard as zstd
import duckdb
import polars as pl
import re
from datetime import date as Date

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
        if not (1 <= month <= 12): month = 1
        if not (1 <= day <= 31): day = 1
        return Date(year, month, day).isoformat()
    except Exception:
        return None

def extract_claim_value_qid(snak):
    if not snak or snak.get("snaktype") != "value":
        return None
    dv = snak.get("datavalue", {})
    val = dv.get("value")
    if isinstance(val, dict) and val.get("entity-type") == "item":
        nid = val.get("numeric-id")
        if isinstance(nid, int):
            return f"Q{nid}"
    return None

def extract_qual_time(claim, qualifier_pid: str):
    quals = claim.get("qualifiers") or {}
    snaks = quals.get(qualifier_pid) or []
    for s in snaks:
        if s.get("snaktype") != "value":
            continue
        dv = s.get("datavalue", {})
        val = dv.get("value")
        if isinstance(val, dict) and "time" in val:
            d = parse_wikidata_time(val.get("time"))
            if d is not None:
                return d
    return None

def labels_exist_batch(con, qids, chunk=50000):
    qids = list(qids)
    out = set()
    for i in range(0, len(qids), chunk):
        chunk_q = qids[i:i+chunk]
        vals = ",".join([f"('{q}')" for q in chunk_q])
        rows = con.execute(f"SELECT id FROM (VALUES {vals}) v(id) JOIN labels USING(id)").fetchall()
        out.update(r[0] for r in rows)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--pids", type=str, required=True)  # comma-separated
    ap.add_argument("--label_db", type=str, required=True)
    ap.add_argument("--shard", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    year = args.year
    wanted = set([p.strip() for p in args.pids.split(",") if p.strip()])
    shard_path = Path(args.shard)
    out_dir = Path(args.outdir); out_dir.mkdir(parents=True, exist_ok=True)

    # Read shard
    dctx = zstd.ZstdDecompressor()
    rows = []
    subj = []
    vals = []

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
                subject_qid = ent.get("id")
                if not (isinstance(subject_qid, str) and subject_qid.startswith("Q")):
                    continue
                claims = ent.get("claims") or {}
                for pid in wanted:
                    for claim in claims.get(pid, []) or []:
                        rank = claim.get("rank", "normal")
                        if rank == "deprecated":
                            continue
                        value_qid = extract_claim_value_qid(claim.get("mainsnak"))
                        if value_qid is None:
                            continue
                        start = extract_qual_time(claim, "P580")
                        end = extract_qual_time(claim, "P582")
                        pit = extract_qual_time(claim, "P585")
                        r = {
                            "snapshot_year": year,
                            "subject_qid": subject_qid,
                            "pid": pid,
                            "value_qid": value_qid,
                            "start_time": start,
                            "end_time": end,
                            "point_in_time": pit,
                            "rank": rank,
                        }
                        rows.append(r)
                        subj.append(subject_qid)
                        vals.append(value_qid)

    if not rows:
        print("No rows in shard for requested PIDs:", wanted)
        return

    con = duckdb.connect(str(args.label_db), read_only=True)
    try:
        subj_ok = labels_exist_batch(con, set(subj))
        val_ok  = labels_exist_batch(con, set(vals))
    finally:
        con.close()

    rows = [r for r in rows if (r["subject_qid"] in subj_ok and r["value_qid"] in val_ok)]
    if not rows:
        print("All rows filtered out due to missing English labels.")
        return

    df = pl.DataFrame(rows)
    out_path = out_dir / f"claims-{year}-{shard_path.stem}.parquet"
    df.write_parquet(out_path)
    print("Wrote", out_path, "rows", df.height)

if __name__ == "__main__":
    main()
