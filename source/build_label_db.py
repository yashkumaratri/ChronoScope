#!/usr/bin/env python3
import argparse
from pathlib import Path

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a DuckDB label map from the Wikidata multilingual parquet."
    )
    parser.add_argument("--labels_parquet", required=True, help="Path to qid_labels_desc.parquet")
    parser.add_argument("--out_db", required=True, help="Output DuckDB path, e.g. labels_en.duckdb")
    parser.add_argument(
        "--lang",
        default="en",
        help="Language code to keep. Use '*' to keep all languages.",
    )
    parser.add_argument(
        "--table_name",
        default="labels",
        help="Output table name expected by downstream code.",
    )
    args = parser.parse_args()

    labels_parquet = Path(args.labels_parquet)
    out_db = Path(args.out_db)
    out_db.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(out_db))
    try:
        con.execute("PRAGMA threads=8")
        con.execute(f"DROP TABLE IF EXISTS {args.table_name}")

        cols = [x[0] for x in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{labels_parquet.as_posix()}')"
        ).fetchall()]
        lang_col = "lang" if "lang" in cols else ("len" if "len" in cols else None)
        desc_col = "description" if "description" in cols else ("des" if "des" in cols else None)

        select_cols = ["id", "label"]
        if lang_col:
            select_cols.append(f"{lang_col} AS lang")
        if desc_col:
            select_cols.append(f"{desc_col} AS description")

        where = ""
        if args.lang != "*":
            if lang_col is None:
                raise ValueError("No language column found in parquet, but --lang filtering was requested.")
            where = f"WHERE {lang_col} = '{args.lang}'"

        con.execute(
            f"""
            CREATE TABLE {args.table_name} AS
            SELECT {", ".join(select_cols)}
            FROM read_parquet('{labels_parquet.as_posix()}')
            {where}
            """
        )

        con.execute(f"CREATE INDEX {args.table_name}_id_idx ON {args.table_name}(id)")
        row_count = con.execute(f"SELECT COUNT(*) FROM {args.table_name}").fetchone()[0]
        print(f"[done] table={args.table_name} rows={row_count:,} db={out_db}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
