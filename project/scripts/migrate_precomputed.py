#!/usr/bin/env python3
"""Pack / unpack a portable precomputed + workspaces.db bundle.

Archive layout (tar.gz)::

    precomputed/          # teaching parquets, manifest, grid_results, runs/
    workspaces.db         # Streamlit SQLite

Typical migration (this machine)::

    uv run python scripts/migrate_precomputed.py pack \\
      --source data/precomputed --db data/workspaces.db \\
      --archive /tmp/precomputed_bundle.tar.gz

    uv run python scripts/migrate_precomputed.py unpack \\
      --archive /tmp/precomputed_bundle.tar.gz \\
      --dest cache/precomputed --db data/workspaces.db

On unpack, rewrites ``data/precomputed`` → ``cache/precomputed`` in
manifest.json, grid_results.parquet ``run_dir``, and workspaces.db JSON.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from pathlib import Path

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "project").is_dir()),
    None,
)
if _REPO_ROOT is not None and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

OLD_PREFIX = "data/precomputed"
NEW_PREFIX = "cache/precomputed"
ARCHIVE_PRECOMPUTED = "precomputed"
ARCHIVE_DB = "workspaces.db"


def _die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _require_path(path: Path, *, kind: str) -> Path:
    if not path.exists():
        _die(f"{kind} not found: {path}")
    return path.resolve()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _rewrite_text(value: str) -> str:
    """Rewrite legacy path prefixes (posix + windows separators)."""
    out = value
    # Absolute / relative posix
    out = out.replace(OLD_PREFIX, NEW_PREFIX)
    # Windows-style separators inside stored strings
    out = out.replace(OLD_PREFIX.replace("/", "\\"), NEW_PREFIX.replace("/", "\\"))
    return out


def _rewrite_json_value(obj):
    if isinstance(obj, str):
        return _rewrite_text(obj)
    if isinstance(obj, list):
        return [_rewrite_json_value(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _rewrite_json_value(v) for k, v in obj.items()}
    return obj


def rewrite_manifest(path: Path) -> int:
    if not path.is_file():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    rewritten = _rewrite_json_value(data)
    path.write_text(json.dumps(rewritten, indent=2) + "\n", encoding="utf-8")
    n = 0
    for entry in rewritten.get("files") or []:
        if isinstance(entry, dict) and str(entry.get("path", "")).startswith(NEW_PREFIX):
            n += 1
    print(f"  manifest.json: rewrote paths ({n} file entries)")
    return 1


def rewrite_grid_results(path: Path) -> int:
    if not path.is_file():
        return 0
    import pandas as pd

    df = pd.read_parquet(path)
    if "run_dir" not in df.columns:
        print("  grid_results.parquet: no run_dir column, skip")
        return 0
    before = df["run_dir"].astype(str)
    df["run_dir"] = before.map(_rewrite_text)
    changed = int((before != df["run_dir"].astype(str)).sum())
    df.to_parquet(path, index=False)
    print(f"  grid_results.parquet: rewrote {changed} run_dir values")
    return changed


def rewrite_workspaces_db(db_path: Path) -> int:
    if not db_path.is_file():
        return 0
    conn = sqlite3.connect(str(db_path))
    try:
        updated = 0
        for table, cols in (
            ("workspaces", ("config",)),
            ("experiments", ("config", "metrics")),
        ):
            rows = conn.execute(f"SELECT id, {', '.join(cols)} FROM {table}").fetchall()
            for row in rows:
                row_id = row[0]
                new_vals = []
                changed = False
                for raw in row[1:]:
                    text = raw or ""
                    if not text:
                        new_vals.append(text)
                        continue
                    try:
                        parsed = json.loads(text)
                        rewritten = _rewrite_json_value(parsed)
                        new_text = json.dumps(rewritten)
                    except json.JSONDecodeError:
                        new_text = _rewrite_text(text)
                    if new_text != text:
                        changed = True
                    new_vals.append(new_text)
                if changed:
                    assignments = ", ".join(f"{c} = ?" for c in cols)
                    conn.execute(
                        f"UPDATE {table} SET {assignments} WHERE id = ?",
                        (*new_vals, row_id),
                    )
                    updated += 1
        conn.commit()
    finally:
        conn.close()
    print(f"  {db_path.name}: updated {updated} row(s)")
    return updated


def rewrite_dest(dest: Path, db_path: Path) -> None:
    print("Rewriting paths data/precomputed → cache/precomputed …")
    rewrite_manifest(dest / "manifest.json")
    rewrite_grid_results(dest / "grid_results.parquet")
    rewrite_workspaces_db(db_path)


def cmd_pack(args: argparse.Namespace) -> None:
    source = _require_path(Path(args.source), kind="source")
    if not source.is_dir():
        _die(f"--source must be a directory: {source}")
    db_path = _require_path(Path(args.db), kind="db")
    if not db_path.is_file():
        _die(f"--db must be a file: {db_path}")
    archive = Path(args.archive)
    if archive.exists() and not args.force:
        _die(f"archive already exists (use --force): {archive}")
    _ensure_parent(archive.resolve())

    print(f"Packing {source} + {db_path} → {archive}")
    mode = "w:gz"
    with tarfile.open(archive, mode) as tar:
        tar.add(source, arcname=ARCHIVE_PRECOMPUTED)
        tar.add(db_path, arcname=ARCHIVE_DB)
    size_mb = archive.stat().st_size / (1024 * 1024)
    print(f"Wrote {archive} ({size_mb:.1f} MiB)")


def _member_under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + "/")


def cmd_unpack(args: argparse.Namespace) -> None:
    archive = _require_path(Path(args.archive), kind="archive")
    dest = Path(args.dest)
    db_path = Path(args.db)

    if dest.exists() and any(dest.iterdir()) and not args.force:
        _die(f"--dest exists and is not empty (use --force): {dest}")
    if db_path.exists() and not args.force:
        _die(f"--db already exists (use --force): {db_path}")

    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
        if ARCHIVE_DB not in names:
            _die(f"archive missing top-level {ARCHIVE_DB!r}")
        if not any(_member_under(n, ARCHIVE_PRECOMPUTED) for n in names):
            _die(f"archive missing {ARCHIVE_PRECOMPUTED}/ tree")

        with tempfile.TemporaryDirectory(prefix="migrate_precomputed_") as tmp:
            tmp_path = Path(tmp)
            print(f"Extracting {archive} …")
            # filter='data' blocks path-traversal members (Py3.12+)
            try:
                tar.extractall(tmp_path, filter="data")
            except TypeError:
                tar.extractall(tmp_path)

            src_pre = tmp_path / ARCHIVE_PRECOMPUTED
            src_db = tmp_path / ARCHIVE_DB
            if not src_pre.is_dir():
                _die(f"extracted tree missing directory: {src_pre}")
            if not src_db.is_file():
                _die(f"extracted tree missing file: {src_db}")

            if dest.exists():
                if args.force:
                    print(f"Removing existing dest: {dest}")
                    shutil.rmtree(dest)
                else:
                    _die(f"--dest exists (use --force): {dest}")
            _ensure_parent(dest.resolve())
            print(f"Moving precomputed → {dest}")
            shutil.move(str(src_pre), str(dest))

            _ensure_parent(db_path.resolve())
            if db_path.exists() and args.force:
                print(f"Overwriting db: {db_path}")
                db_path.unlink()
            print(f"Moving workspaces.db → {db_path}")
            shutil.move(str(src_db), str(db_path))

    rewrite_dest(dest.resolve(), db_path.resolve())
    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    pack = sub.add_parser("pack", help="Create tar.gz from --source + --db")
    pack.add_argument("--source", required=True, help="precomputed directory to pack")
    pack.add_argument("--db", required=True, help="workspaces.db path to pack")
    pack.add_argument("--archive", required=True, help="output .tar.gz path")
    pack.add_argument(
        "--force",
        action="store_true",
        help="overwrite archive if it already exists",
    )
    pack.set_defaults(func=cmd_pack)

    unpack = sub.add_parser(
        "unpack",
        help="Extract tar.gz into --dest + --db and rewrite path prefixes",
    )
    unpack.add_argument("--archive", required=True, help="input .tar.gz path")
    unpack.add_argument("--dest", required=True, help="destination precomputed dir")
    unpack.add_argument("--db", required=True, help="destination workspaces.db path")
    unpack.add_argument(
        "--force",
        action="store_true",
        help="overwrite --dest / --db if they already exist",
    )
    unpack.set_defaults(func=cmd_unpack)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
