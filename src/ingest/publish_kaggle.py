"""Turn the collected shards into a Kaggle dataset version.

    python -m ingest.publish_kaggle --shards shards --dataset <owner>/<slug> [--create]

Run it from cron on the machine that hosts the collection server. Kaggle credentials come from
that machine's environment (``KAGGLE_USERNAME`` / ``KAGGLE_KEY``, or ``~/.kaggle/kaggle.json``);
nothing here writes them anywhere. Each publish is a new dataset *version* holding one gzip per
UTC day (``records-<day>.jsonl.gz``, every install's rows for that day). It always includes every
day, because a Kaggle version contains exactly the files in the folder it is given. What was
last published is remembered in ``<work>/published.json``; with nothing new, Kaggle is not called.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _days(shards: Path) -> dict[str, list[Path]]:
    days: dict[str, list[Path]] = {}
    for day in sorted(p for p in shards.iterdir() if p.is_dir()) if shards.exists() else []:
        files = sorted(day.glob("*.jsonl"))
        if files:
            days[day.name] = files
    return days


def _fingerprint(files: list[Path]) -> dict:
    rows = 0
    for f in files:
        with f.open("rb") as fh:
            rows += sum(1 for line in fh if line.strip())
    return {"files": {f.name: f.stat().st_size for f in files}, "rows": rows}


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def publish(shards: Path, work: Path, dataset: str, *, create: bool = False, license_name: str = "other",
            kaggle: str = "kaggle", note: str | None = None) -> dict:
    """Publish a new dataset version if there are rows not yet published; returns what happened."""
    shards, work = Path(shards), Path(work)
    manifest_path = work / "published.json"
    before = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    now = {day: _fingerprint(files) for day, files in _days(shards).items()}
    changed = [day for day, fp in now.items() if before.get(day) != fp]
    if not changed:
        return {"published": 0, "days": 0, "skipped": "nothing new"}

    folder = work / "dataset"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for day, files in _days(shards).items():
        with gzip.open(folder / f"records-{day}.jsonl.gz", "wb") as out:
            for f in files:
                out.write(f.read_bytes())
    _write_atomic(folder / "dataset-metadata.json", json.dumps({
        "title": dataset.split("/")[-1], "id": dataset, "licenses": [{"name": license_name}]}, indent=2))

    exe = shutil.which(kaggle)
    if exe is None:
        raise RuntimeError(f"the {kaggle!r} command was not found; pip install kaggle")
    command = [exe, "datasets", "create" if create else "version", "-p", str(folder)]
    if not create:
        command += ["-m", note or f"{sum(now[d]['rows'] for d in changed)} new records"]
    done = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if done.returncode != 0:
        raise RuntimeError(f"kaggle exited {done.returncode}: {(done.stderr or done.stdout)[-300:].strip()}")
    published = sum(max(0, now[d]["rows"] - before.get(d, {}).get("rows", 0)) for d in changed)
    _write_atomic(manifest_path, json.dumps(now, indent=1))
    return {"published": published, "days": len(changed)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ingest.publish_kaggle", description=__doc__.split("\n\n")[0])
    ap.add_argument("--shards", default=os.environ.get("INGEST_SHARD_DIR", "shards"))
    ap.add_argument("--work", default="kaggle-work", help="scratch folder and the record of what was published")
    ap.add_argument("--dataset", required=True, help="<owner>/<slug> of the Kaggle dataset")
    ap.add_argument("--create", action="store_true", help="the first publish: create the dataset")
    ap.add_argument("--license", default="other", help="Kaggle license name for the dataset metadata")
    ap.add_argument("--kaggle", default="kaggle", help="the kaggle CLI to call")
    args = ap.parse_args(argv)
    result = publish(Path(args.shards), Path(args.work), args.dataset, create=args.create,
                     license_name=args.license, kaggle=args.kaggle)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
