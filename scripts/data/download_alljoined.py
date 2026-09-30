"""Safely inspect and download the official Alljoined-1.6M Hugging Face dataset.

The script defaults to metadata-only operations. A full download requires an
explicit --full flag and passes a disk-space guard.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO_ID = "Alljoined/Alljoined-1.6M"
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "Alljoined-1.6M"
MANIFEST_PATH = ROOT / "data" / "metadata" / "alljoined_hf_manifest.json"
DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

METADATA_PATTERNS = [
    ".gitattributes",
    "*.md",
    "*.txt",
    "*.json",
    "*.jsonl",
    "*.csv",
    "*.tsv",
    "*.parquet",
    "*.xlsx",
    "*.yaml",
    "*.yml",
    "*.py",
    "LICENSE*",
    "**/*.md",
    "**/*.txt",
    "**/*.json",
    "**/*.jsonl",
    "**/*.csv",
    "**/*.tsv",
    "**/*.parquet",
    "**/*.xlsx",
    "**/*.yaml",
    "**/*.yml",
]


def _get_json(url: str, timeout: int = 60) -> tuple[object, dict[str, str]]:
    request = Request(url, headers={"User-Agent": "ReLiC-SC/0.1"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
        headers = {key.lower(): value for key, value in response.headers.items()}
    return payload, headers


def _next_link(link_header: str | None, endpoint: str) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' not in part:
            continue
        url = part.split(";", 1)[0].strip().strip("<>")
        if endpoint != "https://huggingface.co":
            url = url.replace("https://huggingface.co", endpoint, 1)
        return url
    return None


def build_manifest(endpoint: str, revision: str) -> list[dict]:
    endpoint = endpoint.rstrip("/")
    info_url = f"{endpoint}/api/datasets/{REPO_ID}"
    info, _ = _get_json(info_url)
    resolved_commit = info.get("sha") if isinstance(info, dict) else None

    url = (
        f"{endpoint}/api/datasets/{REPO_ID}/tree/{quote(revision, safe='')}"
        "?recursive=true&expand=true&limit=100"
    )
    records: list[dict] = []
    while url:
        page, headers = _get_json(url)
        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected manifest response from {url}")
        for entry in page:
            if entry.get("type") != "file":
                continue
            records.append(
                {
                    "path": entry["path"],
                    "size": entry.get("size"),
                    "blob_id": entry.get("oid"),
                    "lfs": entry.get("lfs"),
                    "last_commit": entry.get("lastCommit"),
                    "type": entry.get("type"),
                }
            )
        url = _next_link(headers.get("link"), endpoint)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "repo_id": REPO_ID,
                "requested_revision": revision,
                "resolved_commit": resolved_commit,
                "transport_endpoint": endpoint,
                "files": records,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return records


def selected_files(records: list[dict], patterns: list[str]) -> list[dict]:
    return [
        row
        for row in records
        if row.get("size") is not None
        and any(fnmatch.fnmatch(row["path"], pattern) for pattern in patterns)
    ]


def human_gb(value: int) -> float:
    return round(value / (1024**3), 3)


def disk_guard(required_bytes: int, reserve_gb: float) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(RAW_DIR).free
    reserve = int(reserve_gb * 1024**3)
    print(f"Required: {human_gb(required_bytes)} GB")
    print(f"Free:     {human_gb(free)} GB")
    print(f"Reserve:  {reserve_gb:.1f} GB")
    if required_bytes > free - reserve:
        raise SystemExit(
            "Download blocked: selected files do not fit while preserving the reserve. "
            "Choose a smaller --include pattern or free disk space."
        )


def load_or_create_manifest(endpoint: str, revision: str) -> list[dict]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["files"]
    return build_manifest(endpoint, revision)


def command_manifest(args: argparse.Namespace) -> None:
    records = build_manifest(args.endpoint, args.revision)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    total = sum(row.get("size") or 0 for row in records)
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Commit:   {manifest.get('resolved_commit')}")
    print(f"Entries:  {len(records)}")
    print(f"Total known file size: {human_gb(total)} GB")
    for row in sorted(records, key=lambda x: x.get("size") or 0, reverse=True)[:20]:
        print(f"{human_gb(row.get('size') or 0):9.3f} GB  {row['path']}")


def _lfs_sha256(row: dict) -> str | None:
    lfs = row.get("lfs")
    if not isinstance(lfs, dict):
        return None
    oid = lfs.get("oid") or lfs.get("sha256")
    if isinstance(oid, str) and oid.startswith("sha256:"):
        oid = oid.split(":", 1)[1]
    return oid if isinstance(oid, str) and len(oid) == 64 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_one(row: dict, endpoint: str, revision: str) -> None:
    destination = RAW_DIR / Path(row["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(row["size"])
    expected_sha = _lfs_sha256(row)

    if destination.exists() and destination.stat().st_size == expected_size:
        if expected_sha is None or _sha256(destination) == expected_sha:
            print(f"Already complete: {row['path']}")
            return

    partial = destination.with_name(destination.name + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    encoded_path = quote(row["path"], safe="/")
    encoded_revision = quote(revision, safe="")
    url = (
        f"{endpoint.rstrip('/')}/datasets/{REPO_ID}/resolve/"
        f"{encoded_revision}/{encoded_path}?download=true"
    )
    headers = {"User-Agent": "ReLiC-SC/0.1"}
    # Always request a byte range. Large Xet/LFS responses can stall before
    # their first full-body chunk on some Windows networks, while ranged
    # responses stream immediately and remain resumable.
    headers["Range"] = f"bytes={downloaded}-"

    print(f"Downloading: {row['path']} ({human_gb(expected_size):.3f} GB)")
    request = Request(url, headers=headers)
    with urlopen(request, timeout=120) as response:
        append = downloaded > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)

    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {row['path']}: got {partial.stat().st_size}, "
            f"expected {expected_size}. Rerun to resume."
        )
    if expected_sha is not None and _sha256(partial) != expected_sha:
        raise RuntimeError(f"SHA-256 mismatch for {row['path']}")
    partial.replace(destination)
    print(f"Verified:   {row['path']}")


def command_download(args: argparse.Namespace) -> None:
    records = load_or_create_manifest(args.endpoint, args.revision)
    if args.full:
        patterns = ["*"]
    elif args.preset == "metadata":
        patterns = METADATA_PATTERNS
    elif args.include:
        patterns = args.include
    else:
        raise SystemExit(
            "Refusing an implicit full download. Use --preset metadata, one or more "
            "--include patterns, or the explicit --full flag."
        )

    chosen = selected_files(records, patterns)
    required = sum(row["size"] for row in chosen)
    if not chosen:
        raise SystemExit(
            "No manifest files matched the requested patterns. Inspect the manifest "
            "and choose an exact subject/shard pattern."
        )
    disk_guard(required, args.reserve_gb)
    print(f"Matched files: {len(chosen)}")
    for row in sorted(chosen, key=lambda x: x["path"])[:50]:
        print(f"  {row['path']}")

    ordered = sorted(chosen, key=lambda x: x["path"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = [
            pool.submit(_download_one, row, args.endpoint, args.revision)
            for row in ordered
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print(f"Download complete: {RAW_DIR}")


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description=__doc__)
    sub = top.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("manifest", help="fetch file names and sizes only")
    p_manifest.add_argument("--revision", default="main")
    p_manifest.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p_manifest.set_defaults(func=command_manifest)

    p_download = sub.add_parser("download", help="download a safe selected subset")
    p_download.add_argument("--preset", choices=["metadata"])
    p_download.add_argument("--include", action="append", default=[])
    p_download.add_argument("--full", action="store_true")
    p_download.add_argument("--revision", default="main")
    p_download.add_argument("--reserve-gb", type=float, default=10.0)
    p_download.add_argument("--max-workers", type=int, default=2)
    p_download.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p_download.set_defaults(func=command_download)
    return top


def main() -> int:
    args = parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("Interrupted; Hugging Face downloads can resume on the next run.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            f"Download operation failed: {exc.__class__.__name__}: {exc}\n"
            "The project files are intact. Check network access and rerun the same command; "
            "Hugging Face will reuse completed local files.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
