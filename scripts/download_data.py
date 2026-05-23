"""Download an open text corpus for pretraining LocalNeuro.

LocalNeuro ships no data of its own, but a richer base model benefits from
genuine text. The project explicitly allows open *text datasets*; this script
fetches one and saves it as a plain ``.txt`` file you can then feed to
``scripts/prepare_data.py`` via ``--input``. It uses only ``urllib`` -- no extra
dependencies, and no model weights are ever downloaded.

Examples
--------
    python scripts/download_data.py --name tinystories
    python scripts/download_data.py --name tinystories --max-mb 5
    python scripts/download_data.py --url https://example.com/corpus.txt \\
        --output datasets/mine.txt
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from localneuro.utils import configure_console, human_bytes

# Verified public download URLs for small, open text corpora. HuggingFace serves
# any dataset file as a raw download under the ``resolve/main/`` path.
KNOWN_DATASETS = {
    "tinystories": {
        "url": "https://huggingface.co/datasets/roneneldan/TinyStories/"
               "resolve/main/TinyStories-valid.txt",
        "description": "TinyStories validation split (~19 MB) -- short, simple "
                       "stories written for very small models to learn from.",
    },
}


def download(url: str, output: Path, max_bytes: int) -> int:
    """Stream ``url`` to ``output``, stopping early if ``max_bytes`` is reached."""
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "LocalNeuro"})
    written = 0
    chunk_size = 1 << 16
    with urllib.request.urlopen(request, timeout=60) as response:
        header_total = response.headers.get("Content-Length")
        total = int(header_total) if header_total else None
        with output.open("wb") as fh:
            while True:
                data = response.read(chunk_size)
                if not data:
                    break
                if max_bytes and written + len(data) >= max_bytes:
                    fh.write(data[: max_bytes - written])
                    written = max_bytes
                    print(f"\n  reached the {human_bytes(max_bytes)} size cap")
                    break
                fh.write(data)
                written += len(data)
                if total:
                    pct = 100.0 * written / total
                    print(f"\r  {human_bytes(written)} / {human_bytes(total)} "
                          f"({pct:5.1f}%)", end="", flush=True)
                else:
                    print(f"\r  {human_bytes(written)}", end="", flush=True)
    print()
    return written


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", choices=sorted(KNOWN_DATASETS),
                        help="download a known open dataset")
    parser.add_argument("--url", help="direct URL of a plain-text corpus")
    parser.add_argument("--output", default=None, help="output .txt path")
    parser.add_argument("--max-mb", type=float, default=0.0,
                        help="cap the download size in MB (0 = no cap)")
    args = parser.parse_args()

    if args.name:
        entry = KNOWN_DATASETS[args.name]
        url = entry["url"]
        output = Path(args.output or f"datasets/{args.name}.txt")
        print(entry["description"])
    elif args.url:
        if not args.output:
            raise SystemExit("--output is required when using --url")
        url = args.url
        output = Path(args.output)
    else:
        print("available datasets:")
        for key, entry in sorted(KNOWN_DATASETS.items()):
            print(f"  {key:14s} {entry['description']}")
        raise SystemExit("\npass --name <dataset>, or --url <URL> --output <path>")

    max_bytes = int(args.max_mb * 1024 * 1024) if args.max_mb > 0 else 0
    try:
        written = download(url, output, max_bytes)
    except Exception as exc:  # network / HTTP errors
        raise SystemExit(f"download failed: {exc}")

    print(f"saved {human_bytes(written)} -> {output}")
    print(f"next:  python scripts/prepare_data.py --input {output} "
          "--tokenizer checkpoints/tokenizer.json")


if __name__ == "__main__":
    main()
