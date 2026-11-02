"""Download the MSU/ORNL power-system attack dataset (three-class release).

Source page: https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets
(Tommy Morris, Uttam Adhikari, Shengyi Pan; Mississippi State University with
Oak Ridge National Laboratory). No registration is required. The page states
no explicit license; it asks users to cite the papers listed in README.md.
The raw files are NOT redistributed in this repo (data/ is gitignored).

The archive is treated as untrusted data: it is only checksum-verified and
extracted (CSV files), never executed.

Usage:  python scripts/download_data.py [--out data/raw]
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

BASE = "http://www.ece.uah.edu/~thm0009/icsdatasets/"
FILES = {
    # filename: sha256 (recorded when this repo was built, 2026-11-02)
    "triple.7z": "8bcf90a7b656cd3c73dd809a8b8b657444c5512731811875d2dbc6b44448331c",
    "PowerSystem_Dataset_README.pdf": "1b2cd6a1f43134097b960d6ddedb71762658b56389febef0b007415e7b8f4f26",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        dst = out / name
        if not dst.exists():
            print(f"downloading {BASE + name}")
            urllib.request.urlretrieve(BASE + name, dst)
        got = sha256(dst)
        if got != digest:
            raise SystemExit(f"checksum mismatch for {name}: {got}")
        print(f"ok  {name}  sha256={got[:16]}...")

    import py7zr

    target = out / "triple"
    if not (target / "data15.csv").exists():
        with py7zr.SevenZipFile(out / "triple.7z", "r") as z:
            names = z.getnames()
            bad = [n for n in names if not n.endswith(".csv") or "/" in n or "\\" in n or ".." in n]
            if bad:
                raise SystemExit(f"unexpected archive members: {bad}")
            z.extractall(target)
    print(f"extracted {len(list(target.glob('data*.csv')))} CSV files to {target}")


if __name__ == "__main__":
    main()
