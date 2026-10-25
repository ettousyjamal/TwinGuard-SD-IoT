"""Step 0: download the evaluation dataset from Zenodo and verify its checksums.

Dataset: Sanz Rodrigo et al., "Network Digital Twin-Generated Dataset for Machine Learning-based Smart
QoS-aware Zero-Touch Traffic Engineering", Zenodo (2025), doi:10.5281/zenodo.17255307, CC-BY 4.0.

Downloads the six CSV files (plain and weighted labels of D1, D2, D3) into data/zenodo_17255307/ and checks
each file against the MD5 checksum published by Zenodo. Files that are already present and valid are not
downloaded again. Standard library only.

Usage (from the repository root):  python code/00_download_data.py
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RECORD = "17255307"
FILES = [f"dataset_0{i}_TC33_{d}_labeled{v}.csv"
         for i, d in [(1, "27062025"), (2, "02072025"), (3, "05072025")] for v in ("", "_weighted")]
DEST = Path(__file__).resolve().parents[1] / "data" / "zenodo_17255307"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{RECORD}", timeout=60) as r:
        published = {f["key"]: f["checksum"].split(":", 1)[1] for f in json.load(r)["files"]}
    missing = [n for n in FILES if n not in published]
    if missing:
        sys.exit(f"files not found in the Zenodo record: {missing}")
    lines = []
    for name in FILES:
        path = DEST / name
        if not (path.exists() and md5(path) == published[name]):
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(f"https://zenodo.org/api/records/{RECORD}/files/{name}/content", path)
        got = md5(path)
        if got != published[name]:
            sys.exit(f"checksum mismatch for {name}: {got} != {published[name]}")
        print(f"{name}: checksum OK")
        lines.append(f"{name} {got}")
    (DEST / "checksums_zenodo.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
