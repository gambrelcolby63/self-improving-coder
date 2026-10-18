"""Ingest a folder or a list of URLs into a local BM25 store (store/docs.json).

Every chunk keeps its source (path or URL) and the declared license, and a
SOURCES.md manifest is (re)written next to the store.

    python -m sic.ingest --folder seed_docs --license "PSF-2.0" --name python-docs
    python -m sic.ingest --urls urls.txt --license "MIT"
"""
from __future__ import annotations

import argparse
import json
import os

from .retrieval import BM25

TEXT_EXT = {".py", ".md", ".txt", ".rst", ".cpp", ".cc", ".h", ".hpp", ".v", ".sv", ".c", ".json", ".toml"}
DEFAULT_STORE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "store", "docs.json")


def chunk(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    paras, out, cur = text.split("\n\n"), [], ""
    for p in paras:
        if len(cur) + len(p) > size and cur:
            out.append(cur.strip())
            cur = cur[-overlap:]
        cur += p + "\n\n"
        while len(cur) > size * 2:
            out.append(cur[:size].strip())
            cur = cur[size - overlap:]
    if cur.strip():
        out.append(cur.strip())
    return out


def load_store(path: str = DEFAULT_STORE) -> BM25:
    return BM25.load(path) if os.path.exists(path) else BM25()


def ingest_folder(idx: BM25, folder: str, license: str, name: str) -> int:
    n = 0
    for root, _, files in os.walk(folder):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() not in TEXT_EXT:
                continue
            p = os.path.join(root, fn)
            with open(p, errors="replace") as f:
                text = f.read()
            for i, c in enumerate(chunk(text)):
                idx.add(c, {"source": os.path.relpath(p, folder), "collection": name,
                            "license": license, "chunk": i})
                n += 1
    return n


def ingest_urls(idx: BM25, urls: list[str], license: str, name: str) -> int:
    from .web import fetch_url
    n = 0
    for u in urls:
        r = fetch_url(u, max_chars=200_000)
        if r.get("error"):
            print("skip", u, r["error"])
            continue
        for i, c in enumerate(chunk(r["text"])):
            idx.add(c, {"source": u, "collection": name, "license": license, "chunk": i})
            n += 1
    return n


def write_manifest(idx: BM25, store_path: str):
    seen = {}
    for d in idx.docs:
        m = d["meta"]
        seen.setdefault((m.get("collection"), m.get("license")), set()).add(m.get("source"))
    lines = ["# Ingested sources\n", "| collection | license | #files/urls |", "|---|---|---|"]
    for (c, l), s in sorted(seen.items(), key=str):
        lines.append(f"| {c} | {l} | {len(s)} |")
    with open(os.path.join(os.path.dirname(store_path), "SOURCES.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder")
    ap.add_argument("--urls", help="text file, one URL per line")
    ap.add_argument("--license", required=True, help="license of the material (only ingest permissive)")
    ap.add_argument("--name", default="docs")
    ap.add_argument("--store", default=DEFAULT_STORE)
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(a.store), exist_ok=True)
    idx = BM25() if a.reset else load_store(a.store)
    n = 0
    if a.folder:
        n += ingest_folder(idx, a.folder, a.license, a.name)
    if a.urls:
        with open(a.urls) as f:
            n += ingest_urls(idx, [l.strip() for l in f if l.strip() and not l.startswith("#")], a.license, a.name)
    idx.save(a.store)
    write_manifest(idx, a.store)
    print(json.dumps({"added_chunks": n, "total_chunks": len(idx), "store": a.store}))


if __name__ == "__main__":
    main()
