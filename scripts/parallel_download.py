#!/usr/bin/env python3
"""Resumable parallel HTTP range download (Zenodo serves ~1-3 MB/s per connection).

    parallel_download.py URL OUT SIZE [--workers 8] [--chunk-mb 64]

Writes OUT.part (preallocated) plus OUT.part.done (finished chunk indices), so an interrupted
download resumes where it stopped. Checksum verification is left to the caller.
"""
import argparse
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch(url, start, end, path, retries=8):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise IOError(f"expected 206, got {r.status}")
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read {len(data)} != {end - start + 1}")
            with open(path, "r+b") as f:
                f.seek(start)
                f.write(data)
            return len(data)
        except Exception as exc:  # noqa: BLE001 - retry any network error
            if attempt == retries - 1:
                raise
            time.sleep(min(60, 5 * 2 ** attempt))
            print(f"retry {start}: {exc}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("size", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=64)
    a = ap.parse_args()
    part, done_path = a.out + ".part", a.out + ".part.done"
    chunk = a.chunk_mb * 2 ** 20
    n = (a.size + chunk - 1) // chunk
    if not os.path.exists(part) or os.path.getsize(part) != a.size or not os.path.exists(done_path):
        with open(part, "wb") as f:
            f.truncate(a.size)
        open(done_path, "w").close()
    done = {int(x) for x in open(done_path).read().split()}
    todo = [i for i in range(n) if i not in done]
    print(f"{len(done)}/{n} chunks already done; {len(todo)} to fetch with {a.workers} workers", flush=True)
    lock, t0, got = threading.Lock(), time.time(), 0
    with ThreadPoolExecutor(a.workers) as pool, open(done_path, "a") as log:
        futs = {pool.submit(fetch, a.url, i * chunk, min(a.size, (i + 1) * chunk) - 1, part): i for i in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            got += fut.result()
            with lock:
                log.write(f"{futs[fut]}\n")
                log.flush()
            if k % 10 == 0 or k == len(todo):
                rate = got / (time.time() - t0) / 2 ** 20
                print(f"{len(done) + k}/{n} chunks, {rate:.1f} MB/s, eta {(len(todo) - k) * chunk / 2 ** 20 / max(rate, 1e-6) / 60:.0f} min",
                      flush=True)
    os.replace(part, a.out)
    os.remove(done_path)


if __name__ == "__main__":
    main()
