"""Download a restaurant clip from Pexels (free licence) into data/.

Needs a free Pexels API key: https://www.pexels.com/api/ -> "Your API key".

    export PEXELS_API_KEY=...
    python scripts/download_pexels.py --search "restaurant waiter" --list   # show candidates
    python scripts/download_pexels.py --id <video id from --list>           # download one by id
    python scripts/download_pexels.py --search "cafe customers" --pick 2    # 2nd search result

Pick clips filmed from a fixed (static) camera: the zones are fixed polygons, so
pans, zooms and handheld footage will not work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.pexels.com/videos"


def api_get(path: str, key: str, **params) -> dict:
    url = f"{API}/{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
    req = urllib.request.Request(url, headers={"Authorization": key, "User-Agent": "tablewatch"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def best_file(video: dict, max_width: int) -> dict:
    """Largest MP4 rendition no wider than max_width (falls back to the smallest)."""
    files = [f for f in video["video_files"] if f.get("file_type") == "video/mp4" and f.get("width")]
    fitting = [f for f in files if f["width"] <= max_width]
    return max(fitting, key=lambda f: f["width"]) if fitting else min(files, key=lambda f: f["width"])


def download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "tablewatch"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(out, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r{done / total:5.0%} of {total / 1e6:.0f} MB", end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--search", default="restaurant waiter", help="search terms")
    parser.add_argument("--id", type=int, help="download this Pexels video id")
    parser.add_argument("--pick", type=int, default=1, help="which search result to download (1-based)")
    parser.add_argument("--list", action="store_true", help="only list search results")
    parser.add_argument("--max-width", type=int, default=1920)
    parser.add_argument("--out", default="data/sample.mp4")
    args = parser.parse_args()

    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        sys.exit("Set PEXELS_API_KEY (free at https://www.pexels.com/api/), or download manually (see README).")

    if args.id:
        video = api_get(f"videos/{args.id}", key)
    else:
        results = api_get("search", key, query=args.search, orientation="landscape", per_page=15).get("videos", [])
        if not results:
            sys.exit(f"No results for {args.search!r}")
        if args.list:
            for i, v in enumerate(results, 1):
                print(f"{i:2d}. id={v['id']:<9} {v['duration']:>4}s  {v['width']}x{v['height']}  {v['url']}")
            return
        video = results[args.pick - 1]

    f = best_file(video, args.max_width)
    print(f"Pexels video {video['id']} by {video['user']['name']} ({video['duration']}s, {f['width']}x{f['height']})")
    print(f"Page: {video['url']}")
    download(f["link"], Path(args.out))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
