#!/usr/bin/env python3
# tools/hydrate_tmdb_min_range.py
# Consistent with original hydrate output: ONE best "overview" string per movie.
import os, sys, json, asyncio, aiohttp, orjson, time, argparse
from pathlib import Path
from tqdm.asyncio import tqdm

API_KEY = os.environ.get("TMDB_API_KEY")
BASE = "https://api.themoviedb.org/3"
HEADERS = {"Accept": "application/json"}
LANG = "en-US"
MAX_RETRIES = 5

def pick_overview(movie):
    # Prefer base overview; fallback to translations with a simple priority list
    ov = (movie.get("overview") or "").strip()
    if ov:
        return ov
    trans = (movie.get("translations") or {}).get("translations", [])
    pref = ["en", movie.get("original_language", ""), "ja", "fr", "es", "de"]
    seen = set()
    for t in trans:
        lang = t.get("iso_639_1") or ""
        if lang in seen:
            continue
        seen.add(lang)
        ov2 = ((t.get("data") or {}).get("overview") or "").strip()
        if ov2:
            if lang in pref:
                return ov2
            if not pref:
                return ov2
    return None

def simplify(movie):
    overview = pick_overview(movie)
    if not overview:
        return None  # skip movies with no usable overview
    genres = [g.get("name") for g in (movie.get("genres") or []) if g.get("name")]
    rd = (movie.get("release_date") or "")[:4]
    year = int(rd) if rd.isdigit() else None
    return {
        "id": movie.get("id"),
        "title": movie.get("title") or movie.get("original_title"),
        "original_language": movie.get("original_language"),
        "genres": genres,
        "overview": overview,
        "poster_path": movie.get("poster_path"),  # path only
        "year": year
    }

class TokenBucket:
    """Simple token bucket limiter to respect TMDB RPS."""
    def __init__(self, rps=4.0, capacity=40):
        self.rps = rps
        self.capacity = capacity
        self.tokens = capacity
        self.ts = time.monotonic()
        self._lock = asyncio.Lock()
    async def take(self, n=1):
        async with self._lock:
            while self.tokens < n:
                now = time.monotonic()
                refill = (now - self.ts) * self.rps
                if refill:
                    self.tokens = min(self.capacity, self.tokens + refill)
                    self.ts = now
                if self.tokens < n:
                    await asyncio.sleep(0.05)
            self.tokens -= n

async def fetch_movie(session: aiohttp.ClientSession, mid: int, limiter: TokenBucket):
    await limiter.take(1)
    url = f"{BASE}/movie/{mid}"
    params = {"api_key": API_KEY, "language": LANG, "append_to_response": "translations"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with session.get(url, params=params) as r:
                if r.status == 404:
                    return None
                if r.status in (429, 500, 502, 503, 504):
                    wait = min(2 ** attempt, 20)
                    await asyncio.sleep(wait)
                    continue
                r.raise_for_status()
                return await r.json()
        except aiohttp.ClientResponseError as e:
            if 400 <= e.status < 500 and e.status != 429:
                return None
            await asyncio.sleep(min(2 ** attempt, 20))
        except aiohttp.ClientError:
            await asyncio.sleep(min(2 ** attempt, 20))
    return None

def load_seen_ids(out_path: Path):
    seen = set()
    if out_path.exists():
        with open(out_path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = orjson.loads(line)
                    mid = d.get("id")
                    if isinstance(mid, int):
                        seen.add(mid)
                except Exception:
                    pass
    return seen

async def hydrate_range(start_id: int, end_id: int, out_path: Path, concurrency: int, rps: float):
    if not API_KEY:
        print("ERROR: set TMDB_API_KEY in your environment.", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen_ids(out_path)
    print(f"[resume] {len(seen)} ids already in {out_path.name}; skipping duplicates.")

    sem = asyncio.Semaphore(concurrency)
    limiter = TokenBucket(rps=rps, capacity=int(max(10, rps * 10)))
    connector = aiohttp.TCPConnector(limit=None)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=60)

    total = max(0, start_id - end_id + 1)
    wrote = 0
    t0 = time.time()

    # ✅ regular file context
    with open(out_path, "ab") as out_f:
        # ✅ async context for the HTTP client
        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=HEADERS
        ) as session:
            for mid in tqdm(range(start_id, end_id, -1), total=total):
                if mid in seen:
                    continue
                async with sem:
                    movie = await fetch_movie(session, mid, limiter)
                if not movie:
                    continue
                mini = simplify(movie)
                if not mini:
                    continue
                out_f.write(orjson.dumps(mini) + b"\n")
                wrote += 1

    print(f"[done] wrote {wrote} new lines to {out_path} in {time.time()-t0:.1f}s")

def parse_args():
    p = argparse.ArgumentParser(description="Hydrate TMDB movies by ID range (downwards).")
    p.add_argument("--start", type=int, required=True, help="Start ID (e.g. 1200000)")
    p.add_argument("--end", type=int, required=True, help="End ID inclusive (e.g. 1000000)")
    p.add_argument("--out", type=Path, required=True, help="Output NDJSON path")
    p.add_argument("--concurrency", type=int, default=10, help="Parallel requests")
    p.add_argument("--rps", type=float, default=4.0, help="Requests per second (per API key)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"🚀 Hydrating {args.start} → {args.end} into {args.out} (concurrency={args.concurrency}, rps={args.rps})")
    asyncio.run(hydrate_range(args.start, args.end, args.out, args.concurrency, args.rps))