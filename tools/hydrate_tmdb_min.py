# tools/hydrate_tmdb_min.py
import os, sys, gzip, json, asyncio, aiohttp, orjson, time
from pathlib import Path
from typing import Any, Dict, Optional
from tqdm.asyncio import tqdm

API_KEY = os.environ.get("TMDB_API_KEY")
BASE = "https://api.themoviedb.org/3"

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "movies.ndjson"  # ~150–500MB gz if you gzip later

HEADERS = {"Accept": "application/json"}
LANG = "en-US"
CONCURRENCY = 10        # be polite; you can tune to 15–20 later
MAX_RETRIES = 5

def pick_overview(movie: Dict[str, Any]) -> Optional[str]:
    # 1) prefer overview from requested language
    ov = movie.get("overview") or None
    if ov: return ov.strip() or None
    # 2) fall back via translations (if appended)
    tr = movie.get("translations", {}).get("translations", []) or []
    # prefer English-family, then original language, then anything non-empty
    pref = ["en", movie.get("original_language", ""), "ja", "fr", "es", "de"]
    # Build map {lang: overview}
    best = None
    seen = set()
    for t in tr:
        lang = t.get("iso_639_1") or ""
        if lang in seen:  # first hit usually best
            continue
        seen.add(lang)
        ov2 = (t.get("data") or {}).get("overview") or ""
        ov2 = ov2.strip()
        if ov2:
            if lang in pref and best is None:
                best = (lang, ov2)
            if not best:
                best = (lang, ov2)
    return best[1] if best else None

async def fetch_movie(session: aiohttp.ClientSession, mid: int) -> Optional[Dict[str, Any]]:
    url = f"{BASE}/movie/{mid}"
    params = {
        "api_key": API_KEY,
        "language": LANG,
        "append_to_response": "translations"
    }
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

def simplify(movie: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    overview = pick_overview(movie)
    if not overview:
        return None  # skip movies with no usable summary
    genres = [g.get("name") for g in (movie.get("genres") or []) if g.get("name")]
    rd = (movie.get("release_date") or "")[:4]
    year = int(rd) if rd.isdigit() else None
    return {
        "id": movie.get("id"),
        "title": movie.get("title") or movie.get("original_title"),
        "original_language": movie.get("original_language"),
        "genres": genres,
        "overview": overview,
        "poster_path": movie.get("poster_path"),  # keep path only
        "year": year
    }

async def producer(ids_path: Path):
    # stream IDs from gz or plain file
    if ids_path.suffix == ".gz":
        f = gzip.open(ids_path, "rt", encoding="utf-8")
    else:
        f = open(ids_path, "r", encoding="utf-8")
    try:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                mid = obj.get("id")
                if isinstance(mid, int):
                    yield mid
            except json.JSONDecodeError:
                continue
    finally:
        f.close()

async def hydrate(ids_path: Path, out_path: Path, limit: Optional[int] = None):
    if not API_KEY:
        print("ERROR: set TMDB_API_KEY in your environment.", file=sys.stderr)
        sys.exit(1)

    connector = aiohttp.TCPConnector(limit=None)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=60)
    sem = asyncio.Semaphore(CONCURRENCY)

    # open output file (NDJSON)
    out_f = open(out_path, "w", encoding="utf-8")

    async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=HEADERS) as session:
        i = 0
        async for mid in producer(ids_path):
            if limit and i >= limit:
                break
            i += 1

            async with sem:
                movie = await fetch_movie(session, mid)
            if not movie:
                continue
            mini = simplify(movie)
            if not mini:
                continue
            out_f.write(orjson.dumps(mini).decode("utf-8") + "\n")

    out_f.close()
    print(f"Wrote NDJSON to {out_path}")

if __name__ == "__main__":
    # Usage: python tools/hydrate_tmdb_min.py data/ids/movie_ids.json.gz  (optional limit)
    ids = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/ids/movie_ids.json.gz")
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    asyncio.run(hydrate(ids, OUT_PATH, lim))
    