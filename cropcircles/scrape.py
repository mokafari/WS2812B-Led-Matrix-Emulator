#!/usr/bin/env python3
"""Download crop-circle images from Wikimedia Commons (with license metadata).
Usage: scrape.py [out_dir] [max_images]"""
import json, re, hashlib, datetime, sys, time, pathlib, requests

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "cropcircles/images")
MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 200
API = "https://commons.wikimedia.org/w/api.php"
S = requests.Session()
S.headers["User-Agent"] = "CropCircleVectorizeResearch/0.1 (personal research scraper)"

def get(url, **kw):
    for i in range(6):
        r = S.get(url, timeout=60, **kw)
        if r.status_code in (429, 503):
            time.sleep(int(r.headers.get("Retry-After", 5 * 2 ** i))); continue
        r.raise_for_status(); return r
    raise RuntimeError(f"gave up on {url}")

def files(cat):
    p = dict(action="query", format="json", generator="categorymembers", gcmtitle=cat,
             gcmtype="file", gcmlimit=50, prop="imageinfo", iiprop="url|extmetadata|mime|size",
             iiurlwidth=1024)
    while True:
        d = get(API, params=p).json()
        yield from d.get("query", {}).get("pages", {}).values()
        if "continue" not in d: return
        p.update(d["continue"]); time.sleep(1)

OUT.mkdir(parents=True, exist_ok=True)
meta, n = [], 0
for page in files("Category:Crop_circles"):
    ii = (page.get("imageinfo") or [{}])[0]
    if not ii.get("mime", "").startswith("image/") or "svg" in ii["mime"]: continue
    m = ii.get("extmetadata", {})
    name = pathlib.Path(page["title"].removeprefix("File:")).stem.replace(" ", "_")[:80] + ".jpg"
    path = OUT / name
    if not path.exists():
        path.write_bytes(get(ii.get("thumburl") or ii["url"]).content); time.sleep(1)
    v = lambda k: re.sub(r"<[^>]+>", "", m.get(k, {}).get("value") or "").strip() or None
    meta.append(dict(file=name, title=page["title"], page_url=ii.get("descriptionurl"), original_url=ii["url"],
                     width=ii.get("width"), height=ii.get("height"), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     license=v("LicenseShortName"), license_url=v("LicenseUrl"), artist=v("Artist"), credit=v("Credit"),
                     attribution_required=v("AttributionRequired"), date=v("DateTimeOriginal"),
                     description=v("ImageDescription"), lat=v("GPSLatitude"), lon=v("GPSLongitude"),
                     source="Wikimedia Commons", retrieved=datetime.date.today().isoformat()))
    n += 1; print(n, name, flush=True)
    if n >= MAX: break
(OUT / "metadata.json").write_text(json.dumps(meta, indent=1))
