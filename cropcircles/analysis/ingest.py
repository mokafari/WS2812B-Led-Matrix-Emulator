#!/usr/bin/env python3
"""Step 1: dedupe by perceptual fingerprint, classify (diagram/aerial/ground), split tune(20%)/locked(80%).
Usage: ingest.py <images_dir>  -> writes <images_dir>/manifest.json
Classification uses ONLY pixels (+ tags as a weak cross-check); no dates/places/labels are used for analysis."""
import json, sys, hashlib, pathlib, numpy as np, imagehash
from PIL import Image

D = pathlib.Path(sys.argv[1]); meta = json.loads((D / "metadata.json").read_text())

def feats(im):
    a = np.asarray(im.convert("RGB").resize((256, 256)), float) / 255
    mx, mn = a.max(2), a.min(2); sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    q = (a * 15).astype(int); ncol = len(np.unique(q[..., 0] * 256 + q[..., 1] * 16 + q[..., 2]))
    top = a[:40]; sky = float(((top[..., 2] > top[..., 0]) & (top.std(axis=(0, 1)).mean() < .08)).mean())
    h = np.asarray(im.convert("HSV").resize((256, 256)), float) / 255
    field = float(((h[..., 0] > .08) & (h[..., 0] < .30) & (h[..., 1] > .15)).mean())
    med = np.median(a.reshape(-1, 3), 0)
    return dict(field=field, bgwhite=bool(med.min() > .9), sat=float(sat.mean()), ncol=int(ncol), sky_top=sky, bg=float((np.abs(a - np.median(a.reshape(-1, 3), 0)).max(2) < .06).mean()))

def kind(f, tags):  # rules tuned on the tune split only (visual check of 28 images)
    if f["bg"] > .5 and f["bgwhite"]: return "graphic"      # white-background drawings / art
    if f["field"] < .3: return "other"                       # not a crop field (art, posters, people)
    if f["sky_top"] > .5 or any(t.split()[0] in ("L","L2","L3","L4","L5","L6","L7","LOW","LOW2","LOW3","CL") for t in tags): return "ground"
    return "aerial"

rows = []
for m in meta:
    try: im = Image.open(D / m["file"])
    except Exception: continue
    f = feats(im); tags = m["alt_parts"][3:]
    rows.append(dict(file=m["file"], formation=m["project_url"], w=im.width, h=im.height, ph=str(imagehash.phash(im.convert("L"), 16)),
                     tag_overhead=any(t.startswith("OH") for t in tags), **f, kind=kind(f, tags)))
# dedupe: global, hamming <= 10 of 256 bits; keep highest resolution
rows.sort(key=lambda r: -r["w"] * r["h"]); keep, hs = [], []
for r in rows:
    h = imagehash.hex_to_hash(r["ph"])
    if any(h - k <= 10 for k in hs): r["dup"] = True; continue
    hs.append(h); r["dup"] = False; keep.append(r)
# split by FORMATION (no leakage between images of one circle); deterministic hash, 20% tune / 80% locked
for r in rows: r["split"] = "tune" if int(hashlib.sha256(r["formation"].encode()).hexdigest(), 16) % 5 == 0 else "locked"
(D / "manifest.json").write_text(json.dumps(rows, indent=1))
import collections as c
print(len(rows), "images,", len(keep), "unique;", c.Counter((r["kind"], r["split"]) for r in keep))
oh = [r for r in keep if r["tag_overhead"]]; print("tagged OH:", len(oh), "->", c.Counter(r["kind"] for r in oh))
