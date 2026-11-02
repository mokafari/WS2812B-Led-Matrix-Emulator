#!/usr/bin/env python3
"""Private-analysis download of Temporary Temples crop circle photos + metadata.
Images (c) Temporary Temples / photographers - do not redistribute.
Usage: scrape_tt.py [out_dir] [first_year] [last_year]"""
import json, re, sys, time, hashlib, datetime, pathlib, html, requests

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "cropcircles/images_tt")
Y0, Y1 = int(sys.argv[2]) if len(sys.argv) > 2 else 1994, int(sys.argv[3]) if len(sys.argv) > 3 else 2026
B, DELAY = "https://temporarytemples.co.uk", 2
S = requests.Session()
S.headers["User-Agent"] = "CropCircleVectorizeResearch/0.1 (private analysis, low rate)"

def get(url):
    for i in range(6):
        r = S.get(url, timeout=60); time.sleep(DELAY)
        if r.status_code in (429, 503): time.sleep(10 * 2 ** i); continue
        if r.status_code == 404: return None
        r.raise_for_status(); return r
    raise RuntimeError(url)

txt = lambda s: html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s))).strip()
OUT.mkdir(parents=True, exist_ok=True)
mp = OUT / "metadata.json"
meta = json.loads(mp.read_text()) if mp.exists() else []
done = {m["project_url"] for m in meta}

for y in range(Y0, Y1 + 1):
    r = get(f"{B}/crop-circles/{y}-crop-circles")
    if not r: continue
    urls = list(dict.fromkeys(re.findall(r'href="(%s/project/[^"]+)"' % B, r.text)))
    print(y, len(urls), "projects", flush=True)
    for u in urls:
        if u in done: continue
        p = get(u)
        if not p: continue
        h = p.text
        title = txt(re.search(r"<title>(.*?)</title>", h, re.S).group(1)).removesuffix(" - Temporary Temples")
        main = h[h.find('id="logo"'):]
        imgs = [(m.group(1), (re.search(r'alt="([^"]*)"', m.group(0)) or [0, ""])[1]) for m in
                re.finditer(r'<img[^>]+src="(%s/wp-content/uploads/[^"]+)"[^>]*>' % B, main)]
        body = re.search(r"Detail.{0,10}Location:(.*?)(?:Crop Circle|Share|Related|$)", txt(main))
        maps = re.search(r'href="(https://(?:www\.)?(?:google\.[^/]+/maps|goo\.gl|maps\.app)[^"]*)"', h)
        for src, alt in dict.fromkeys(imgs):
            full = re.sub(r"-\d+x\d+(\.\w+)$", r"\1", src)
            if "logo" in src or not alt.startswith(title.split("|")[0].strip()[:6]): continue
            name = full.rsplit("/", 1)[1]
            f = OUT / str(y) / name; f.parent.mkdir(exist_ok=True)
            if not f.exists():
                ir = get(full) or get(src)
                if not ir: continue
                f.write_bytes(ir.content)
            meta.append(dict(year=y, file=f"{y}/{name}", title=html.unescape(title), project_url=u, image_url=full,
                             alt=html.unescape(alt or ""), alt_parts=[a.strip() for a in html.unescape(alt or "").split("|")],
                             description=body.group(1).strip() if body else None, maps_url=maps.group(1) if maps else None,
                             sha256=hashlib.sha256(f.read_bytes()).hexdigest(), source="Temporary Temples",
                             rights="(c) Temporary Temples / photographer; private analysis only, do not redistribute",
                             retrieved=datetime.date.today().isoformat()))
        done.add(u); mp.write_text(json.dumps(meta, indent=1))
        print(" ", title, flush=True)
