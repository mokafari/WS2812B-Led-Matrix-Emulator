#!/usr/bin/env python3
"""Statistical profile of the Crabwood 2002 text against ordinary English.
Reference: 3000 random word-aligned windows of the same length (151 chars) from three public-domain novels
(War of the Worlds, Alice in Wonderland, Frankenstein; GITenberg mirrors). For each statistic, Crabwood's percentile among
the windows, a two-sided empirical p, and Holm correction across all statistics. Fixed before running."""
import re, zlib, math, sys, pathlib, collections, numpy as np, requests
sys.path.insert(0, str(pathlib.Path(__file__).parent)); from decode import crabwood

REFS = ["https://raw.githubusercontent.com/GITenberg/The-War-of-the-Worlds_36/master/36.txt",
        "https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11.txt",
        "https://raw.githubusercontent.com/GITenberg/Frankenstein_84/master/84.txt"]

def corpus():
    out = []
    for u in REFS:
        t = requests.get(u, timeout=60).text; n = len(t)
        out.append(re.sub(r"\s+", " ", t[n // 10: n * 9 // 10]))   # skip licence header/footer
    return out

def windows(texts, L, n, rng):
    res = []
    while len(res) < n:
        t = texts[rng.integers(len(texts))]; i = rng.integers(len(t) - 3 * L); i = t.find(" ", i) + 1
        w = t[i:i + L]
        if len(w) == L: res.append(w)
    return res

def stats(s, eng):
    L = [c.lower() for c in s if c.isalpha()]; n = len(L); cnt = collections.Counter(L)
    words = re.findall(r"[A-Za-z]+", s); bits = "".join(f"{ord(c) & 255:08b}" for c in s)
    b = np.array(list(bits), int); runs = 1 + int((b[1:] != b[:-1]).sum()); p1 = b.mean()
    mu = 1 + 2 * len(b) * p1 * (1 - p1); sd = math.sqrt(max(2 * len(b) * p1 * (1 - p1) * (2 * len(b) * p1 * (1 - p1) - 1) / (len(b) - 1), 1e-9))
    ch = collections.Counter(s)
    return {
        "index of coincidence": sum(v * (v - 1) for v in cnt.values()) / (n * (n - 1)),
        "letter-frequency chi2 vs English": sum((cnt.get(c, 0) - n * eng[c]) ** 2 / (n * eng[c]) for c in eng),
        "char entropy (bits/char)": -sum(v / len(s) * math.log2(v / len(s)) for v in ch.values()),
        "zlib compressed bytes": len(zlib.compress(s.encode("latin-1"), 9)),
        "mean word length": np.mean([len(w) for w in words]),
        "type/token ratio": len({w.lower() for w in words}) / len(words),
        "uppercase share of letters": sum(c.isupper() for c in s) / n,
        "punctuation share": sum(not c.isalnum() and c != " " for c in s) / len(s),
        "space share": s.count(" ") / len(s),
        "share of 1-bits": p1,
        "bit runs z-score": (runs - mu) / sd,
    }

if __name__ == "__main__":
    msg, _, _ = crabwood()
    if "--normalised" in sys.argv:   # control: sentence case + a space after each full stop, words unchanged
        msg = re.sub(r"\.(?=\S)", ". ", msg).replace("\\", "")
        msg = re.sub(r"(^|\. )([a-z])", lambda m: m.group(1) + m.group(2).upper(), msg.lower()).replace(" i ", " I ")
        print("normalised:", repr(msg))
    rng = np.random.default_rng(0); texts = corpus()
    allL = collections.Counter(c for t in texts for c in t.lower() if "a" <= c <= "z"); tot = sum(allL.values())
    eng = {c: v / tot for c, v in allL.items()}
    obs = stats(msg, eng); ref = [stats(w, eng) for w in windows(texts, len(msg), 3000, rng)]
    rows = []
    for k, v in obs.items():
        r = np.array([x[k] for x in ref]); pct = (r < v).mean() + .5 * (r == v).mean(); p = min(1, 2 * min(pct, 1 - pct) + 1 / 3001)
        rows.append([k, v, np.median(r), np.percentile(r, 2.5), np.percentile(r, 97.5), pct, p])
    order = np.argsort([x[6] for x in rows]); m = len(rows); adj = [0] * m; run = 0
    for rank, i in enumerate(order): run = max(run, min(1, (m - rank) * rows[i][6])); adj[i] = run
    print(f"Crabwood: {len(msg)} chars, {sum(c.isalpha() for c in msg)} letters, {len(re.findall('[A-Za-z]+', msg))} words | reference: 3000 windows\n")
    print(f"{'statistic':34s} {'Crabwood':>9s} {'English median':>15s} {'English 95% range':>20s} {'pctile':>7s} {'p (Holm)':>9s}")
    for (k, v, med, lo, hi, pct, p), a in zip(rows, adj):
        print(f"{k:34s} {v:9.3f} {med:15.3f} {f'{lo:.3f} - {hi:.3f}':>20s} {pct * 100:6.1f}% {a:9.3f}{'  *' if a < .05 else ''}")
