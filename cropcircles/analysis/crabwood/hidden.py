#!/usr/bin/env python3
"""Search the Crabwood 2002 capitalisation for a hidden layer, alone and XOR-ed with the 1974 Arecibo message.

Rules fixed before looking at any output:
  Streams  S1 case of every letter (1 = upper)
           S2 per word: 1 if it has a capital that is not a sentence-initial first letter (the "false capitals")
           S3 all 1208 message bits
  Keys     none, or Arecibo 1679 bits read row-major (as sent), reversed, column-major, row-major complemented;
           every alignment offset
  Decoders 8-bit ASCII, 7-bit ASCII, 5-bit A1Z26 (0 = space); every phase, both polarities, both bit orders
  Score    mean English unigram log-probability per character (letters + space; anything else = log 1e-4)
  Null     the same max-over-everything search with 199 shuffled-Arecibo keys (XOR) or 199 shuffled streams (no key).
           p = (1 + #null >= observed) / 200. A hidden message must beat chance AFTER the whole search, not just read nicely."""
import re, sys, pathlib, numpy as np, requests
sys.path.insert(0, str(pathlib.Path(__file__).parent)); from decode import crabwood

ARECIBO_SRC = ["https://raw.githubusercontent.com/ducktors/arecibo/HEAD/src/arecibo-message.ts",
               "https://raw.githubusercontent.com/n8mob/py/HEAD/arecibo_message.py"]
F = dict(zip(" etaoinshrdlcumwfgypbvkjxqz", [18, 10.2, 7.4, 6.7, 6.1, 5.8, 5.5, 5.2, 5.0, 4.9, 3.5, 3.3, 3.2, 2.2, 2.0, 1.9, 1.8, 1.6, 1.6, 1.5, 1.2, .8, .6, .15, .15, .1, .06]))
NOSPACE = "--nospace" in sys.argv
if NOSPACE: F[" "] = 1e-4 * sum(F.values())   # robustness: spaces get no credit
tot = sum(F.values()); LP = {c: np.log(v / tot) for c, v in F.items()}; BAD = np.log(1e-4)

def table(kind):
    if kind == "a1z26": return np.array([LP[" "]] + [LP[chr(96 + i)] for i in range(1, 27)] + [BAD] * 5)
    w = 8 if kind == "ascii8" else 7
    return np.array([LP.get(chr(v).lower(), BAD) if (chr(v).isalpha() and v < 128) or v == 32 else BAD for v in range(2 ** w)])
DEC = {"ascii8": (8, table("ascii8")), "ascii7": (7, table("ascii7")), "a1z26": (5, table("a1z26"))}

def best(M, label_offsets=None):
    """M: (m, n) candidate bit rows. Max score over decoders/phase/polarity/bit order; returns (score, info)."""
    top = (-1e9, None)
    for name, (w, T) in DEC.items():
        for ph in range(w):
            c = (M.shape[1] - ph) // w
            if c < 3: continue
            B = M[:, ph:ph + c * w].reshape(len(M), c, w)
            for inv in (0, 1):
                for rev in (0, 1):
                    X = (B ^ inv)[:, :, ::-1] if rev else B ^ inv
                    v = X @ (1 << np.arange(w - 1, -1, -1)); s = T[v].mean(1); i = int(s.argmax())
                    if s[i] > top[0]: top = (float(s[i]), (name, ph, inv, rev, i, v[i]))
    return top

def text(info):
    name, *_ , v = info
    return "".join((" " if x == 0 else chr(96 + x) if x <= 26 else "?") if name == "a1z26" else (chr(x) if 32 <= x < 127 else "?") for x in v)

def keyset(k):  # 4 readings of a 1679-bit key
    g = k.reshape(73, 23); return {"row": k, "reversed": k[::-1], "column": g.T.ravel(), "complement": 1 - k}

def search(x, keys):
    if keys is None: return best(x[None]) + ("plain",)
    top = (-1e9, None, None)
    for kn, k in keys.items():
        n = len(x); W = np.lib.stride_tricks.sliding_window_view(k, n)        # every alignment
        s, info = best(W ^ x)
        if s > top[0]: top = (s, info, f"{kn}, offset {info[4]}")
    return top

if __name__ == "__main__":
    msg, g, _ = crabwood(); rng = np.random.default_rng(0)
    srcs = ["".join(re.findall(r"[01]{10,}", requests.get(u, timeout=60).text)) for u in ARECIBO_SRC]
    assert len(srcs[0]) == 1679 and srcs[0] == srcs[1], "Arecibo sources disagree"
    A = np.array(list(srcs[0]), np.uint8)
    S1 = np.array([c.isupper() for c in msg if c.isalpha()], np.uint8)
    words = re.findall(r"[A-Za-z]+|[.&]", msg); S2, start = [], True
    for wd in words:
        if wd in ".&": start = wd == "." or start; continue
        S2.append(int(any(ch.isupper() for ch in (wd[1:] if start else wd)))); start = False
    S2 = np.array(S2, np.uint8); S3 = np.array(list("".join(g)), np.uint8)
    print(f"S1 {len(S1)} bits ({S1.sum()} upper) | S2 {len(S2)} words: {''.join(map(str, S2))} | S3 {len(S3)} bits | Arecibo 1679 bits (2 sources agree)\n")
    for sn, x in (("S1 letter case", S1), ("S2 false-capital words", S2), ("S3 full message", S3)):
        if NOSPACE and sn[:2] != "S1": continue
        for mode in ("plain", "XOR Arecibo"):
            if mode == "plain":
                obs = search(x, None); null = [search(rng.permutation(x), None)[0] for _ in range(199)]
            else:
                obs = search(x, keyset(A)); null = [search(x, keyset(rng.permutation(A)))[0] for _ in range(199)]
            p = (1 + sum(n >= obs[0] for n in null)) / 200
            print(f"{sn:24s} {mode:12s} best {obs[0]:6.2f} vs null median {np.median(null):6.2f} (95th {np.percentile(null, 95):6.2f})  p={p:.3f}"
                  f"  [{obs[1][0]} {obs[2]}] -> {text(obs[1])[:60]!r}")
