#!/usr/bin/env python3
"""Use the message itself as the key. Rules fixed before running:
  F1 delimiters  odd characters split the text. Delimiter sets: capitals, punctuation (. & \\), capitals+punctuation,
                 the two mid-word capitals (oPpose, COnduit). Readings: segment lengths as letters (A=1, mod 26) and as
                 parity bits (5-bit Bacon / 8-bit ASCII); the character just before / after each delimiter.
  F2 self-key    the message as its own key: bytes XOR the message shifted by k (k = 1..150) or reversed;
                 Vigenere decryption of the letters with running key = the message shifted by k, or key = the capitals.
Score = share covered by common English words (as hidden2.py). Null: words shuffled, each keeping its case pattern,
punctuation left in place; 200 runs; p = (1 + #null >= obs) / 201; Bonferroni over 2 families -> 0.025."""
import re, sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent)); from decode import crabwood; from hidden2 import cover

A = "abcdefghijklmnopqrstuvwxyz"
DELIMS = {"capitals": str.isupper, "punctuation": lambda c: c in ".&\\", "capitals+punctuation": lambda c: c.isupper() or c in ".&\\"}

def bits_to(b, w):
    return [int("".join(map(str, b[i:i + w])), 2) for i in range(0, len(b) - w + 1, w)]

def f1(m):
    out = []
    mid = [i for i, c in enumerate(m) if c.isupper() and i > 0 and m[i - 1].isalpha() and any(x.islower() for x in re.findall(r"[A-Za-z]+", m[:i + 1])[-1] + re.match(r"[A-Za-z]*", m[i:]).group())]
    sets = {k: [i for i, c in enumerate(m) if f(c)] for k, f in DELIMS.items()} | {"mid-word capitals": mid}
    for name, idx in sets.items():
        if len(idx) < 2: continue
        seg = np.diff([-1] + idx + [len(m)]) - 1
        out += [(f"{name}: segment lengths as letters", "".join(A[(x - 1) % 26] if x > 0 else " " for x in seg)),
                (f"{name}: char after each", "".join(m[i + 1] for i in idx if i + 1 < len(m))),
                (f"{name}: char before each", "".join(m[i - 1] for i in idx if i > 0))]
        par = list(seg % 2)
        out += [(f"{name}: segment-length parity, Bacon-26", "".join(A[v] if v < 26 else "?" for v in bits_to(par, 5))),
                (f"{name}: segment-length parity, ASCII-8", "".join(chr(v) if 32 <= v < 127 else "?" for v in bits_to(par, 8)))]
    return out

def f2(m):
    b = np.frombuffer(m.encode("latin-1"), np.uint8); L = re.sub(r"[^a-z]", "", m.lower()); out = []
    asc = lambda a: "".join(chr(v) if 32 <= v < 127 else "?" for v in a)
    for k in range(1, min(151, len(L))):
        out.append((f"XOR with itself shifted {k}", asc(b ^ np.roll(b, k))))
        key = L[k:] + L[:k]; out.append((f"Vigenere, running key = message shifted {k}", "".join(A[(A.index(c) - A.index(q)) % 26] for c, q in zip(L, key))))
    out.append(("XOR with itself reversed", asc(b ^ b[::-1])))
    caps = "".join(c for c in m if c.isupper()).lower()
    out.append(("Vigenere, key = the capitals", "".join(A[(A.index(c) - A.index(caps[i % len(caps)])) % 26] for i, c in enumerate(L))))
    return out

def shuffled(m, rng):
    toks = re.split(r"([A-Za-z]+)", m); wi = [i for i, t in enumerate(toks) if re.fullmatch(r"[A-Za-z]+", t)]
    perm = rng.permutation(wi); new = toks[:]
    for a, b in zip(wi, perm): new[a] = toks[b]
    return "".join(new)

def best(cands):
    return max(((cover(t), lab, t) for lab, t in cands), key=lambda z: z[0])

if __name__ == "__main__":
    msg, _, _ = crabwood(); rng = np.random.default_rng(0)
    for name, f in (("F1 odd chars as delimiters", f1), ("F2 message as its own key", f2)):
        s, lab, t = best(f(msg)); ns = [best(f(shuffled(msg, rng)))[0] for _ in range(200)]
        p = (1 + sum(x >= s for x in ns)) / 201
        print(f"{name:28s} best {s:.2f} vs null median {np.median(ns):.2f} (95th {np.percentile(ns, 95):.2f}) p={p:.3f} "
              f"{'SIGNIFICANT' if p < .025 else 'not significant'}\n    best: {lab}: {t[:70]!r}")
    print("\nall delimiter readings of the real message:")
    for lab, t in f1(msg): print(f"  {lab:45s} {t[:70]!r}")
