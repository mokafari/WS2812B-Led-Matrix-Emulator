#!/usr/bin/env python3
"""Battery of classic hidden-text techniques on the Crabwood 2002 text, each against a shuffle null (fixed before running).
  T1 acrostics     first / last letter of each word and each sentence, forwards and backwards   null: shuffled words, same sentence lengths
  T2 Bacon cipher  letter case as A/B 5-bit groups (24- and 26-letter alphabets, all phases, both polarities,
                   forwards/backwards), on letter-level and word-level case bits                 null: shuffled words, each keeping its case pattern (keeps runs)
  T3 skip ciphers  every n-th letter, n = 2..40, every start, forwards/backwards                 null: shuffled letters
  T4 stray digits  the transcriber's 9 extra digits, shown (too few bits to test)
Score = share of the output covered by common English words (>= 3 letters, 10k list), best over the whole family.
p = (1 + #null >= observed) / 201; 3 testable families, so Bonferroni threshold 0.05 / 3 = 0.017."""
import re, sys, pathlib, numpy as np, requests
sys.path.insert(0, str(pathlib.Path(__file__).parent)); from decode import crabwood

W = {w for w in requests.get("https://raw.githubusercontent.com/first20hours/google-10000-english/master/"
     "google-10000-english-no-swears.txt", timeout=60).text.split() if len(w) >= 3}
MAXL = max(map(len, W))

def cover(s):
    """Max share of letters covered by non-overlapping dictionary words (DP)."""
    s = re.sub(r"[^a-z]", "", s.lower()); n = len(s)
    if n < 6: return 0.0
    b = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        b[i] = b[i + 1]
        for L in range(3, min(MAXL, n - i) + 1):
            if s[i:i + L] in W: b[i] = max(b[i], L + b[i + L])
    return b[0] / n

def fam(cands):  # best (score, label, text) over a family of candidate strings
    return max(((cover(t), lab, t) for lab, t in cands), key=lambda z: z[0])

def acrostics(words, sents):
    out = []
    for nm, seq in (("word", words), ("sentence", sents)):
        for pos, f in (("first", lambda w: w[0]), ("last", lambda w: w[-1])):
            s = "".join(f(w) for w in seq if w)
            out += [(f"{pos} letter of each {nm}", s), (f"{pos} letter of each {nm}, reversed", s[::-1])]
    return out

A26 = "abcdefghijklmnopqrstuvwxyz"; A24 = "abcdefghiklmnopqrstuwxyz"     # Bacon's original merges I/J and U/V
def bacon(bits, tag):
    out = []
    for direction, b in (("", bits), (" reversed", bits[::-1])):
        for inv in (0, 1):
            for ph in range(5):
                v = [int("".join(map(str, b[i:i + 5] ^ inv)), 2) for i in range(ph, len(b) - 4, 5)]
                for an, A in (("26", A26), ("24", A24)):
                    out.append((f"Bacon-{an} on {tag}, phase {ph}, {'inverted' if inv else 'upper=B'}{direction}",
                                "".join(A[x] if x < len(A) else "?" for x in v)))
    return out

def skips(letters):
    return [(f"every {n}th letter from {st}{' reversed' if r else ''}", (letters[::-1] if r else letters)[st::n])
            for n in range(2, 41) for st in range(n) for r in (0, 1)]

if __name__ == "__main__":
    msg, g, _ = crabwood(); rng = np.random.default_rng(0)
    words = re.findall(r"[A-Za-z]+", msg); sents = [s.strip() for s in re.split(r"[.\\]", msg) if s.strip()]
    letters = "".join(words).lower(); lc = np.array([c.isupper() for c in "".join(words)], np.uint8)
    wc = np.array([int(any(ch.isupper() for ch in w[1:]) or (w[0].isupper() and i > 0 and not msg[:msg.find(w)].rstrip().endswith(".")))
                   for i, w in enumerate(words)], np.uint8)
    slen = [len(re.findall(r"[A-Za-z]+", x)) for x in sents]
    resplit = lambda ws: [" ".join(ws[sum(slen[:i]):sum(slen[:i + 1])]) for i in range(len(slen))]  # same sentence lengths
    print("capitals in order:", "".join(c for c in msg if c.isupper()))
    print("stray transcriber digits: 01 (lead-in) 0 (after 'th') 1 (after 'B') 0101 (inside BELIEVE) 0 (end) -> 9 bits, too few to carry text\n")
    tests = {
        "T1 acrostics": (lambda: acrostics(words, sents), lambda: (lambda ws: acrostics(ws, resplit(ws)))(list(rng.permutation(words)))),
        "T2 Bacon (case bits)": (lambda: bacon(lc, "letter case") + bacon(wc, "word case"),
                                 lambda: (lambda o: bacon(np.concatenate([np.array([c.isupper() for c in words[i]], np.uint8) for i in o]), "letter case")
                                          + bacon(wc[o], "word case"))(rng.permutation(len(words)))),
        "T3 skip ciphers": (lambda: skips(letters), lambda: skips("".join(rng.permutation(list(letters))))),
    }
    for name, (real, null) in tests.items():
        s, lab, t = fam(real()); ns = [fam(null())[0] for _ in range(200)]
        p = (1 + sum(x >= s for x in ns)) / 201
        print(f"{name:22s} best coverage {s:.2f} vs null median {np.median(ns):.2f} (95th {np.percentile(ns, 95):.2f})  p={p:.3f}"
              f"  {'SIGNIFICANT' if p < .05 / 3 else 'not significant'}\n    best: {lab}: {t[:70]!r}")
