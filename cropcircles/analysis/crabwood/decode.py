#!/usr/bin/env python3
"""Decode the Crabwood 2002 disc from the published bit transcription (R. Collie, cropcircleconnector.com/anasazi/time2007n.html).
Rules fixed before decoding: groups separated by the transcriber's 'S' markers; 8-bit ASCII, MSB first; the transcriber's
flagged extra digits are dropped (2-bit lead-in '01', one trailing digit on 2 groups, '0101' inside BELIEVE, final '0').
Also reports how unlikely the result is under random bits."""
import re, html, math, requests
t = html.unescape(re.sub(r"<[^>]+>", "", requests.get("https://cropcircleconnector.com/anasazi/time2007n.html", timeout=60,
    headers={"User-Agent": "CropCircleVectorizeResearch/0.1 (private analysis)"}).text))
body = t[t.find("MESSAGE", t.find("START")) + 7:t.find("END OF")]
g = [x for x in re.sub(r"[^01S]", "", body).split("S") if x]
fix = {0: lambda x: x[2:], 8: lambda x: x[:8], 83: lambda x: x[:8], 88: lambda x: x[4:]}
g = [fix.get(i, lambda x: x)(x) for i, x in enumerate(g)]
extra = sum(len(x) for x in g if len(x) != 8); g = [x for x in g if len(x) == 8]
msg = "".join(chr(int(x, 2)) for x in g)
print(f"{len(g)} characters, {len(g) * 8} bits (+{extra} stray), all 8-bit: {all(len(x) == 8 for x in g)}")
print(repr(msg))
caps = sum(c.isupper() for c in msg); print("uppercase letters:", caps, "| non-printable:", sum(not (32 <= ord(c) < 127) for c in msg))
p = sum(32 <= v < 127 and (chr(v).isalpha() or chr(v) in " .,&'\\") for v in range(256)) / 256
print(f"chance random bytes all land in letters/space/punctuation: ({p:.3f})^{len(g)} = 10^{len(g) * math.log10(p):.0f}")
