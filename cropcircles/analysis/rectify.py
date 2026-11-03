#!/usr/bin/env python3
"""Step 1b: flatten an aerial photo to a top-down view using evenly spaced parallel tramlines.
Tramlines -> vanishing point v; equal spacing along a transversal -> horizon point p; orthogonality of the two
(principal point = image centre, square pixels) -> focal length f -> metric ground homography. Returns None + reason on failure."""
import cv2, numpy as np, itertools, random

def tramlines(g, tol=np.deg2rad(1.5), min_lines=4):
    e = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 2), 40, 120)
    L = cv2.HoughLinesP(e, 1, np.pi / 360, 80, minLineLength=.3 * min(g.shape), maxLineGap=25)
    if L is None or len(L) < 4: return None, None
    L = L.reshape(-1, 4).astype(float); H = np.cross(np.c_[L[:, :2], np.ones(len(L))], np.c_[L[:, 2:], np.ones(len(L))])
    best, rng = (0, None, None), random.Random(0)
    for i, j in (rng.sample(list(itertools.combinations(range(len(L)), 2)), min(600, len(L) * (len(L) - 1) // 2))):
        v = np.cross(H[i], H[j])
        if abs(v[2]) < 1e-9: continue
        vp = v[:2] / v[2]; mid = (L[:, :2] + L[:, 2:]) / 2
        a1 = np.arctan2(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0]); a2 = np.arctan2(vp[1] - mid[:, 1], vp[0] - mid[:, 0])
        d = np.abs((a1 - a2 + np.pi / 2) % np.pi - np.pi / 2); inl = d < tol
        if inl.sum() > best[0]: best = (inl.sum(), vp, inl)
    return (best[1], L[best[2]]) if best[0] >= min_lines else (None, None)

def rectify(img, out=1000):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); h, w = g.shape; c = np.array([w / 2, h / 2])
    v, L = tramlines(g)
    if v is None: return None, "no tramline vanishing point"
    # transversal: image row through centre (rotate-free); positions of each tramline on it, merge near-duplicates
    xs = sorted({round(float(x), 0) for l in L for x in [l[0] + (c[1] - l[1]) * (l[2] - l[0]) / (l[3] - l[1] + 1e-9)] if abs(x) < 5 * w})
    if len(xs) < 4: return None, 'tramlines not crossing the transversal'
    m = [xs[0]]; [m.append(x) for x in xs[1:] if x - m[-1] > 8]
    if len(m) < 4: return None, f"only {len(m)} distinct tramlines"
    s = np.array(m); i = np.arange(len(s))
    from scipy.optimize import least_squares  # x = (a*i+b)/(c*i+1)
    r = least_squares(lambda q: (q[0] * i + q[1]) / (q[2] * i + 1) - s, [np.ptp(s) / len(s), s[0], 0], bounds=([-1e5, -1e5, -.2], [1e5, 1e5, .2]))
    res = float(np.sqrt(np.mean(r.fun ** 2)) / np.ptp(s))
    if res > .02: return None, f"spacing not even (rel rms {res:.3f})"
    px = r.x[0] / r.x[2] if abs(r.x[2]) > 1e-6 else 1e9; p = np.array([px, c[1]])
    f2 = -np.dot(v - c, p - c)
    if f2 <= 0: return None, "orthogonality gives f^2<=0"
    f = np.sqrt(f2); K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]]); Ki = np.linalg.inv(K)
    r1 = Ki @ np.r_[v, 1]; r1 /= np.linalg.norm(r1); r2 = Ki @ np.r_[p, 1]; r2 -= r1 * (r1 @ r2); r2 /= np.linalg.norm(r2)
    Hg = K @ np.c_[r1, r2, np.cross(r1, r2) * 0 + Ki @ np.r_[c, 1] * 1.0]   # ground->image (ground z=0)
    Hi = np.linalg.inv(Hg)
    q = Hi @ np.c_[[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]; q = (q[:2] / q[2]).T
    if not np.isfinite(q).all(): return None, "degenerate homography"
    sc = out / max(np.ptp(q[:, 0]), np.ptp(q[:, 1])); T = np.array([[sc, 0, -q[:, 0].min() * sc], [0, sc, -q[:, 1].min() * sc], [0, 0, 1]])
    return dict(img=cv2.warpPerspective(img, T @ Hi, (int(np.ptp(q[:, 0]) * sc), int(np.ptp(q[:, 1]) * sc))), f=float(f), spacing_rms=res, n_lines=len(m)), "ok"

if __name__ == "__main__":
    import sys; sys.path.insert(0, ".."); sys.path.insert(0, ".")
    from common import load, ROOT
    for r in load("tune", ("aerial", "graphic")):
        im = cv2.imread(str(ROOT / r["file"])); o, why = rectify(im) if r["kind"] == "aerial" else (None, "graphic")
        print(r["file"][:50], r["kind"], "OH" if r["tag_overhead"] else "--", why)
        if o: cv2.imwrite(f"/tmp/claude-0/rect_{r['file'].split('/')[-1]}", o["img"])
