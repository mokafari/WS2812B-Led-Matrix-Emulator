#!/usr/bin/env python3
"""Flatten an aerial formation photo to a top-down view, compensating the camera angle.
Two concentric coplanar circles (rings of the formation) image to two conics whose pencil gives the imaged centre
and the plane's vanishing line l = C1*o (no focal length, EXIF or tramlines needed). Homography H = A*[[1,0,0],[0,1,0],l]
sends l to infinity (removes perspective) and A (from C1) makes ring 1 round. Only concentricity+coplanarity is assumed;
ring ratios are NOT forced: ring 2 is left free and its roundness is reported as a check.
Returns dict(img, H, ring_ratio_free, roundness_err, ...) or (None, reason)."""
import cv2, numpy as np, itertools

def conic(e):
    (cx, cy), (w, h), t = e; a, b, t = w / 2, h / 2, np.deg2rad(t); R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    Q = R @ np.diag([1 / a**2, 1 / b**2]) @ R.T; c = np.array([cx, cy]); q = Q @ c
    return np.block([[Q, -q[:, None]], [-q[None], c @ q - 1]])

def ellipses(g, min_pts=80, support=.55):
    e = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.5), 30, 90); d = cv2.dilate(e, np.ones((3, 3), np.uint8))
    out = []
    for c in cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]:
        if len(c) < min_pts: continue
        el = cv2.fitEllipse(c); (cx, cy), (w, h), t = el
        if min(w, h) < 0.08 * min(g.shape) or max(w, h) / min(w, h) > 4: continue
        th = np.linspace(0, 2 * np.pi, 180); r = np.deg2rad(t)
        x = cx + w / 2 * np.cos(th) * np.cos(r) - h / 2 * np.sin(th) * np.sin(r); y = cy + w / 2 * np.cos(th) * np.sin(r) + h / 2 * np.sin(th) * np.cos(r)
        ok = (x >= 0) & (x < g.shape[1]) & (y >= 0) & (y < g.shape[0])
        if ok.sum() < 60: continue
        s = (d[y[ok].astype(int), x[ok].astype(int)] > 0).mean()
        if s >= support: out.append((s * w * h, el, s))
    return [e for _, e, _ in sorted(out, key=lambda z: -z[0])]

def core(C1, C2):
    """Two concentric-circle conics (any scale) -> T (maps their plane to a metric frame where ring 1 is the unit circle),
    ring2/ring1 radius ratio and ring-2 roundness (1 = perfectly round). None if the pair is not concentric/coplanar-consistent."""
    C1, C2 = C1 / np.linalg.norm(C1), C2 / np.linalg.norm(C2)
    ev, V = np.linalg.eig(np.linalg.inv(C1) @ C2)
    d = [abs(ev[(i + 1) % 3] - ev[(i + 2) % 3]) for i in range(3)]; k = int(np.argmin(d))   # k = the odd one out? (see below)
    # two eigenvalues coincide for concentric circles; the pair with the SMALLEST gap is the repeated one, the other index is the centre
    rep = [i for i in range(3) if i != k]; gap = abs(ev[rep[0]] - ev[rep[1]]); k = [i for i in range(3) if i not in rep][0]
    # NOTE: argmin picks index whose *other two* are closest => k is the distinct eigenvalue's index
    o = np.real(V[:, k]); l = C1 @ o; l = l / np.linalg.norm(l[:2])
    Hp = np.array([[1, 0, 0], [0, 1, 0], l]); Hi = np.linalg.inv(Hp)
    Cr = Hi.T @ C1 @ Hi; Cr = Cr / -Cr[2, 2]; Q = (Cr[:2, :2] + Cr[:2, :2].T) / 2
    if np.linalg.eigvalsh(Q).min() <= 0: return None
    c0 = np.linalg.solve(Q, -Cr[:2, 2]); A = np.linalg.cholesky(Q).T
    T = np.block([[A, -(A @ c0)[:, None]], [np.zeros((1, 2)), np.ones((1, 1))]]) @ Hp
    Ti = np.linalg.inv(T); C2p = Ti.T @ C2 @ Ti; C2p = C2p / -C2p[2, 2]; e2 = np.linalg.eigvalsh((C2p[:2, :2] + C2p[:2, :2].T) / 2)
    if e2.min() <= 0: return None
    return T, float(1 / np.sqrt(e2.mean())), float(np.sqrt(e2.min() / e2.max()))

def flatten(img, out=1000):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); h, w = g.shape; sc = max(h, w)
    N = np.array([[1 / sc, 0, -w / 2 / sc], [0, 1 / sc, -h / 2 / sc], [0, 0, 1]]); Ni = np.linalg.inv(N)
    es = ellipses(g)
    if len(es) < 2: return None, "fewer than 2 ring ellipses found"
    best = None
    e1 = es[0]                                                     # ring 1 = largest supported ellipse (outer circle)
    for e2 in es[1:12]:
        r = core(Ni.T @ conic(e1) @ Ni, Ni.T @ conic(e2) @ Ni)
        if r and 0.15 < r[1] < 0.95 and (best is None or r[2] > best[2]): best = r  # ring 2 = inner ring most consistent with a concentric circle
    if best is None: return None, "no concentric pair"
    T, ratio, rnd = best; S = np.array([[out / 2.3, 0, out / 2], [0, out / 2.3, out / 2], [0, 0, 1]]); Hf = S @ T @ N
    return dict(img=cv2.warpPerspective(img, Hf, (out, out)), H=Hf, ring2_over_ring1=ratio, ring2_roundness=rnd), "ok"

if __name__ == "__main__":
    import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from common import load, ROOT
    for r in load("tune", ("aerial",)):
        if not r["tag_overhead"]: continue
        im = cv2.imread(str(ROOT / r["file"])); o, why = flatten(im)
        print(r["file"].split("/")[-1][:45], why, {k: round(v, 3) for k, v in o.items() if k.startswith("ring")} if o else "")
        if o: cv2.imwrite(f"/tmp/claude-0/flat_{r['file'].split('/')[-1]}", np.hstack([cv2.resize(im, (o['img'].shape[1] * im.shape[1] // im.shape[0], o['img'].shape[0])), o["img"]]))
