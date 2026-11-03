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

def ridge_points(g, n=5000, seed=0):
    """Dark narrow lines (ditches in the crop): DoG response, thresholded, thinned. Returns (x, y) in a <=1000px frame."""
    k = min(1000 / max(g.shape), 1); g = cv2.resize(g, None, fx=k, fy=k).astype(np.float32)
    r = cv2.GaussianBlur(g, (0, 0), 6) - cv2.GaussianBlur(g, (0, 0), 1.6)          # >0 where a dark line sits
    m = (r > max(6, np.percentile(r, 92))).astype(np.uint8)
    m = cv2.ximgproc.thinning(m * 255) > 0 if hasattr(cv2, "ximgproc") else m > 0
    gs = cv2.GaussianBlur(g, (0, 0), 2); hxx, hyy, hxy = cv2.Sobel(gs, cv2.CV_32F, 2, 0), cv2.Sobel(gs, cv2.CV_32F, 0, 2), cv2.Sobel(gs, cv2.CV_32F, 1, 1)
    th = 0.5 * np.arctan2(2 * hxy, hxx - hyy) + np.pi / 2            # direction ALONG the ridge (mod pi)
    ys, xs = np.nonzero(m); P = np.c_[xs, ys].astype(float); T = th[ys, xs]; rng = np.random.default_rng(seed)
    sel = rng.choice(len(P), n, replace=False) if len(P) > n else np.arange(len(P))
    return np.c_[P[sel], T[sel]], k

def _fit_eval(P, el, tol):
    (cx, cy), (w, h), t = el; r = np.deg2rad(t); R = np.array([[np.cos(r), np.sin(r)], [-np.sin(r), np.cos(r)]]); u = (P[:, :2] - [cx, cy]) @ R.T
    d = np.abs(np.hypot(u[:, 0] / (w / 2), u[:, 1] / (h / 2)) - 1) * min(w, h) / 2
    nrm = np.arctan2(u[:, 1] / (h / 2) ** 2, u[:, 0] / (w / 2) ** 2) + r; da = np.abs((P[:, 2] - nrm) % np.pi - np.pi / 2)  # tangent vs ridge direction
    return (d < tol) & (np.abs(da - np.pi / 2) > 0) & (np.abs((P[:, 2] - nrm) % np.pi - np.pi / 2) < np.deg2rad(15)), u

def centre_vote(P, shape, cell=4):
    """Gradient-Hough: every ridge point votes along its normal; the formation centre is the accumulator peak."""
    H, W = shape; n = P[:, 2] + np.pi / 2; t = np.arange(-.6, .6, cell / 1000) * max(H, W)
    X = (P[:, 0, None] + np.cos(n)[:, None] * t).ravel() / cell; Y = (P[:, 1, None] + np.sin(n)[:, None] * t).ravel() / cell
    ok = (X >= 0) & (X < W / cell) & (Y >= 0) & (Y < H / cell)
    acc = np.bincount((Y[ok].astype(int) * int(np.ceil(W / cell)) + X[ok].astype(int)), minlength=int(np.ceil(H / cell)) * int(np.ceil(W / cell)))
    acc = cv2.GaussianBlur(acc.reshape(int(np.ceil(H / cell)), int(np.ceil(W / cell))).astype(np.float32), (0, 0), 2)
    y, x = np.unravel_index(acc.argmax(), acc.shape); return np.array([x + .5, y + .5]) * cell

def ellipses(g, iters=3000, tol=2.0, cov_min=.5, max_rings=7, seed=0):
    """Rings of one formation: centre by gradient-Hough voting, then per ring a 3-point RANSAC for the shape matrix with the
    centre fixed (linear), tangent-consistent inliers, perimeter coverage check, full 5-parameter refit, sequential removal."""
    P, k = ridge_points(g, n=40000, seed=seed); rng = np.random.default_rng(seed); H, W = g.shape[0] * k, g.shape[1] * k; found = []
    c = centre_vote(P, (H, W))
    for _ in range(max_rings):
        U = P[:, :2] - c; rad = np.hypot(*U.T); cand = np.nonzero((rad > .06 * W) & (rad < .6 * W))[0]
        if len(cand) < 150: break
        best = (0, None)
        for _ in range(iters):
            S = U[rng.choice(cand, 3, replace=False)]
            try: q = np.linalg.solve(np.c_[S[:, 0] ** 2, 2 * S[:, 0] * S[:, 1], S[:, 1] ** 2], np.ones(3))
            except np.linalg.LinAlgError: continue
            Q = np.array([[q[0], q[1]], [q[1], q[2]]]); lam, V = np.linalg.eigh(Q)
            if lam.min() <= 0 or lam.max() / lam.min() > 9: continue
            a0, a1 = 1 / np.sqrt(lam)
            if found:                                   # shape prior: rings of one formation share aspect and orientation (+-15%, +-12deg)
                (_, (fw, fh), ft) = found[0]; asp, asp0 = a0 / a1, fw / fh; dth = abs((np.degrees(np.arctan2(V[1, 0], V[0, 0])) - ft + 90) % 180 - 90)
                if abs(asp / asp0 - 1) > .15 or (asp0 > 1.08 and dth > 12): continue
            el = ((c[0], c[1]), (2 * a0, 2 * a1), float(np.degrees(np.arctan2(V[1, 0], V[0, 0]))))
            inl, u = _fit_eval(P[cand], el, tol * 1.5); n = inl.sum()
            if n > best[0] and n > 50:
                ang = np.arctan2(u[inl, 1] / a1, u[inl, 0] / a0); cov = len(np.unique((ang // (2 * np.pi / 72)).astype(int))) / 72
                if cov >= cov_min: best = (n, el, cand[inl])
        if best[1] is None: break
        pts = P[best[2], :2].astype(np.float32)
        try: el = cv2.fitEllipse(pts)
        except cv2.error: break
        inl, _ = _fit_eval(P, el, tol); P = P[~inl]
        e = ((el[0][0] / k, el[0][1] / k), (el[1][0] / k, el[1][1] / k), el[2])
        if inl.sum() >= 50 and all(abs(e[1][0] / q[1][0] - 1) > .04 or np.hypot(e[0][0] - q[0][0], e[0][1] - q[0][1]) > .03 * q[1][0] for q in found): found.append(e)
    return sorted(found, key=lambda e: -e[1][0] * e[1][1])

def core(C1, C2):
    """Two concentric-circle conics (any scale) -> T (maps their plane to a metric frame where ring 1 is the unit circle),
    ring2/ring1 radius ratio and ring-2 roundness (1 = perfectly round). None if the pair is not concentric/coplanar-consistent."""
    C1 = C1 / np.linalg.norm(C1)
    if C2 is None: return _metric(C1, np.array([0., 0, 1]), None)          # affine-only fallback (outer ring assumed circular)
    C2 = C2 / np.linalg.norm(C2)
    ev, V = np.linalg.eig(np.linalg.inv(C1) @ C2)
    d = [abs(ev[(i + 1) % 3] - ev[(i + 2) % 3]) for i in range(3)]; k = int(np.argmin(d))   # k = the odd one out? (see below)
    # two eigenvalues coincide for concentric circles; the pair with the SMALLEST gap is the repeated one, the other index is the centre
    rep = [i for i in range(3) if i != k]; gap = abs(ev[rep[0]] - ev[rep[1]]); k = [i for i in range(3) if i not in rep][0]
    # NOTE: argmin picks index whose *other two* are closest => k is the distinct eigenvalue's index
    o = np.real(V[:, k]); l = C1 @ o; l = l / np.linalg.norm(l[:2])
    return _metric(C1, l, C2)

def _metric(C1, l, C2):
    Hp = np.array([[1, 0, 0], [0, 1, 0], l]); Hi = np.linalg.inv(Hp)
    Cr = Hi.T @ C1 @ Hi; Cr = Cr / -Cr[2, 2]; Q = (Cr[:2, :2] + Cr[:2, :2].T) / 2
    if np.linalg.eigvalsh(Q).min() <= 0: return None
    c0 = np.linalg.solve(Q, -Cr[:2, 2]); A = np.linalg.cholesky(Q).T
    T = np.block([[A, -(A @ c0)[:, None]], [np.zeros((1, 2)), np.ones((1, 1))]]) @ Hp
    if C2 is None: return T, None, None
    Ti = np.linalg.inv(T); C2p = Ti.T @ C2 @ Ti; C2p = C2p / -C2p[2, 2]; e2 = np.linalg.eigvalsh((C2p[:2, :2] + C2p[:2, :2].T) / 2)
    if e2.min() <= 0: return None
    return T, float(1 / np.sqrt(e2.mean())), float(np.sqrt(e2.min() / e2.max()))

def flatten(img, out=1000):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); h, w = g.shape; sc = max(h, w)
    N = np.array([[1 / sc, 0, -w / 2 / sc], [0, 1 / sc, -h / 2 / sc], [0, 0, 1]]); Ni = np.linalg.inv(N)
    es = ellipses(g)
    if not es: return None, "no outer ring found"
    C1 = Ni.T @ conic(es[0]) @ Ni; best = None
    for e2 in es[1:]:
        r = core(C1, Ni.T @ conic(e2) @ Ni)
        if r and r[1] and .15 < r[1] < .95 and (best is None or r[2] > best[2]): best = r      # concentric pair: full perspective removal
    method = "conic-pencil"
    if best is None or best[2] < .9: best, method = core(C1, None), "affine-only"
    if best is None: return None, "ring 1 not an ellipse"
    T, ratio, rnd = best; S = np.array([[out / 2.3, 0, out / 2], [0, out / 2.3, out / 2], [0, 0, 1]]); Hf = S @ T @ N
    return dict(img=cv2.warpPerspective(img, Hf, (out, out)), H=Hf, method=method, ring2_over_ring1=ratio, ring2_roundness=rnd), method

if __name__ == "__main__":
    import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from common import load, ROOT
    for r in load("tune", ("aerial",)):
        if not r["tag_overhead"]: continue
        im = cv2.imread(str(ROOT / r["file"])); o, why = flatten(im)
        print(r["file"].split("/")[-1][:45], why, {k: round(v, 3) for k, v in o.items() if k.startswith("ring") and v} if o else "")
        if o: cv2.imwrite(f"/tmp/claude-0/flat_{r['file'].split('/')[-1]}", np.hstack([cv2.resize(im, (o['img'].shape[1] * im.shape[1] // im.shape[0], o['img'].shape[0])), o["img"]]))
