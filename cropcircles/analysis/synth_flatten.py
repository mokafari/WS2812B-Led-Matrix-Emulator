"""Step-3 gate for flatten.core: concentric circles of known radius ratio seen through random perspective cameras."""
import numpy as np, sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent)); from flatten import core, core_vp
rng = np.random.default_rng(0)
def cam(tilt, f, yaw):
    R = lambda a, ax: np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]]) if ax=='x' else np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    Rm = R(np.deg2rad(tilt), 'x') @ R(np.deg2rad(yaw), 'z'); K = np.diag([f, f, 1]); t = np.array([0, 0, 1.0])  # ground z=0 at distance 1... scale free
    return K @ np.c_[Rm[:, 0], Rm[:, 1], np.array([0, 0, 4.0])]
def conic_circle(r): return np.diag([1, 1, -r * r])
for tilt in (0, 10, 20, 30, 40, 50):
    errs, rnds, aff, vpe = [], [], [], []
    for _ in range(200):
        ratio = rng.uniform(1.2, 3); H = cam(tilt, rng.uniform(1.5, 6), rng.uniform(0, 360)); Hi = np.linalg.inv(H)
        # add conic noise equal to ~0.5% of axis length
        C1, C2 = [Hi.T @ conic_circle(r) @ Hi for r in (1, ratio)]
        P = lambda C: C + rng.normal(0, 3e-3, (3, 3)) * np.abs(C).max(); sym = lambda C: (C + C.T) / 2
        out = core(sym(P(C1)), sym(P(C2)))
        T = core(sym(P(C1)), None)[0]; Ti = np.linalg.inv(T); Cp = Ti.T @ P(C2) @ Ti; Cp = Cp / -Cp[2, 2]; e2 = np.linalg.eigvalsh((Cp[:2, :2] + Cp[:2, :2].T) / 2)
        v = H @ np.r_[np.cos(phi := rng.uniform(0, np.pi)), np.sin(phi), 0]; v = v[:2] / v[2] if abs(v[2]) > 1e-9 else np.array([1e6, 1e6])
        Tv = core_vp(sym(P(C1)), v + rng.normal(0, 0.01, 2) * 0)
        if Tv and Tv[0] is not None:
            Tm = np.linalg.inv(Tv[0]); Cq = Tm.T @ P(C2) @ Tm; Cq = Cq / -Cq[2, 2]; e3 = np.linalg.eigvalsh((Cq[:2, :2] + Cq[:2, :2].T) / 2)
            vpe.append(abs(1 / np.sqrt(e3.mean()) / ratio - 1) if e3.min() > 0 else np.nan)
        aff.append(abs(1 / np.sqrt(e2.mean()) / ratio - 1) if e2.min() > 0 else np.nan)
        if out: errs.append(abs(out[1] / ratio - 1)); rnds.append(out[2])
        else: errs.append(np.nan)
    e = np.array(errs); print(f"tilt {tilt:2d}deg  ok {np.mean(~np.isnan(e)):.2f}  median |ratio err| {np.nanmedian(e):.4f}  p90 {np.nanpercentile(e,90):.4f}  within1% {np.nanmean(e[~np.isnan(e)]<.01):.2f} | affine-only median {np.nanmedian(aff):.4f} p90 {np.nanpercentile(aff,90):.4f} within1% {np.nanmean(np.array(aff)[~np.isnan(aff)]<.01):.2f} | ring+tramline VP median {np.nanmedian(vpe):.4f} p90 {np.nanpercentile(vpe,90):.4f} within1% {np.nanmean(np.array(vpe)[~np.isnan(vpe)]<.01):.2f}")
