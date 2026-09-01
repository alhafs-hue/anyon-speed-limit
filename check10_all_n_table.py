"""Proposition 6.1, Lemmas 6.3/6.4 and Table 3: the Temperley-Lieb path model
at higher strand number.  Pass --fast to stop at dim V_n <= 5.
"""
import sys, numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from common import su2k, fibonacci, check, summary
print(__doc__)
FAST = "--fast" in sys.argv

def delta_big(gens, k=10):
    D = gens[0].shape[0]
    A = [x for g in gens for x in (g, g.conj().T)]
    K = [np.kron(a, a) for a in A]; m = D*D
    def mv(v):
        T = (v[:m*m] + 1j*v[m*m:]).reshape(m, m)
        R = sum(Kk @ T @ Kk.conj().T for Kk in K)/len(K)
        return np.concatenate([R.real.ravel(), R.imag.ravel()])
    L = LinearOperator((2*m*m, 2*m*m), matvec=mv, dtype=float)
    if 2*m*m < 3000:
        Mt = np.zeros((2*m*m, 2*m*m))
        for i in range(2*m*m):
            e = np.zeros(2*m*m); e[i] = 1; Mt[:, i] = mv(e)
        w = np.sort(np.linalg.eigvalsh((Mt+Mt.T)/2))[::-1]
    else:
        w = np.sort(eigsh(L, k=min(k, 2*m*m-2), which='LA', tol=1e-11)[0])[::-1]
    nu = [x for x in w if abs(abs(x)-1) > 1e-7]
    return max(abs(np.array(nu)))

def pibar_max(gens, c):
    D = gens[0].shape[0]
    S = np.zeros((D*D, D*D), dtype=complex)
    for x in gens:
        for y in (x, x.conj().T): S += np.kron(y, y.conj())/(2*len(gens))
    v = np.eye(D).reshape(-1)/np.sqrt(D); Q = np.eye(D*D) - np.outer(v, v)
    w = np.sort(np.linalg.eigvalsh((Q@S@Q + (Q@S@Q).conj().T).real/2))[::-1]
    return (w[0]-c)/(1-c)

ROWS = [("Ising", lambda n: su2k(2, n), 0.0, [(4,2,0.5,0.5), (6,4,0.8,0.946410), (8,8,0.857143,0.978251)]),
        ("Fibonacci", fibonacci, (1-np.sqrt(5))/4, [(4,2,0.618034,0.618034), (6,5,0.926085,0.923918), (8,13,0.954656,0.962325)]),
        ("SU(2)_4", lambda n: su2k(4, n), -0.5, [(6,5,0.925206,0.921584)]),
        ("SU(2)_6", lambda n: su2k(6, n), -np.cos(2*np.pi/8), [(6,5,0.932566,0.924225)])]
print(f"\n  {'model':10s} {'n':>2s} {'dim':>4s} {'N':>2s}  {'lam_max(Pibar)':>14s} {'Delta(M_2)':>11s}")
for name, mk, c, exp in ROWS:
    for n, dimv, epi, edl in exp:
        if FAST and dimv > 5: continue
        D, g = mk(n)
        if D != dimv:
            check(f"{name} n={n}: dim V_n = {dimv}", False, f"got {D}"); continue
        check(f"{name} n={n}: dim V_n = {dimv}", True)
        br = max([np.max(np.abs(g[i]@g[i+1]@g[i]-g[i+1]@g[i]@g[i+1])) for i in range(len(g)-1)]+[0])
        un = max(np.max(np.abs(x@x.conj().T-np.eye(D))) for x in g)
        check(f"{name} n={n}: braid relations and unitarity", br < 1e-10 and un < 1e-10, f"{br:.1e}/{un:.1e}")
        N = len(g) if n > 4 else 2
        gg = g if n > 4 else [g[0], g[1]]
        pb = pibar_max(gg, c); dl = delta_big(gg)
        check(f"{name} n={n}: lam_max(Pibar) = {epi}", abs(pb-epi) < 2e-6, f"got {pb:.6f}")
        check(f"{name} n={n}: Delta(M_2) = {edl}", abs(dl-edl) < 2e-6, f"got {dl:.6f}")
        print(f"  {name:10s} {n:2d} {D:4d} {N:2d}  {pb:14.6f} {dl:11.6f}")
if __name__ == "__main__":
    raise SystemExit(0 if summary() else 1)
