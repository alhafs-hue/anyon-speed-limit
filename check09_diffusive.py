"""Lemma 6.7, Theorem 6.8: the exclusion process, and the exact n=6 spectrum.

The n=6 factorization is the slow part; pass --fast to skip it.
"""
import sys, itertools, numpy as np, sympy as sp
from common import check, summary
print(__doc__)
FAST = "--fast" in sys.argv

# (a) the transposition graph has the one-particle gap (Aldous)
for n in (6, 8, 10, 12):
    subs = [frozenset(s) for s in itertools.combinations(range(n), 2)]
    ix = {s: i for i, s in enumerate(subs)}; m = len(subs)
    L = np.zeros((m, m))
    for s in subs:
        i = ix[s]
        for b in range(n-1):
            t = set(s)
            if (b in t) != ((b+1) in t): t.symmetric_difference_update({b, b+1})
            j = ix[frozenset(t)]; L[j, i] += 1; L[i, i] -= 1
    gap = np.sort(np.linalg.eigvalsh(-(L+L.T)/2))[1]
    one = 2 - 2*np.cos(np.pi/n)
    check(f"n={n}: gap(L_G) = 2-2cos(pi/n) = {one:.8f}", abs(gap-one) < 1e-9, f"got {gap:.8f}")

# (b) the four-strand case: a path on three vertices, gap 1 (Remark 6.10)
reps = sorted({min(tuple(sorted(s)), tuple(sorted(set(range(4))-set(s))))
               for s in itertools.combinations(range(4), 2)})
ix = {r: i for i, r in enumerate(reps)}; L = np.zeros((3, 3))
for r in reps:
    i = ix[r]
    for b in (0, 1):
        t = set(r)
        if (b in t) != ((b+1) in t): t.symmetric_difference_update({b, b+1})
        c = min(tuple(sorted(t)), tuple(sorted(set(range(4))-t)))
        L[ix[c], i] += 1; L[i, i] -= 1
w = np.sort(np.linalg.eigvalsh(-(L+L.T)/2))
check("n=4: path on 3 vertices, Laplacian gap 1, so Delta >= 1/2", abs(w[1]-1) < 1e-12 and abs(1-w[1]/2-0.5) < 1e-12)

# (c) exact characteristic polynomial of M_2 for Ising at n=6
if not FAST:
    I2 = np.eye(2); X = np.array([[0,1],[1,0]],dtype=complex)
    Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1,-1]).astype(complex)
    def kr(*a):
        o = np.array([[1]], dtype=complex)
        for mm in a: o = np.kron(o, mm)
        return o
    n = 6; mq = n//2; g = []
    for k in range(mq):
        pre = [Z]*k
        g.append(kr(*(pre+[X]+[I2]*(mq-k-1)))); g.append(kr(*(pre+[Y]+[I2]*(mq-k-1))))
    P = np.eye(2**mq, dtype=complex)
    for k in range(n): P = P @ g[k]
    P = P*(1j)**(n*(n-1)//2)
    wv, V = np.linalg.eigh(P); Vp = V[:, wv > 0]; D = Vp.shape[1]
    rho = [(np.eye(2**mq)-g[j]@g[j+1])/np.sqrt(2) for j in range(n-1)]
    rr = [Vp.conj().T@r@Vp for r in rho]
    mats = []
    for k in range(0, n+1, 2):
        for S in itertools.combinations(range(n), k):
            pm = np.eye(2**mq, dtype=complex)
            for xx in S: pm = pm@g[xx]
            pm = pm*(1j)**(k*(k-1)//2)
            B = Vp.conj().T@pm@Vp
            M = np.array([x.reshape(-1) for x in mats]+[B.reshape(-1)])
            if np.linalg.matrix_rank(M, tol=1e-8) > len(mats): mats.append(B)
    Eb = [B/np.sqrt(np.trace(B.conj().T@B).real/D) for B in mats]
    def Ph(u_):
        return np.array([[np.trace(Eb[a].conj().T@u_@Eb[b]@u_.conj().T).real/D
                          for b in range(len(Eb))] for a in range(len(Eb))])
    A = [x for r in rr for x in (r, r.conj().T)]
    M2 = sum(np.kron(Ph(a), Ph(a)) for a in A)/len(A)
    Zi = np.rint(10*M2).astype(np.int64)
    check("10*M_2 is an integer matrix", np.max(np.abs(10*M2-Zi)) < 1e-9)
    x = sp.symbols('x')
    cp = sp.Matrix(Zi.tolist()).charpoly(x).as_expr()
    fl = sp.factor_list(sp.expand(cp.subs(x, 10*x)/10**256))[1]
    check("16 irreducible factors of total degree 256", len(fl) == 16 and
          sum(sp.degree(f)*e for f, e in fl) == 256, f"{len(fl)} factors")
    rts = [(r, e) for f, e in fl for r in sp.Poly(f, x).all_roots()]
    df = sum(e for f, e in fl for r in sp.Poly(f, x).all_roots() if abs(complex(r)-1) < 1e-12)
    check("dim Fix(M_2) = 2, no eigenvalue -1", df == 2 and
          not any(abs(complex(r)+1) < 1e-12 for r, e in rts))
    nz = sorted([(abs(complex(r)), r) for r, e in rts if abs(abs(complex(r))-1) > 1e-12], reverse=True)
    check("Delta(M_2) = (3+sqrt3)/5", sp.simplify(nz[0][1] - (3+sp.sqrt(3))/5) == 0,
          f"{nz[0][0]:.13f}")
    check("next largest modulus 0.885454...", abs(nz[1][0]-0.8854540388) < 1e-9, f"{nz[1][0]:.10f}")
    print("\n  factorization of det(xI - M_2) at n = 6:")
    for f, e in fl: print(f"    ({sp.factor(f)})^{e}")
if __name__ == "__main__":
    raise SystemExit(0 if summary() else 1)
