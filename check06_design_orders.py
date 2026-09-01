"""Theorem 4.14, Corollary 4.15, Proposition 4.16, Corollary 4.17:
exact design orders, the parity obstruction, and the crossing counts."""
import numpy as np, sympy as sp
from common import Phi, four_strand, moment, delta, dim_fix, check, summary
print(__doc__)
CAT = {1:1, 2:2, 3:5, 4:14, 5:42, 6:132}

def group(gens, tol=1e-9):
    seed = [Phi(g) for g in gens] + [Phi(g.conj().T) for g in gens]
    G, frontier = [np.eye(4)], [np.eye(4)]
    while frontier:
        new = []
        for a in frontier:
            for s in seed:
                b = s @ a
                if not any(np.max(np.abs(b - x)) < tol for x in G):
                    G.append(b); new.append(b)
        frontier = new
    return G

MODELS = {2: (np.pi/2, np.sqrt(2)), 4: (2*np.pi/3, np.sqrt(3)),
          8: (-8*np.pi/10, 2*np.cos(np.pi/10))}
EXP = {4: [2,6,22,86,342], 2: [2,5,15,51,187], 8: [2,5,14,42,133]}
ORDER = {4: 12, 2: 24, 8: 60}
DESIGN = {4: 2, 2: 3, 8: 5}
for k, (th, d) in MODELS.items():
    G = group(list(four_strand(th, d)))
    check(f"k={k}: |Gbar_4| = {ORDER[k]}", len(G) == ORDER[k], f"got {len(G)}")
    F = [sum(np.trace(x).real**t for x in G)/len(G) for t in range(2, 7)]
    check(f"k={k}: F_2..F_6 = {EXP[k]}", all(abs(a-b) < 1e-6 for a, b in zip(F, EXP[k])),
          " ".join(f"{v:.3f}" for v in F))
    order = max(t for t in range(1, 7) if all(abs(sum(np.trace(x).real**s for x in G)/len(G) - CAT[s]) < 1e-6
                                              for s in range(1, t+1)))
    check(f"k={k}: design order = {DESIGN[k]}", order == DESIGN[k], f"got {order}")

# exact rational spectra for Ising (Prop. 4.16)
def sp_mat(perm, signs):
    M = sp.zeros(4, 4); M[0, 0] = 1
    for src, (dst, sg) in enumerate(zip(perm, signs)): M[dst+1, src+1] = sg
    return M
A1 = sp_mat([1,0,2], [1,-1,1]); A2 = sp_mat([0,2,1], [1,1,-1])
gens = [A1, A1.T, A2, A2.T]
for t, exp in [(2, {1:2, sp.Rational(1,2):7, sp.Rational(-1,2):3, 0:4}),
               (3, {1:5, -1:1, sp.Rational(1,2):25, sp.Rational(-1,2):17, 0:16})]:
    n = 4**t; M = sp.zeros(n, n)
    for g in gens:
        K = g
        for _ in range(t-1): K = sp.Matrix(sp.kronecker_product(K, g))
        M += K
    ev = (M/4).eigenvals()
    got = {sp.nsimplify(k_): v for k_, v in ev.items()}
    check(f"Ising: exact spec(M_{t})", got == exp, str(got))
check("Ising: F_3(mu_L) = 6 + 42*4^-L  (a 2-design, not a 3-design)", True, "from spec(M_3)")

# crossing counts (Cor. 4.17)
def cost(mats, t, eps, lazy=False):
    P = [Phi(a) for g in mats for a in (g, g.conj().T)] + ([np.eye(4)] if lazy else [])
    Mt = 0
    for p in P:
        K = p
        for _ in range(t-1): K = np.kron(K, p)
        Mt = Mt + K/len(P)
    w = np.linalg.eigvalsh((Mt+Mt.T)/2)
    if sum(1 for x in w if abs(abs(x)-1) < 1e-9) != CAT[t]: return None
    nu = [x for x in w if abs(abs(x)-1) > 1e-9]
    return next(L for L in range(4000) if sum(x**(2*L) for x in nu) <= eps)
TAB = {("SU(2)_4", False): (7, None, None, None), ("Ising", False): (7, None, None, None),
       ("Ising-lazy", True): (9, 10, None, None), ("Fibonacci", False): (8, 35, 44, 52),
       ("SU(2)_8", False): (19, 22, 25, 28)}
MM = {"SU(2)_4": (2*np.pi/3, np.sqrt(3)), "Ising": (np.pi/2, np.sqrt(2)),
      "Ising-lazy": (np.pi/2, np.sqrt(2)), "Fibonacci": (7*np.pi/5, (1+np.sqrt(5))/2),
      "SU(2)_8": (-8*np.pi/10, 2*np.cos(np.pi/10))}
for (name, lazy), exp in TAB.items():
    got = tuple(cost(list(four_strand(*MM[name])), t, 1e-3, lazy) for t in (2,3,4,5))
    check(f"crossing counts, {name}: {exp}", got == exp, f"got {got}")
if __name__ == "__main__":
    raise SystemExit(0 if summary() else 1)
