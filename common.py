"""Shared constructions for the verification scripts.

Conventions follow the paper:
  Phi(g)_{ab} = tr(E_a g E_b g^dagger)/D          (transfer matrix, eq. 2.1)
  M_t = (1/|A|) sum_{a in A} Phi(a)^{tensor t}    (moment operator)
  Delta(M) = |lambda_2(M)|                        (largest modulus below 1)
"""
import numpy as np

# ---- qubit (D = 2) -------------------------------------------------------
PAULI = [np.eye(2, dtype=complex),
         np.array([[0, 1], [1, 0]], dtype=complex),
         np.array([[0, -1j], [1j, 0]]),
         np.diag([1, -1]).astype(complex)]


def Phi(g, basis=None, D=None):
    """Transfer matrix of g in a Hermitian orthonormal operator basis."""
    E = PAULI if basis is None else basis
    D = 2 if D is None else D
    return np.array([[np.trace(E[a].conj().T @ g @ E[b] @ g.conj().T).real / D
                      for b in range(len(E))] for a in range(len(E))])


def four_strand(theta, d):
    """Braid generators on V_4 in the fusion basis, eq. (3.1)."""
    g1 = np.diag([1, np.exp(1j * theta)])
    s = np.sqrt(max(d * d - 1, 0))
    F = np.array([[1 / d, s / d], [s / d, -1 / d]], dtype=complex)
    return g1, F @ g1 @ F


def moment(mats, t, weights=None):
    """M_t from a list of unitaries (each entering with its inverse)."""
    A = [x for g in mats for x in (g, g.conj().T)]
    P = [Phi(a) for a in A]
    w = np.ones(len(P)) / len(P) if weights is None else np.asarray(weights, float)
    out = 0
    for wi, p in zip(w, P):
        K = p
        for _ in range(t - 1):
            K = np.kron(K, p)
        out = out + wi * K
    return out


def delta(M, tol=1e-9):
    """Delta(M) = largest eigenvalue modulus strictly below 1."""
    ev = np.linalg.eigvalsh((M + M.T) / 2)
    nu = [x for x in ev if abs(abs(x) - 1) > tol]
    return max(abs(np.array(nu))) if nu else 0.0


def dim_fix(M, tol=1e-9):
    ev = np.linalg.eigvalsh((M + M.T) / 2)
    return int(sum(1 for x in ev if abs(x - 1) < tol))


# ---- Temperley-Lieb path model (Prop. 7.1) -------------------------------
def tl_model(adj, dvec, dq, start, n, theta):
    """Braid generators on the path model of the fusion graph `adj`."""
    V = len(adj)
    P = [(start,)]
    for _ in range(n):
        P = [p + (q,) for p in P for q in range(V) if adj[p[-1]][q]]
    P = [p for p in P if p[-1] == start]
    idx = {p: i for i, p in enumerate(P)}
    D = len(P)
    gens = []
    for i in range(1, n):
        E = np.zeros((D, D))
        for p in P:
            if p[i - 1] != p[i + 1]:
                continue
            for q in range(V):
                if not adj[p[i - 1]][q]:
                    continue
                pp = p[:i] + (q,) + p[i + 1:]
                if pp in idx:
                    E[idx[pp], idx[p]] = np.sqrt(dvec[q] * dvec[p[i]]) / dvec[p[i - 1]]
        gens.append(np.eye(D, dtype=complex) + (np.exp(1j * theta) - 1) * (E / dq))
    return D, gens


def su2k(k, n):
    """Fundamental object of SU(2)_k on n strands."""
    r = k + 2
    adj = [[1 if abs(a - b) == 1 else 0 for b in range(k + 1)] for a in range(k + 1)]
    dv = [np.sin((j + 1) * np.pi / r) / np.sin(np.pi / r) for j in range(k + 1)]
    return tl_model(adj, dv, 2 * np.cos(np.pi / r), 0, n, -k * np.pi / (k + 2))


def fibonacci(n):
    """Fibonacci category (fusion graph 1 - tau with a loop at tau)."""
    phi = (1 + np.sqrt(5)) / 2
    return tl_model([[0, 1], [1, 1]], [1, phi], phi, 0, n, 7 * np.pi / 5)


# ---- reporting -----------------------------------------------------------
_RESULTS = []


def check(label, ok, detail=""):
    _RESULTS.append((label, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    return ok


def summary():
    bad = [r for r in _RESULTS if not r[1]]
    print(f"\n  {len(_RESULTS) - len(bad)}/{len(_RESULTS)} checks passed")
    return len(bad) == 0
