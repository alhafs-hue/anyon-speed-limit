"""Section 5: the Majorana grading, the braid image, and dim Fix(M_2)."""
import numpy as np, itertools
from common import check, summary
print(__doc__)

def majorana(n):
    I2 = np.eye(2); X = np.array([[0,1],[1,0]],dtype=complex)
    Y = np.array([[0,-1j],[1j,0]]); Z = np.diag([1,-1]).astype(complex)
    def kr(*a):
        o = np.array([[1]], dtype=complex)
        for m in a: o = np.kron(o, m)
        return o
    m = n//2; g = []
    for k in range(m):
        pre = [Z]*k
        g.append(kr(*(pre+[X]+[I2]*(m-k-1)))); g.append(kr(*(pre+[Y]+[I2]*(m-k-1))))
    return g

def even_sector(n):
    g = majorana(n); m = n//2
    P = np.eye(2**m, dtype=complex)
    for k in range(n): P = P @ g[k]
    P = P*(1j)**(n*(n-1)//2)
    w, V = np.linalg.eigh(P)
    return g, V[:, w > 0]

for n in (4, 6, 8):
    g, Vp = even_sector(n); D = Vp.shape[1]
    check(f"n={n}: dim V_n = 2^(n/2-1) = {2**(n//2-1)}", D == 2**(n//2-1), f"got {D}")
    rho = [(np.eye(2**(n//2)) - g[j]@g[j+1])/np.sqrt(2) for j in range(n-1)]
    rr = [Vp.conj().T @ r @ Vp for r in rho]
    br = max([np.max(np.abs(rr[i]@rr[i+1]@rr[i] - rr[i+1]@rr[i]@rr[i+1])) for i in range(len(rr)-1)] + [0])
    check(f"n={n}: braid relations hold", br < 1e-10, f"residual {br:.2e}")
    # weight-preserving signed permutation of the monomial basis
    mats, keys = [], []
    for k in range(0, n+1, 2):
        for S in itertools.combinations(range(n), k):
            pm = np.eye(2**(n//2), dtype=complex)
            for xx in S: pm = pm @ g[xx]
            pm = pm*(1j)**(k*(k-1)//2)
            B = Vp.conj().T @ pm @ Vp
            M = np.array([x.reshape(-1) for x in mats] + [B.reshape(-1)])
            if np.linalg.matrix_rank(M, tol=1e-8) > len(mats): mats.append(B); keys.append(S)
    check(f"n={n}: dim End(V_n) = 2^(n-2) = {2**(n-2)}", len(mats) == 2**(n-2), f"got {len(mats)}")
    Eb = [B/np.sqrt(np.trace(B.conj().T@B).real/D) for B in mats]
    def Ph(u_):
        return np.array([[np.trace(Eb[a].conj().T@u_@Eb[b]@u_.conj().T).real/D
                          for b in range(len(Eb))] for a in range(len(Eb))])
    P0 = Ph(rr[0])
    sgnperm = all(sum(1 for v in row if abs(abs(v)-1) < 1e-8) == 1 and
                  np.max(np.abs(np.sort(np.abs(row))[:-1])) < 1e-8 for row in P0)
    check(f"n={n}: Phi(sigma_1) is a signed permutation matrix", sgnperm)
    A = [x for r in rr for x in (r, r.conj().T)]
    Ps = [Ph(a) for a in A]
    M2 = sum(np.kron(p, p) for p in Ps)/len(Ps)
    ev = np.linalg.eigvalsh((M2+M2.T)/2)
    df = int(sum(1 for x in ev if abs(x-1) < 1e-8))
    check(f"n={n}: dim Fix(M_2) = floor(n/4)+1 = {n//4+1}", df == n//4+1, f"got {df}")
if __name__ == "__main__":
    raise SystemExit(0 if summary() else 1)
