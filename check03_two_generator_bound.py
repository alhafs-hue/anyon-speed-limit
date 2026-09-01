"""Theorem 4.6 and Lemma 4.5: the two-generator speed limit on a qubit."""
import sys, numpy as np
from scipy.optimize import minimize
from common import Phi, moment, delta, check, summary
print(__doc__)
FAST = "--fast" in sys.argv
rng = np.random.default_rng(11)

def su2(p):
    a, b, c = p
    return (np.array([[np.exp(-1j*a/2), 0], [0, np.exp(1j*a/2)]])
            @ np.array([[np.cos(b/2), -np.sin(b/2)], [np.sin(b/2), np.cos(b/2)]])
            @ np.array([[np.exp(-1j*c/2), 0], [0, np.exp(1j*c/2)]]))

def f(p, k):
    return delta(moment([su2(p[3*i:3*i+3]) for i in range(k)], 2))

# (a) global minimum over two-generator symmetric gate sets is 1/2
best = 9.0
for _ in range(8 if FAST else 40):
    r = minimize(f, rng.uniform(0, 2*np.pi, 6), args=(2,), method='Nelder-Mead',
                 options={'maxiter': 6000, 'fatol': 1e-13, 'xatol': 1e-11})
    best = min(best, r.fun)
check("min Delta(M_2) over two-generator sets = 1/2", abs(best - 0.5) < 1e-7, f"min {best:.9f}")

# (b) three generators go strictly below 1/2 (Clifford value 1/3)
def rot(ax, th):
    s = [np.array([[0,1],[1,0]],dtype=complex), np.array([[0,-1j],[1j,0]]), np.diag([1,-1]).astype(complex)][ax]
    return np.cos(th/2)*np.eye(2) - 1j*np.sin(th/2)*s
cliff3 = delta(moment([rot(a, np.pi/2) for a in range(3)], 2))
check("three-generator Clifford set gives 1/3", abs(cliff3 - 1/3) < 1e-12, f"{cliff3:.12f}")

# (c) the same group with two generators {H, S^{+-1}} does not (Remark 4.9)
H = np.array([[1,1],[1,-1]], dtype=complex)/np.sqrt(2); S = np.diag([1, 1j])
d_HS = delta(moment([H, S], 2))
check("{H,S} gives cos(pi/5), not 1/3", abs(d_HS - np.cos(np.pi/5)) < 1e-9, f"{d_HS:.9f}")

# (d) the Rayleigh bound of Lemma 4.5 and the three-quantity inconsistency
ms = [2,1,0,-1,-2]; j = 2
Lz = np.diag(ms).astype(float); Lx = np.zeros((5,5))
for a, m in enumerate(ms):
    for b, mp in enumerate(ms):
        if m == mp+1: Lx[a,b] += np.sqrt(j*(j+1)-mp*(mp+1))/2
        if m == mp-1: Lx[a,b] += np.sqrt(j*(j+1)-mp*(mp-1))/2
def fpoly(cc, L):
    L2 = L@L
    return np.eye(5) - (cc-1)*(cc-7)/6*L2 + (cc-1)**2/6*(L2@L2)
viol_r, viol_c = 0, 0
for _ in range(20000 if FAST else 200000):
    k1, k2 = rng.uniform(0,1,2); w = rng.uniform(1e-9,1)
    c1, c2 = 1-2*k1, 1-2*k2; u = np.sqrt(max(1-w,0))
    M5 = 0.5*(fpoly(c1,Lz) + fpoly(c2, np.sqrt(1-u*u)*Lx + u*Lz))
    A1, A2 = 1-2*k1*w, 1-2*k2*w
    Lam = (0.5*(A1*A1+A2*A2) + 2*w - 1)/(2*w)
    if Lam > max(np.linalg.eigvalsh(M5)) + 1e-9: viol_r += 1
    S = k1+k2; DK = np.sqrt(max(S*S - 4*k1*k2*w, 0))
    if abs(1-S) < .5 and abs(1-(S-DK)/2) < .5 and abs(Lam) < .5: viol_c += 1
check(f"Rayleigh bound Lambda <= lambda_max(M_5), {20000 if FAST else 200000} points", viol_r == 0, f"{viol_r} violations")
check(f"three-quantity inconsistency, {20000 if FAST else 200000} points", viol_c == 0, f"{viol_c} violations")
if __name__ == "__main__":
    raise SystemExit(0 if summary() else 1)
