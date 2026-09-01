"""Theorem 3.4 and Lemmas 3.6, 3.7: the closed-form spectrum of M_2 on V_4."""
import numpy as np
from common import four_strand, moment, check, summary

def closed_form(theta, d):
    c = np.cos(theta); u = 2 / d**2 - 1; kap = (1 - c) / 2; v = abs(u)
    ad = [1 - 2*kap, 1 - kap*(1 - v), 1 - kap*(1 + v)]           # eq. (3.4)
    la = 0.5*(1 + c**2 - (1 - c)**2 * u**2)                       # eq. (3.5)
    lp = 0.5*((2*c - 1)*(c + 1) + (2*c + 1)*(c - 1)*u)            # eq. (3.6)
    lm = 0.5*((2*c - 1)*(c + 1) - (2*c + 1)*(c - 1)*u)
    B = 1 - 3*c**2 - 2*c - u**2 - c**2*u**2 + 2*c*u**2            # eq. (3.7)
    C = 3*c**3 - c - 2*u**2 + c*u**2 + 4*c**2*u**2 - 3*c**3*u**2
    q = np.roots([2, B, C])
    return sorted([1, 1] + ad*3 + [la, lp, lm] + list(np.real(q)))

print(__doc__)
rng = np.random.default_rng(0); worst = 0.0
for _ in range(500):
    th = rng.uniform(0, 2*np.pi); d = rng.uniform(1.001, 1.999)
    got = sorted(np.linalg.eigvalsh(moment(list(four_strand(th, d)), 2)))
    worst = max(worst, float(np.max(np.abs(np.array(got) - np.array(closed_form(th, d))))))
check("closed form vs direct diagonalization, 500 random (theta,d)", worst < 1e-9, f"max dev {worst:.2e}")

# the braid relation holds exactly on u = c/(1-c) and nowhere else
res_on, res_off = 0.0, 1.0
for d in [np.sqrt(2), np.sqrt(3), (1+np.sqrt(5))/2, 1.7, 1.9]:
    u = 2/d**2 - 1; c = u/(1+u); th = np.arccos(c)
    g1, g2 = four_strand(th, d)
    res_on = max(res_on, float(np.max(np.abs(g1@g2@g1 - g2@g1@g2))))
    h1, h2 = four_strand(th + 0.3, d)
    res_off = min(res_off, float(np.max(np.abs(h1@h2@h1 - h2@h1@h2))))
check("braid relation holds on the curve u = c/(1-c)", res_on < 1e-12, f"residual {res_on:.2e}")
check("braid relation fails off the curve", res_off > 1e-2, f"residual {res_off:.2e}")
if __name__ == "__main__":
    raise SystemExit(0 if summary() else 1)
