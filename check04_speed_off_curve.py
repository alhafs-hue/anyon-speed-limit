"""Theorem 4.10 and Remark 4.11: the bound off the curve, and its four minimizers."""
import sys, numpy as np
from common import four_strand, moment, delta, dim_fix, check, summary
print(__doc__)
FAST = "--fast" in sys.argv
D = lambda c, u: delta(moment(list(four_strand(np.arccos(np.clip(c,-1,1)), np.sqrt(2/(1+u)))), 2))

bad = []
for c in np.linspace(-1, 1, 61 if FAST else 161):
    for u in np.linspace(-0.4999, 0.9999, 41 if FAST else 121):
        if D(c, u) < 0.5 - 1e-7: bad.append((round(c,4), round(u,4)))
exc = [p for p in bad if not (abs(p[0]-1) < 1e-9 or (abs(p[0]+1) < 1e-9 and abs(p[1]) < 1e-9))]
check("Delta >= 1/2 on the rectangle apart from c=1 and (c,u)=(-1,0)", not exc, str(exc[:5]))

M = moment(list(four_strand(np.pi, np.sqrt(2))), 2)
ev = np.sort(np.linalg.eigvalsh(M))
ok = (sum(1 for x in ev if abs(x-1)<1e-9) == 4 and sum(1 for x in ev if abs(x+1)<1e-9) == 4
      and sum(1 for x in ev if abs(x)<1e-9) == 8)
check("spec(M_2) at (c,u)=(-1,0) is {1^4,(-1)^4,0^8}", ok)
check("F_2(mu_L) = 8 there for L>=1", abs(sum(x**2 for x in ev) - 8) < 1e-9)

targets = [(0.0, 0.0), (-0.5, -1/3), (-0.5, 1/3), (-1.0, 0.5)]
vals = [D(*t) for t in targets]
check("the four listed points attain Delta = 1/2", all(abs(v-0.5) < 1e-9 for v in vals),
      "  ".join(f"{t}->{v:.9f}" for t, v in zip(targets, vals)))
grid = [(c, u) for c in np.linspace(-0.999, 0.999, 41 if FAST else 121) for u in np.linspace(-0.4999, 0.9999, 31 if FAST else 91)]
extra = [(round(c,3), round(u,3)) for c, u in grid if abs(D(c,u)-0.5) < 1e-4
         and not any(abs(c-t[0]) < .05 and abs(u-t[1]) < .05 for t in targets)]
check("no other point of the interior grid attains 1/2", not extra, str(extra[:5]))
if __name__ == "__main__":
    raise SystemExit(0 if summary() else 1)
