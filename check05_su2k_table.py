"""Table 1: the SU(2)_k family on four strands."""
import numpy as np
from common import four_strand, moment, delta, dim_fix, check, summary
print(__doc__)
exp_q = {2:4, 3:10, 4:3, 5:14, 6:8, 8:5, 10:12}
exp_D = {2:0.5, 3:(np.sqrt(5)-1)/2, 4:0.5, 5:np.cos(2*np.pi/7), 6:1/np.sqrt(2),
         8:(1+np.sqrt(5))/4, 10:np.sqrt(3)/2}
rows = []
for k in [2,3,4,5,6,8,10]:
    d = 2*np.cos(np.pi/(k+2)); c = -np.cos(2*np.pi/(k+2)); u = 2/d**2 - 1
    M = moment(list(four_strand(np.arccos(c), d)), 2)
    q = next(qq for qq in range(1, 200) if abs((qq*k/(k+2)) % 2) < 1e-9 or abs((qq*k/(k+2)) % 2 - 2) < 1e-9)
    rows.append((k, d, c, q, c*(2*c+1), delta(M), dim_fix(M)))
    check(f"k={{}}: on the curve".format(k) if False else f"k={k}: u = c/(1-c)", abs(u - c/(1-c)) < 1e-12)
    check(f"k={k}: q = {exp_q[k]}", q == exp_q[k], f"got {q}")
    check(f"k={k}: Delta = {exp_D[k]:.6f}", abs(delta(M)-exp_D[k]) < 1e-9, f"got {delta(M):.6f}")
    check(f"k={k}: dim Fix = 2", dim_fix(M) == 2)
print("\n   k      d          c         q    c(2c+1)     Delta")
for k,d,c,q,g,dl,df in rows:
    print(f"  {k:2d}  {d:.6f}  {c:+.6f}  {q:3d}  {g:+.6f}  {dl:.6f}")
if __name__ == "__main__":
    raise SystemExit(0 if summary() else 1)
