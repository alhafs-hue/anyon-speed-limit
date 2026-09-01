"""Theorem 4.1: on u = c/(1-c) the sixteen eigenvalues collapse to seven values."""
import numpy as np, sympy as sp
from common import four_strand, moment, check, summary
print(__doc__)

# symbolic collapse
c, u, x = sp.symbols('c u x')
la = sp.Rational(1,2)*(1 + c**2 - (1-c)**2*u**2)
lp = sp.Rational(1,2)*((2*c-1)*(c+1) + (2*c+1)*(c-1)*u)
lm = sp.Rational(1,2)*((2*c-1)*(c+1) - (2*c+1)*(c-1)*u)
Q  = 2*x**2 + (1-3*c**2-2*c-u**2-c**2*u**2+2*c*u**2)*x \
     + (3*c**3-c-2*u**2+c*u**2+4*c**2*u**2-3*c**3*u**2)
sub = {u: c/(1-c)}
check("lambda_a -> 1/2", sp.simplify(la.subs(sub) - sp.Rational(1,2)) == 0)
check("lambda_+ -> -1/2", sp.simplify(lp.subs(sub) + sp.Rational(1,2)) == 0)
check("lambda_- -> 2c^2+c-1/2", sp.simplify(lm.subs(sub) - (2*c**2+c-sp.Rational(1,2))) == 0)
check("quadratic factors as (2x+1)(x - c(2c+1))",
      sp.simplify(sp.expand(Q.subs(sub)) - sp.expand((2*x+1)*(x - c*(2*c+1)))) == 0)

# numerical collapse
worst = 0.0
for d in np.linspace(1.415, 1.999, 80):
    uu = 2/d**2 - 1; cc = uu/(1+uu)
    got = sorted(np.linalg.eigvalsh(moment(list(four_strand(np.arccos(cc), d)), 2)))
    pred = sorted([1,1] + [cc, cc+.5, .5]*3 + [.5, -.5, -.5, 2*cc*cc+cc-.5, cc*(2*cc+1)])
    worst = max(worst, float(np.max(np.abs(np.array(got)-np.array(pred)))))
check("collapsed spectrum vs direct diagonalization along the curve", worst < 1e-12, f"max dev {worst:.2e}")

# none of the six is a unit eigenvalue for c in (-1,1/2)   [proof of Cor. 4.3]
bad = []
for v in [c, c+sp.Rational(1,2), 2*c**2+c-sp.Rational(1,2), c*(2*c+1)]:
    for tgt in (1, -1):
        for r in sp.solve(sp.Eq(v, tgt), c):
            if r.is_real and -1 < float(r) < 0.5: bad.append((v, tgt, r))
check("no collapsed eigenvalue equals +-1 for c in (-1,1/2)", not bad, str(bad))
if __name__ == "__main__":
    raise SystemExit(0 if summary() else 1)
