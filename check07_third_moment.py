"""Theorem 4.13: the third-moment rate of Fibonacci braiding."""
import numpy as np, sympy as sp
from common import four_strand, moment, delta, dim_fix, check, summary
print(__doc__)
M3 = moment(list(four_strand(7*np.pi/5, (1+np.sqrt(5))/2)), 3)
w = np.sort(np.linalg.eigvalsh((M3+M3.T)/2))[::-1]
check("dim Fix(M_3) = 5 (the Haar value C_3)", dim_fix(M3) == 5, f"got {dim_fix(M3)}")
D3 = delta(M3)
x = sp.symbols('x'); quartic = 64*x**4 - 224*x**3 - 148*x**2 + 202*x + 61
check("Delta(M_3) is a root of 64x^4-224x^3-148x^2+202x+61", abs(float(quartic.subs(x, D3))) < 1e-9,
      f"value {float(quartic.subs(x, D3)):.2e}")
check("that quartic is irreducible over Q", len(sp.factor_list(quartic)[1]) == 1)
fac = sp.factor_list(quartic, extension=sp.sqrt(5))[1]
check("it factors over Q(sqrt 5) into two quadratics", len(fac) == 2 and all(sp.degree(f) == 2 for f, _ in fac))
closed = 7/8 - np.sqrt(5)/4 + np.sqrt(201 - 80*np.sqrt(5))/8
check("closed form 7/8 - sqrt5/4 + sqrt(201-80 sqrt5)/8", abs(D3 - closed) < 1e-12, f"{D3:.13f}")
check("L ~ 4.94 log(1/eps)", abs(-1/(2*np.log(D3)) - 4.9438) < 1e-3, f"{-1/(2*np.log(D3)):.4f}")
if __name__ == "__main__":
    raise SystemExit(0 if summary() else 1)
