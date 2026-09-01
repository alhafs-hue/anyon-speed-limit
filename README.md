# Verification code for *Anyon braiding at the speed limit*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22238459.svg)](https://doi.org/10.5281/zenodo.22238459)

Mohammad F. Marashdeh — Department of Mathematics, Mutah University, Al-Karak, Jordan

This repository reproduces every computational claim in the paper. Each script
corresponds to one or more numbered results, prints `PASS`/`FAIL` per claim, and
exits non-zero if anything fails.

## Running

```bash
pip install -r requirements.txt
python3 run_all.py            # full run, ~15-25 min
python3 run_all.py --fast     # reduced sample sizes, ~50 s
```

Python 3.11 with `numpy`, `scipy`, `sympy`, `mpmath`. No other dependencies, no
data files, no network access. Everything is deterministic: random scans use
fixed seeds, and the symbolic and exact-rational computations have no tolerance.

## What reproduces what

| Script | Paper result | What is checked |
|---|---|---|
| `check01_four_strand_spectrum.py` | Lem. 3.2, 3.4, 3.5, Thm. 3.6 | The sixteen closed-form eigenvalues of `M_2` against direct diagonalization at 500 random `(θ,d)`; the braid relation holds on `u = c/(1−c)` and fails off it |
| `check02_onecurve_collapse.py` | Thm. 4.1, Cor. 4.2, Cor. 4.7 | Symbolic collapse to the seven values; the quadratic factors as `(2x+1)(x−c(2c+1))`; no collapsed eigenvalue is `±1` for `c ∈ (−1,1/2)` |
| `check03_two_generator_bound.py` | Lem. 4.3, Thm. 4.4, Rem. 4.5 | Global minimum of `Δ(M_2)` over two-generator sets is `1/2`; three generators reach `1/3`; `{H,S^{±1}}` gives `cos(π/5)`; the Rayleigh bound and the three-quantity inconsistency, on 2·10⁵ points each |
| `check04_speed_off_curve.py` | Thm. 4.6, Rem. 4.8 | `Δ ≥ 1/2` on the rectangle apart from the two degenerate points; the Klein four-group spectrum at `(−1,0)`; the four attaining points |
| `check05_su2k_table.py` | Cor. 4.14, Table 1 | Every `d`, `c`, `q`, `c(2c+1)`, `Δ` entry for `k = 2,3,4,5,6,8,10` |
| `check06_design_orders.py` | Cor. 4.12, Thm. 4.16, Cor. 4.17, Prop. 4.18, Cor. 4.19 | Group orders 12/24/60; the integer frame potentials `F_2…F_6`; design orders 2/3/5; exact rational `spec(M_2)`, `spec(M_3)` for Ising; every crossing count |
| `check07_third_moment.py` | Thm. 4.15 | `dim Fix(M_3) = 5`; the quartic and its irreducibility; the closed form for `Δ(M_3)`; the length `L ≈ 4.94 log(1/ε)` |
| `check08_ising_grading.py` | Thm. 5.1, Prop. 5.2, Thm. 5.3, Thm. 5.4 | `dim V_n = 2^{n/2−1}`; braid relations; `Φ(σ_1)` is a signed permutation; `dim Fix(M_2) = ⌊n/4⌋+1` at `n = 4,6,8` |
| `check09_diffusive.py` | Lem. 6.7, Thm. 6.8, Rem. 6.10, App. A | The transposition graph has the one-particle gap `2−2cos(π/n)` at `n = 6,8,10,12`; the `n = 4` path on three vertices; the **exact** factorization of `det(xI−M_2)` at `n = 6` into sixteen irreducible factors over `Q`, giving `Δ(M_2) = (3+√3)/5` |
| `check10_all_n_table.py` | Prop. 6.1, Lem. 6.2, 6.3, Thm. 6.4, Table 2 | The Temperley–Lieb path model at higher `n`: dimensions, braid relations, unitarity, `λ_max(Π̄)` and `Δ(M_2)` |

`common.py` holds the shared constructions: the transfer matrix `Φ`, the
four-strand generators of eq. (4), the moment operators `M_t`, and the
Temperley–Lieb path model.

## Note on the `n = 6` factorization

`check09` builds `M_2` for Ising at `n = 6` from Majorana operators by
Jordan–Wigner, restricts to the even-parity sector, and factors the integral
matrix `10·M_2` over `Z` exactly. This is the computation printed in Appendix A
of the paper; it is exact, not numerical, and the script prints the sixteen
factors so they can be compared with the paper line by line.

## Citing

Please cite the paper and this archive:

> M. F. Marashdeh, *Anyon braiding at the speed limit* (2026).
>
> M. F. Marashdeh, *Verification code for "Anyon braiding at the speed limit"*,
> version 1.0.0, Zenodo (2026), DOI 10.5281/zenodo.22238459.

## License

MIT (see `LICENSE`).
