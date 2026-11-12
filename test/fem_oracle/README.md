# Reference tables for local finite element spaces (C8)

`reference_tables.npz` is pinned test data: values, gradients and Hessians of Lagrange (equispaced and
GLL / warp-and-blend nodes), Bernstein and Q_k bases at fixed sample points in the reference cell and on
each face. `test/test_fem_oracle.py` asserts the properties the tables must have and compares `FemTools`
with them. The tests read the file only.

To regenerate (needs sympy and mpmath; about 5 s):

    python test/fem_oracle/generate_reference_tables.py

The generator's docstring has the provenance (the derivations are in `cekees/AnalyticalADR`,
`FEM_Basis_Construction_Sympy.ipynb`), the conventions (reference cells, orderings, face numbering) and the
table layout. Regenerating changes the file only if the construction changes; a diff in the tables is a
change of the oracle and needs review like a change of a reference solution.
