"""SparsityInfo.getCSR on a pattern with a row nobody couples (a hole in the row numbers).

getCSR used to size rowptr by the map's size and read the rows with operator[], which inserts the missing rows
and grows the map under its own loop bound: rowptr[I] then ran off its allocation (a heap overrun, found by
test_cube_mms[3-3-4-element-1] under valgrind and glibc's abort at exit). Rows are now 0..largest key; a row
nobody inserted is empty. Run it under valgrind to see the difference (the unfixed code reports an invalid write).
"""
import numpy as np
import pytest

csparsity = pytest.importorskip("proteus.csparsity")
from proteus.csparsity import PySparsityInfo


def _find(element_dofs, index_dtype):
    nE, nd = len(element_dofs), len(element_dofs[0])
    free = np.asarray(element_dofs, dtype=index_dtype).reshape(-1)
    nfree = np.full(nE, nd, dtype=np.intc)
    empty_i = np.zeros(1, dtype=index_dtype)
    empty_c = np.zeros(1, dtype=np.intc)
    info = PySparsityInfo()
    info.findNonzeros(nE, nd, nd, nfree, free, nfree, free, 0, 1, 0, 1, 0, 0, 0, 2,
                      empty_i, 0, empty_i, empty_i, empty_c, 0, 0, empty_i, 0, 0)
    return info


@pytest.mark.parametrize("element_dofs,rows", [
    ([[0, 1], [1, 2]], 3),                 # no hole
    ([[0, 1], [3, 4]], 5),                 # row 2 never inserted
    ([[2, 3], [3, 4]], 5),                 # rows 0 and 1 never inserted
    ([[0, 1], [5, 6], [9, 10]], 11),       # several holes
])
def test_rows_are_numbered_by_the_largest_key_and_holes_are_empty(element_dofs, rows):
    # proteus_index_t is the mesh index type; try the native 32-bit first, then 64-bit
    for dt in (np.int32, np.int64):
        try:
            info = _find(element_dofs, dt)
            break
        except (ValueError, TypeError):
            continue
    rowptr, colind, nnz, nzval = info.getCSR()
    assert len(rowptr) == rows + 1
    assert rowptr[0] == 0 and np.all(np.diff(rowptr) >= 0)
    assert nnz == rowptr[-1] == len(colind) == len(nzval)
    present = {int(d) for el in element_dofs for d in el}
    for I in range(rows):
        cols = colind[rowptr[I]:rowptr[I + 1]]
        if I in present:
            expected = sorted({int(j) for el in element_dofs if I in el for j in el})
            assert list(cols) == expected
        else:
            assert len(cols) == 0
