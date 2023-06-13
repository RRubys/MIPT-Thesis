import numpy as np


def commute(elem1, elem2):
    """
    Estimates commutator of two np.arrays;
    Obtains two np.arrays of shape n by n;
    Returns np.array of shape n by n.
    """
    return (elem1@elem2-elem2@elem1)


def get_basis_of_algebra(H0, V, tol=1e-3):
    """
    Generates dynamical Lie algebra of a system with Hamiltonians H0 and V, also returning its size;
    Obtains two np.arrays of shape n by n and the level of sensetivity (tolerance) for rank problem solution;
    Returns integer and list of np.arrays of shape n by n.
    """
    basis_matrix = np.concatenate((1j*H0.reshape((-1, 1)), 1j*V.reshape((-1, 1))), axis=1)
    n = H0.shape[0]
    basis_array = []
    cur_elem_ind = 0
    total_elems = 2
    
    while True:
        for another_elem_ind in range(0, cur_elem_ind):
            commutator = commute(basis_matrix[:, cur_elem_ind].reshape((n, n)), basis_matrix[:, another_elem_ind].reshape((n, n)))
            basis_matrix_update = np.concatenate((basis_matrix, commutator.reshape((-1, 1))), axis=1)
            if np.linalg.matrix_rank(basis_matrix_update, tol=tol) > total_elems:
                total_elems += 1
                basis_matrix = basis_matrix_update
        cur_elem_ind += 1
        if cur_elem_ind == min([total_elems, n**2]):
            break
    
    for elem_ind in range(basis_matrix.shape[1]):
        basis_array.append(basis_matrix[:, elem_ind].reshape((n, n)))
    
    return len(basis_array), basis_array
