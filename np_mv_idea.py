import numpy as np
import time as t

def generate_sparse_matrix(shape, sparsity, dtype):
    """
    random sparse matrix of ones and zeros
    """

    assert shape[1] // sparsity

    non_zero_per_row = int(shape[1] * (1 - sparsity))

    rows = []

    possible_indices = np.arange(shape[1])

    for i in range(shape[0]):

        non_zero_indices = np.random.choice(possible_indices, size=non_zero_per_row, replace=False)

        row = np.zeros(shape[1], dtype=dtype)
        row[non_zero_indices] = 1

        rows.append([row])

    return np.concatenate(rows)

class SparseMatrix:
    def __init__(self, M, sparsity=0.5):
        """
        general idea here is that the matrix of weights does not change once trained
        so can store the compressed version with indicies during inference
        real question is how much we can optimise / parallelise the region below
        """

        self.M = M

        self.M_star, self.I = self._make_small_and_dense(M, sparsity=sparsity)

    def _make_small_and_dense(self, M, sparsity):
        # this function will currently fall apart if the matrix has too many zero values :)

        """
        args
        M: np.array of structured sparse matrix
        sparsity: float of sparsity level

        returns
        data_matrix: np.array of compressed dense matrix
        index_matrix: np.array of row indicies
        """
    
        index_list = []
        row_list = []
    
        for row in M:
            
            # get indicies and compressed row
            indices = np.nonzero(row)[0]  # returns tuple
            compressed_row = row[indices]
    
            # checks each row has the required sparsity
            assert 1 - len(indices)/len(row) == sparsity
    
            # append to lists
            index_list.append([indices])
            row_list.append([compressed_row])
    
        # list -> np.array
        index_matrix = np.concatenate(index_list)
        data_matrix = np.concatenate(row_list)
    
        return data_matrix, index_matrix

    def product(self, v):
        """
        args
        v: np.array of vector

        returns
        self.M @ v: np.array
        """

        output_vals = []

        ################################# OPTIMISABLE? PARELLELISABLE?

        """
        I'm thinking it must be possible to parallelise this
        Intuitively I'm thinking the for loop below can be done in parallel on GPU
        I don't know much about computer hardware though, not sure if it would be fast to
        get N copies of the vector each sliced with different indicies
        """

        # iterate over rows
        for i in range(self.M.shape[0]):

            # get relevant row of M
            M_row = self.M_star[i]

            # get required values from v
            v_row = v[self.I[i]]

            # dot product
            output_vals.append(M_row @ v_row)

        ################################# OPTIMISABLE? PARELLELISABLE?

        # one nice thing about mv product is you don't need to reconstruct anything :)

        return np.array(output_vals)

# simulating 2048 x 2048 matrix product with 2048 vector
# repeating 512 times e.g batch size

size = 2048
sparsity = 0.5
dtype = int
repeats = 512

### random matrix of ones and zeros ###

A = generate_sparse_matrix((size, size), sparsity, dtype)

A_sparse = SparseMatrix(A, sparsity=sparsity)
b = np.ones(size).astype(dtype)

time = t.time()

for _ in range(repeats):
    c = A @ b

print('np alg', t.time() - time)

time = t.time()

for _ in range(repeats):
    c = A_sparse.product(b)

print('our alg', t.time() - time)
