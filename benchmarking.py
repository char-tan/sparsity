import torch
import torch.utils.benchmark as benchmark


def make_fss(M, sparsity=(2,4)):
    """
    make M fine-structured-sparse by applying mask
    """

    print('\nsparsifying matrix')

    # only matricies
    assert len(M.shape) == 2

    rows, cols = M.shape

    # sparsity is defined as non_zero:total e.g 2:4
    non_zero, total = sparsity

    assert non_zero <= total
    assert cols % total == 0

    # find possible sparsity combinations for mini masks
    index_combinations = torch.combinations(torch.arange(total), r=non_zero)

    # produce possible mini masks
    mini_masks = torch.zeros((index_combinations.shape[0], total))
    for i in range(mini_masks.shape[0]):
        mini_masks[i,index_combinations[i]] = 1

    # random selection of mini masks to cover matrix
    mask_selection = torch.randint(index_combinations.shape[0], (rows, cols // total))

    # produce final mask
    mask = torch.zeros((rows, cols))
    for i in range(mask_selection.shape[0]):
        for j in range(mask_selection.shape[1]):
            mask[i,j*total:(j+1)*total] = mini_masks[mask_selection[i,j],:]

    print('done!\n')

    return M * mask


size = 1024  # TODO might want options for different size A, B
sparsity = (2, 4)
dtype = torch.float  # TODO need to think carefully about dtypes

A = torch.rand((size, size))
A_s = make_fss(A)
B = torch.rand((size, size))

timer = benchmark.Timer(
    stmt='torch.matmul(A, B)',
    setup='from torch import matmul',
    globals={'A': A, 'B': B})

print(timer.timeit(100))
