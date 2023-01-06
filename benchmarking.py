import torch
import torch.utils.benchmark as benchmark


def make_fss(M, sparsity=(2,4)):
    """
    make M fine-structured-sparse by applying mask
    """

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
    mask = mini_masks[mask_selection,:].flatten(1)

    return M * mask


size = 1024  # TODO might want options for different size A, B
sparsity = (2, 4)

A = torch.rand((size, size)) # TODO need to think carefully about dtypes

print('\nsparsifying matrix')
A = make_fss(A)
B = torch.rand((size, size))

# https://pytorch.org/docs/stable/benchmark_utils.html
timer = benchmark.Timer(
    stmt='torch.matmul(A, B)',
    setup='import torch',
    globals={'A': A, 'B': B},
    label='native PyTorch matmul')

print('\nbenchmarking\n')
print(timer.timeit(100))
