import torch
import torchvision

import numpy as np
import pandas

from utils import resnet18_small_input
from myfirstcnn import MyFirstCNN


def seek_row_perm(p, p_pruned, mask):

    n_rows = p.shape[0]

    maxes = torch.zeros(n_rows)
    argmaxes = torch.zeros(n_rows, dtype=int)

    for i in range(n_rows):

        print(i)

        eqs = torch.zeros(n_rows)

        for j in range(n_rows):

            args = torch.argwhere(mask[j]).flatten()
            eqs[j] = torch.eq(p[i][args], p_pruned[j][args]).sum()

        maxes[i] = torch.max(eqs)
        argmaxes[i] = torch.argmax(eqs)

    return maxes, argmaxes


def find_layer_perm(p, p_pruned, mask):

    assert p.shape == p_pruned.shape

    orig_shape = p.shape

    p = p.view(orig_shape[0], -1)
    p_pruned = p_pruned.view(orig_shape[0], -1)
    mask = mask.view(orig_shape[0], -1)

    c_len, r_len = p.shape[:2]

    r_maxes, r_arg_maxes = seek_row_perm(p, p_pruned, mask)
    c_maxes, c_arg_maxes = seek_row_perm(p.t(), p_pruned.t(), mask.t())

    r_normalised_unmatched = (r_len - r_maxes.mean()) / r_len
    c_normalised_unmatched = (c_len - c_maxes.mean()) / c_len

    if r_normalised_unmatched < c_normalised_unmatched:
        perm = r_arg_maxes.long(), 'R'
    elif r_normalised_unmatched > c_normalised_unmatched:
        perm = c_arg_maxes.long(), 'C'
    elif torch.count_nonzero(mask) == mask.numel():
        # mask is all ones (no mask), can return either
        perm = r_arg_maxes.long(), 'R'
    else:
        print('problem!')
        assert ValueError

    return perm


def apply_layer_perm(p, perm):

    orig_shape = p.shape

    p = p.view(orig_shape[0], -1)

    #print(p.shape)
    #print(perm[0].shape)

    if perm[-1] == 'R':
        p = p[perm[0],:]
    elif perm[-1] == 'C':
        p = p[:,perm[0]]
    else:
        print('problem!')
        assert ValueError

    p = p.view(orig_shape)

    return p


torch.set_printoptions(linewidth=250)

phase1 = MyFirstCNN()
phase1_pruned = MyFirstCNN()

weight1 = torch.load("phase1_mf.pt", map_location="cpu")
weight1_pruned = torch.load("phase1_pruned_mf.pt", map_location="cpu")

phase1.load_state_dict(weight1)
phase1_pruned.load_state_dict(weight1_pruned, strict=False)

#print(weight1_pruned.keys())

layer_perms = dict()

with torch.no_grad():
    for (n, p), (n_pruned, p_pruned) in zip(
        phase1.named_parameters(), phase1_pruned.named_parameters()
    ):

        print('\nseeking permutaion for', n)

        if 'weight' in n:

            # check if there is a weight mask, if not mask all ones
            if n.replace('.weight', '.__weight_mma_mask') in weight1_pruned.keys():
                mask = weight1_pruned[n_pruned.replace('.weight', '.__weight_mma_mask')]
            else:
                mask = torch.ones_like(p)

            if "conv" in n: 
                perm = find_layer_perm(p, p_pruned, mask)

            elif 'linear' in n:
                # TODO - can't handle permuted linear layers
                perm = torch.arange(p.shape[0]), 'R'

            elif "bn" in n:
                # batch norm has same permutation as previous layer
                perm = perm
            
            elif "downsample" in n:
                # TODO handle DS - think is just same as conv
                perm = find_layer_perm(p, p_pruned, mask)
                breakpoint()

        elif "bias" in n:
            if perm[-1] == 'R':
                # bias has same permutation as previous layer
                perm = perm
            else:
                perm = torch.arange(len(p)), 'R'

        else:
            raise ValueError

        layer_perms[n] = perm

#print(layer_perms)

for n, p in phase1.named_parameters():

    print('\npermuting', n)

    old_p = p.clone()
    perm = layer_perms[n]
    p = apply_layer_perm(p, perm)

    assert p.shape == old_p.shape
    print(torch.eq(old_p, p).sum(), p.numel())

breakpoint()
