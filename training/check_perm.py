import torch
import torchvision

import numpy as np
import pandas

from utils import resnet18_small_input
from myfirstcnn import MyFirstCNN


def seek_row_perm(p, p_pruned, mask):

    n_rows = p.shape[0]

    maxes = torch.zeros(n_rows)
    argmaxes = torch.zeros(n_rows)

    for i in range(n_rows):

        eqs = torch.zeros(n_rows)

        for j in range(n_rows):

            print(i, j)

            args = torch.argwhere(mask[j]).flatten()
            eqs[j] = torch.eq(p[i][args], p_pruned[j][args]).sum()

        maxes[i] = torch.max(eqs)
        argmaxes[i] = torch.argmax(eqs)

    return maxes, argmaxes


def find_layer_perm(p, p_pruned, mask):

    print(p.shape)
    breakpoint()

    assert p.shape == p_pruned.shape

    orig_shape = p.shape

    p = p.view(orig_shape[0], -1)
    p_pruned = p_pruned.view(orig_shape[0], -1)
    mask = mask.view(orig_shape[0], -1)

    c_len, r_len = p.shape[:2]

    r_maxes, r_arg_maxes = seek_row_perm(p, p_pruned, mask)
    c_maxes, c_arg_maxes = seek_row_perm(p.t(), p_pruned.t(), mask.t())

    if torch.max(r_maxes / r_len) > torch.max(c_maxes / c_len):
        return r_arg_maxes, 'R'
    else:
        return c_arg_maxes, 'C'

torch.set_printoptions(linewidth=250)

#phase1 = resnet18_small_input()
#phase1_pruned = resnet18_small_input()

phase1 = MyFirstCNN()
phase1_pruned = MyFirstCNN()

weight1 = torch.load("phase1_mf.pt", map_location="cpu")
weight1_pruned = torch.load("phase1_pruned_mf.pt", map_location="cpu")

phase1.load_state_dict(weight1)
phase1_pruned.load_state_dict(weight1_pruned, strict=False)

layer_perms = dict()

with torch.no_grad():
    for (n, p), (n_pruned, p_pruned) in zip(
        phase1.named_parameters(), phase1_pruned.named_parameters()
    ):

        print(n)
        
        if n_pruned.replace('.weight', '.__weight_mma_mask') in weight1_pruned.keys():
            mask = weight1_pruned[n_pruned.replace('.weight', '.__weight_mma_mask')]
        else:
            mask = torch.ones_like(p)

        if "conv" in n:
            perm = find_layer_perm(p, p_pruned, mask)

        elif "bn" in n:
            # batch norm has same permutation as previous layer
            perm = perm

        elif "downsample" in n:
            perm = torch.arange(p.shape[0])

        elif "fc.weight" in n:
            perm = find_layer_perm(p, p_pruned)

        elif "fc.bias" in n:
            perm = perm

        elif 'linear' in n:
            perm = find_layer_perm(p, p_pruned, mask)
        else:
            raise ValueError

        layer_perms[n] = perm

    print(layer_perms)

# for k, v in layer_perms.items():
#    print(k, v)
