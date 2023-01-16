import torch
import torch.nn.functional as F
import torchvision

import copy
import numpy as np
import pandas
from tqdm import tqdm

from utils import resnet18_small_input, cifar10_dataloaders, Config
from myfirstcnn import MyFirstCNN

def inverse_permutation(perm):

    inv_perm = torch.empty(len(perm), dtype=int)

    for i in torch.arange(len(perm)):
        inv_perm[perm[i]] = i

    return inv_perm


def seek_row_perm(p, p_pruned, mask):

    n_rows = p.shape[0]

    maxes = torch.zeros(n_rows)
    argmaxes = torch.zeros(n_rows, dtype=int)

    for i in tqdm(range(n_rows)):

        eqs = torch.zeros(n_rows)

        for j in range(n_rows):

            args = torch.argwhere(mask[j]).flatten()
            eqs[j] = torch.eq(p[i][args], p_pruned[j][args]).sum()

        #print(eqs, len(p[i]))
        #breakpoint()

        maxes[i] = torch.max(eqs)
        argmaxes[i] = torch.argmax(eqs)

    if (maxes == p.shape[1]).sum() == p.shape[0]:
        print('unpruned layer matched')
        success = True
    elif (maxes.mean() > p.shape[1] // 3):
        print('pruned layer matched')
        success = True
    else:
        print('matched fail')
        success = False

    perm = inverse_permutation(argmaxes)

    return perm, success


def find_layer_perm(p, p_pruned, mask):

    assert p.shape == p_pruned.shape

    orig_shape = p.shape

    p = p.view(orig_shape[0], -1)
    p_pruned = p_pruned.view(orig_shape[0], -1)
    mask = mask.view(orig_shape[0], -1)

    c_len, r_len = p.shape[:2]

    r_arg_maxes, r_success = seek_row_perm(p, p_pruned, mask)

    if r_success:
        return r_arg_maxes, 'R'

    else:
        c_arg_maxes, c_success = seek_row_perm(p.t(), p_pruned.t(), mask.t())

        if c_success:
            return c_arg_maxes, 'C'
        else:
            raise ValueError


def find_model_perm(model_factory, weight1, weight1_pruned):

    phase1 = model_factory()
    phase1_pruned = model_factory()

    phase1.load_state_dict(weight1)
    phase1_pruned.load_state_dict(weight1_pruned, strict=False)

    model_perm = dict()
    
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
    
            elif "bias" in n:
                if perm[-1] == 'R':
                    # bias has same permutation as previous layer
                    perm = perm
                else:
                    perm = torch.arange(len(p)), 'R'
    
            else:
                raise ValueError
    
            model_perm[n] = perm

    return model_perm


def apply_layer_perm(p, perm):

    orig_shape = p.shape

    p = p.view(orig_shape[0], -1)

    if perm[-1] == 'R':
        p = p[perm[0],:]
    elif perm[-1] == 'C':
        p = p[:,perm[0]]
    else:
        print('problem!')
        assert ValueError

    p = p.view(orig_shape)

    return p


def apply_model_perm(weights, model_perm):

    permuted_weights = copy.deepcopy(weights)

    for n, p in weights.items():

        if 'mma_mask' in n:
            continue

        #print(permuted_weights['conv0.weight'][0][0])
        #breakpoint()

        print('\npermuting', n)

        perm = model_perm[n]
        permuted_weights[n] = apply_layer_perm(p, perm)
    
        print(torch.eq(weights[n], permuted_weights[n]).sum(), 'equal out of', p.numel())

    print()

    return permuted_weights

def check_outputs(model_my_perm, model_unpermuted, dataloaders):

    with torch.no_grad():
    
        for dataloader in dataloaders:
            for i, (data, target) in enumerate(tqdm(dataloader)):
            
                p1_out = model_my_perm(data)
                p1_unp_out = model_unpermuted(data)
            
                mse = F.mse_loss(p1_out, p1_unp_out)

                assert mse < 1e-12

### setup

#model_factory = resnet18_small_input
#file1 = 'phase1_2.pt'
#file1_pruned = 'phase1_pruned_2.pt'

model_factory = MyFirstCNN
f = 'phase1_mf.pt'
f_pruned = 'phase1_pruned_mf.pt'

### load weights

weights = torch.load(f, map_location="cpu")
weights_pruned = torch.load(f_pruned, map_location="cpu")

### seek permutation

model_perm = find_model_perm(model_factory, weights, weights_pruned)

#for key in weights.keys():
#    
#    perm = model_perm[key][0][0]
#
#    try:
#        print('top')
#        print(weights[key][0][0])
#        print(weights_pruned[key][perm][0])
#    except:
#        pass
#
#    breakpoint()
#    break

### apply permutation
 
weights_permuted = apply_model_perm(weights, model_perm)

breakpoint()

### check vs original model

phase1_unpermuted = model_factory()
print(phase1_unpermuted.load_state_dict(weights, strict=False))

phase1_pruned = model_factory()
print(phase1_pruned.load_state_dict(weights_pruned, strict=False))

phase1_permuted = model_factory()
print(phase1_permuted.load_state_dict(weights_permuted, strict=False))

#for p1, p2, p3 in zip(phase1_unpermuted.parameters(), phase1_pruned.parameters(), phase1_permuted.parameters()):
#
#    print(p1.shape)
#    print(p2.shape)
#    print(p3.shape)
#
#    try:
#        print('unpermuted')
#        print(p1[0][0])
#        print('pruned')
#        print(p2[0][0])
#        print('permuted')
#        print(p3[0][0])
#
#    except:
#        print('unpermuted')
#        print(p1[0])
#        print('pruned')
#        print(p2[0])
#        print('permuted')
#        print(p3[0])
#
#    breakpoint()

config = Config(batch_size=16)
dataloaders = cifar10_dataloaders(config)

check_outputs(phase1_unpermuted, phase1_permuted, dataloaders)

print('success!')
