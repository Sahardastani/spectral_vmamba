import math
import time
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange

import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
   
class CrossScan(nn.Module):

    def __init__(self, csms6s_mode, top_k, knn, alpha, ambiguity, binary, division_rate, device, weights, topk):
        super().__init__()
        self.csms6s_mode = csms6s_mode
        self.top_k = top_k
        self.knn = knn
        self.alpha = alpha
        self.ambiguity = ambiguity
        self.device = device
        self.binary = binary
        self.division_rate = division_rate

        self.weights = weights
        self.topk = topk

    def adjacency(self, feature_vector):

        batch_size, num_nodes, feature_dim = feature_vector.shape
        distances = torch.cdist(feature_vector, feature_vector, p=2)

        if self.weights == 'old':
            distances = torch.exp(-1 * self.alpha * (distances)**2)
        elif self.weights == 'new':
            sigma = torch.mean(distances)
            distances = torch.exp(-distances ** 2 / (2 * sigma ** 2))

        if self.topk == 'yes':
            value, indices = torch.topk(distances, self.knn, dim=2, largest=True)

            adjacency_matrix = torch.zeros(batch_size, num_nodes, num_nodes, device=self.device) 
            b_idx = torch.arange(batch_size, device='cuda')[:, None, None]
            n_idx = torch.arange(num_nodes, device='cuda')[:, None]

            # Use gathered distances as weights
            adjacency_matrix[b_idx, n_idx, indices] = value
            adjacency_matrix[b_idx, indices, n_idx] = value  # Ensure symmetry    

            return adjacency_matrix   
        elif self.topk == 'no':
            return distances
    
    def compute_symmetric_laplacian(self, adjacency):

        degree = torch.sum(adjacency, dim=2)

        eps = 1e-5
        D_inv_sqrt = torch.pow(degree, -0.5)
        D_inv_sqrt = torch.diag_embed(D_inv_sqrt)

        I = torch.eye(adjacency.size(1), device=adjacency.device).unsqueeze(0)
        I = I.repeat(adjacency.size(0), 1, 1)

        laplacian = I - torch.bmm(torch.bmm(D_inv_sqrt, adjacency), D_inv_sqrt)

        return laplacian

    def topk_eigenvectors(self, eigenvalues, eigenvectors):  

        # sort eigenvalues and corresponding eigenvectors + topk eigenvectors
        sorted_eigenvalues, indices = torch.sort(eigenvalues, dim=1)
        sorted_eigenvectors = torch.gather(eigenvectors, 2, indices.unsqueeze(1).expand(-1, eigenvectors.shape[1], -1))
        smallest_eigenvectors = sorted_eigenvectors[:, :, 1:self.top_k+1]

        with torch.no_grad():
            mean_vals = torch.mean(smallest_eigenvectors, dim=2)
            signs = mean_vals.sign()
            signs = signs.unsqueeze(2)
            smallest_eigenvectors *= signs

        sorted_smallest_eigenvectors, new_indices = torch.sort(smallest_eigenvectors, dim=1) 
        return sorted_smallest_eigenvectors, new_indices

    def forward(self, features): 

        if (self.csms6s_mode == "NORMAL"):
            B, D, H, W = features.shape
            features = features.view(B, D, -1).transpose(1, 2)  

            w_matrix = self.adjacency(features)    
            L_sym = self.compute_symmetric_laplacian(w_matrix) 
            
            eigenvalues, eigenvectors = torch.linalg.eigh(L_sym) 
            sorted_smallest_eigenvectors, topk_eigenvector_indexes = self.topk_eigenvectors(eigenvalues, eigenvectors) 

            return sorted_smallest_eigenvectors, topk_eigenvector_indexes   

class CrossMerge(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, ys, vec_indices):
        B, K, D, H, W = ys.shape

        argsorted_vec_indices = torch.argsort(vec_indices, 1).permute(0, 2, 1)
        argsorted_vec_indices = argsorted_vec_indices.unsqueeze(2).expand(-1, -1, D, -1)
        ys_partial = ys[:, :int(K/2), :, :].reshape(B, int(K/2), D, -1)

        result = torch.gather(ys_partial, dim=-1, index=argsorted_vec_indices)

        result_flip = ys[:, int(K/2):, :, :].reshape(B, int(K/2), D, -1).flip(-1)
        result_flip = torch.gather(result_flip, dim=-1, index=argsorted_vec_indices)

        ys = result + result_flip 
        
        ys_ = 0
        for i in range(ys.shape[1]):
            ys_ += ys[:, i]

        return ys_

# these are for ablations =============
class CrossScan_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)  
        return x
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        return ys.sum(1).view(B, -1, H, W)

class CrossMerge_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        return ys.contiguous().sum(1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x.view(B, 4, C, H, W)

class CrossScan_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        return x
    
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        return ys.view(B, 4, -1, H, W).sum(1)

class CrossMerge_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, C, H, W = ys.shape
        ctx.shape = (B, C, H, W)
        return ys.view(B, 4, -1, H * W).sum(1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, C, H, W = ctx.shape
        return x.view(B, 1, C, H, W).repeat(1, 4, 1, 1, 1)

# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)


def check_nan_inf(tag: str, x: torch.Tensor, enable=True):
    if enable:
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(tag, torch.isinf(x).any(), torch.isnan(x).any(), flush=True)
            import pdb; pdb.set_trace()


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
  
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L  
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

# cross selective scan ===============================
# comment all checks if inside cross_selective_scan
class SelectiveScanMamba(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd 
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def selective_scan_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn, verbose=True):
    if verbose:
        print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops




