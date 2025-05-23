import os
import time
import math
import copy
import random
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from torchvision.models import VisionTransformer
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# train speed is slower after enabling this opts.
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

try:
    from .csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1, getCSM
    from .csm_triton import CrossScanTritonF, CrossMergeTritonF, CrossScanTriton1b1F
    from .csms6s import CrossScan, CrossMerge
    from .csms6s import CrossScan_Ab_1direction, CrossMerge_Ab_1direction, CrossScan_Ab_2direction, CrossMerge_Ab_2direction
    from .csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex
    from .csms6s import flops_selective_scan_fn, flops_selective_scan_ref, selective_scan_flop_jit
except:
    from csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1, getCSM
    from csm_triton import CrossScanTritonF, CrossMergeTritonF, CrossScanTriton1b1F
    from csms6s import CrossScan, CrossMerge
    from csms6s import CrossScan_Ab_1direction, CrossMerge_Ab_1direction, CrossScan_Ab_2direction, CrossMerge_Ab_2direction
    from csms6s import SelectiveScanMamba, SelectiveScanCore, SelectiveScanOflex
    from csms6s import flops_selective_scan_fn, flops_selective_scan_ref, selective_scan_flop_jit


# # =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        trig = False
        if isinstance(x, tuple) and (len(x) == 2):
            trig = True
            x, indices = x
            indices = indices[:, :1]
        x = x.permute(0, 2, 3, 1) 
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)  
        x = x.permute(0, 3, 1, 2)
        if (trig):
            x = torch.cat([x, indices], 1)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError

# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


# support: v0, v0seq
class SS2Dv0:
    def __initv0__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        # ======================
        dropout=0.0,
        # ======================
        seq=False,
        force_fp32=True,
        **kwargs,
    ):
        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0 
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
            
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)     

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forwardv0(self, x: torch.Tensor, SelectiveScan = SelectiveScanMamba, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (b, h, w, d)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.contiguous() # (b, k, d_state, l)
        Cs = Cs.contiguous() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()) # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i], 
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


# support: v01-v05; v051d,v052d,v052dc; 
# postfix: _onsigmoid,_onsoftmax,_ondwconv3,_onnone;_nozact,_noz;_oact;_no32;
# history support: v2,v3;v31d,v32d,v32dc;
class SS2Dv2:
    def __initv2__(
        self,
        # basic dims ===========
        i_layer,
        d,
        d_model=256,
        d_state=1,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=1, # < 2 means no conv 
        conv_bias=False,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v05_noz",
        channel_first=False,
        # ======================
        patch_size=16,
        top_k=4,
        knn=5,
        alpha=100,
        ambiguity=False,
        binary=False,
        k_group=8,
        division_rate=16, 
        mode="RFN",
        dimension="INCREASE",
        csms6s_mode="NORMAL",
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None} 
        super().__init__()
        d_inner = int(ssm_ratio * d_model)  
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        self.forward = self.forwardv2

        self.i_layer = i_layer
        self.d = d

        self.top_k = top_k
        self.knn = knn
        self.alpha = alpha
        self.k_group = k_group
        self.ambiguity=ambiguity,
        self.binary=binary,
        self.division_rate=division_rate
        self.mode = mode
        self.csms6s_mode=csms6s_mode
        self.dimension = dimension

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_cnorm:
            self.out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = LayerNorm(d_inner) 
            #self.out_norm = LayerNorm(d_inner_2) 

        # # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba), # will be deleted in the future
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba, CrossScan=CrossScan, CrossMerge=CrossMerge),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan, CrossMerge=CrossMerge),
            v04=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan, CrossMerge=CrossMerge),
            v05=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScan, CrossMerge=CrossMerge),
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=getCSM(1)[0], CrossMerge=getCSM(1)[1],
            ),
            v052d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=getCSM(2)[0], CrossMerge=getCSM(2)[1],
            ),
            v052dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, cascade2d=True),
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            # v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            # v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScan, CrossMerge=CrossMerge),
            # ===============================
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
            ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
            ),
            v32dc=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cascade2d=True),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        if (self.dimension[0] == "KEEP"):
            d_proj = d_inner
        else:
            d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # CrossScan =======================================
        self.CrossScan = CrossScan(csms6s_mode=self.csms6s_mode,
                                   top_k=self.top_k, 
                                   knn=self.knn, 
                                   alpha=self.alpha, 
                                   ambiguity=self.ambiguity, 
                                   binary=self.binary, 
                                   division_rate=self.division_rate,
                                   device="cuda",
                                   weights = 'new', 
                                   topk = 'yes')

        # # CrossMerge_spectral =======================================
        self.CrossMerge = CrossMerge()

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        #self.out_proj = Linear(d_inner_2, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(self.k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=self.k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=self.k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((self.k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((self.k_group, d_inner, dt_rank))) # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((self.k_group, d_inner))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((self.k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((self.k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((self.k_group, d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        indices: torch.Tensor=None,
        # ==============================
        to_dtype=True, # True: final out to dtype
        force_fp32=False, # True: input fp32
        # ==============================
        ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=SelectiveScanOflex,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        # ==============================
        cascade2d=False,
        **kwargs,
    ):
        x_proj_weight = self.x_proj_weight
        x_proj_bias = getattr(self, "x_proj_bias", None)
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        delta_softplus = True
        out_norm = getattr(self, "out_norm", None)
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, -1, -1, ssoflex)

        def sort_b_based_on_a(a, b):
            B, N, _ = a.shape  # B=batch size, N=9
            M = b.shape[1]     # M=6
            max_index = int(a[:,:,0].max().item()) + 1  # Assuming indices from 0 to max_index-1
            index = torch.full((B, max_index), N+1, dtype=torch.long, device=a.device)
            
            # Build the mapping from value to position in a[i, :, 0]
            index[torch.arange(B).unsqueeze(1), a[:,:,0].long()] = torch.arange(N, device=a.device).unsqueeze(0)
            
            # For each batch, get positions of b[i, :] in a[i, :, 0]
            b_positions_in_a = index[torch.arange(B).unsqueeze(1), b.long()]
            
            # Now sort b[i, :] according to b_positions_in_a
            sorted_positions = b_positions_in_a.argsort(dim=1)
            sorted_b = torch.gather(b, 1, sorted_positions)
            
            # Build out tensor
            out = torch.zeros(B, M, 1, dtype=a.dtype, device=a.device)
            out[:,:,0] = sorted_b
            return out

        #################### CROSS SCAN ####################
        global vec_indices
        if (self.csms6s_mode == "NORMAL"):
            # first layer first vssblock
            if (self.i_layer == 0) and (self.d == 0):

                eig_vec, vec_indices = self.CrossScan(x)  
                
                x_reshaped = x.view(B, D, -1).transpose(1, 2)

                sorted_features = torch.stack(tuple(
                                    torch.gather(x_reshaped, 1, vec_indices[..., i].unsqueeze(-1).expand(-1, -1, x_reshaped.shape[-1]))
                                    for i in range(self.top_k)
                                ), dim=1)
                
                flip_sorted_features = torch.stack(tuple(
                                    torch.gather(x_reshaped, 1, vec_indices[..., i].unsqueeze(-1).expand(-1, -1, x_reshaped.shape[-1])).flip(dims=(1,))
                                    for i in range(self.top_k)
                                ), dim=1)

                x = torch.cat((sorted_features, flip_sorted_features), dim=1).permute(0, 1, 3, 2)

            else:
                if (self.i_layer == 1) and (self.d == 0):
                    indices_flat = indices.view(B, -1)

                    if (self.top_k == 1):
                        vec_indices_decreased_0 = sort_b_based_on_a(vec_indices[..., 0:1], indices_flat)

                        n_x = int(x.shape[-1]*2)
                        row_vec_indices_0 = vec_indices_decreased_0 // n_x  
                        col_vec_indices_0 = vec_indices_decreased_0 % n_x 

                        row_b_0 = row_vec_indices_0 // 2
                        col_b_0 = col_vec_indices_0 // 2

                        b_index_0 = row_b_0 * int(n_x/2) + col_b_0

                        vec_indices = b_index_0

                    if (self.top_k == 2):
                        vec_indices_decreased_0 = sort_b_based_on_a(vec_indices[..., 0:1], indices_flat)
                        vec_indices_decreased_1 = sort_b_based_on_a(vec_indices[..., 1:2], indices_flat)

                        n_x = int(x.shape[-1]*2)
                        row_vec_indices_0 = vec_indices_decreased_0 // n_x  
                        col_vec_indices_0 = vec_indices_decreased_0 % n_x 
                        row_vec_indices_1 = vec_indices_decreased_1 // n_x  
                        col_vec_indices_1 = vec_indices_decreased_1 % n_x 

                        row_b_0 = row_vec_indices_0 // 2
                        col_b_0 = col_vec_indices_0 // 2
                        row_b_1 = row_vec_indices_1 // 2
                        col_b_1 = col_vec_indices_1 // 2

                        b_index_0 = row_b_0 * int(n_x/2) + col_b_0
                        b_index_1 = row_b_1 * int(n_x/2) + col_b_1

                        vec_indices = torch.cat((b_index_0, b_index_1), -1)

                    if (self.top_k == 3):    

                        vec_indices_decreased_0 = sort_b_based_on_a(vec_indices[..., 0:1], indices_flat)
                        vec_indices_decreased_1 = sort_b_based_on_a(vec_indices[..., 1:2], indices_flat)
                        vec_indices_decreased_2 = sort_b_based_on_a(vec_indices[..., 2:3], indices_flat)

                        n_x = int(x.shape[-1]*2)
                        row_vec_indices_0 = vec_indices_decreased_0 // n_x  
                        col_vec_indices_0 = vec_indices_decreased_0 % n_x 
                        row_vec_indices_1 = vec_indices_decreased_1 // n_x  
                        col_vec_indices_1 = vec_indices_decreased_1 % n_x 
                        row_vec_indices_2 = vec_indices_decreased_2 // n_x  
                        col_vec_indices_2 = vec_indices_decreased_2 % n_x 

                        row_b_0 = row_vec_indices_0 // 2
                        col_b_0 = col_vec_indices_0 // 2
                        row_b_1 = row_vec_indices_1 // 2
                        col_b_1 = col_vec_indices_1 // 2
                        row_b_2 = row_vec_indices_2 // 2
                        col_b_2 = col_vec_indices_2 // 2

                        b_index_0 = row_b_0 * int(n_x/2) + col_b_0
                        b_index_1 = row_b_1 * int(n_x/2) + col_b_1
                        b_index_2 = row_b_2 * int(n_x/2) + col_b_2
                        

                        vec_indices = torch.cat((b_index_0, b_index_1, b_index_2), -1)

                    if (self.top_k == 4):    

                        vec_indices_decreased_0 = sort_b_based_on_a(vec_indices[..., 0:1], indices_flat)
                        vec_indices_decreased_1 = sort_b_based_on_a(vec_indices[..., 1:2], indices_flat)
                        vec_indices_decreased_2 = sort_b_based_on_a(vec_indices[..., 2:3], indices_flat)
                        vec_indices_decreased_3 = sort_b_based_on_a(vec_indices[..., 3:4], indices_flat)

                        n_x = int(x.shape[-1]*2)
                        row_vec_indices_0 = vec_indices_decreased_0 // n_x  
                        col_vec_indices_0 = vec_indices_decreased_0 % n_x 
                        row_vec_indices_1 = vec_indices_decreased_1 // n_x  
                        col_vec_indices_1 = vec_indices_decreased_1 % n_x 
                        row_vec_indices_2 = vec_indices_decreased_2 // n_x  
                        col_vec_indices_2 = vec_indices_decreased_2 % n_x 
                        row_vec_indices_3 = vec_indices_decreased_3 // n_x  
                        col_vec_indices_3 = vec_indices_decreased_3 % n_x 

                        row_b_0 = row_vec_indices_0 // 2
                        col_b_0 = col_vec_indices_0 // 2
                        row_b_1 = row_vec_indices_1 // 2
                        col_b_1 = col_vec_indices_1 // 2
                        row_b_2 = row_vec_indices_2 // 2
                        col_b_2 = col_vec_indices_2 // 2
                        row_b_3 = row_vec_indices_3 // 2
                        col_b_3 = col_vec_indices_3 // 2

                        b_index_0 = row_b_0 * int(n_x/2) + col_b_0
                        b_index_1 = row_b_1 * int(n_x/2) + col_b_1
                        b_index_2 = row_b_2 * int(n_x/2) + col_b_2
                        b_index_3 = row_b_3 * int(n_x/2) + col_b_3  
                        

                        vec_indices = torch.cat((b_index_0, b_index_1, b_index_2, b_index_3), -1)    

                if (self.i_layer == 2) and (self.d == 0):
                    indices_flat = indices.view(B, -1)

                    if (self.top_k == 1):
                        vec_indices_decreased_0 = sort_b_based_on_a(vec_indices[..., 0:1], indices_flat)

                        n_x = int(x.shape[-1]*2)
                        row_vec_indices_0 = vec_indices_decreased_0 // n_x
                        col_vec_indices_0 = vec_indices_decreased_0 % n_x

                        row_b_0 = row_vec_indices_0 // 2
                        col_b_0 = col_vec_indices_0 // 2

                        b_index_0 = row_b_0 * int(n_x/2) + col_b_0

                        vec_indices = b_index_0

                    if (self.top_k == 2):
                        vec_indices_decreased_0 = sort_b_based_on_a(vec_indices[..., 0:1], indices_flat)
                        vec_indices_decreased_1 = sort_b_based_on_a(vec_indices[..., 1:2], indices_flat)

                        n_x = int(x.shape[-1]*2)
                        row_vec_indices_0 = vec_indices_decreased_0 // n_x
                        col_vec_indices_0 = vec_indices_decreased_0 % n_x
                        row_vec_indices_1 = vec_indices_decreased_1 // n_x 
                        col_vec_indices_1 = vec_indices_decreased_1 % n_x

                        row_b_0 = row_vec_indices_0 // 2
                        col_b_0 = col_vec_indices_0 // 2
                        row_b_1 = row_vec_indices_1 // 2
                        col_b_1 = col_vec_indices_1 // 2

                        b_index_0 = row_b_0 * int(n_x/2) + col_b_0
                        b_index_1 = row_b_1 * int(n_x/2) + col_b_1

                        vec_indices = torch.cat((b_index_0, b_index_1), -1)   

                    if (self.top_k == 3):    

                        vec_indices_decreased_0 = sort_b_based_on_a(vec_indices[..., 0:1], indices_flat)
                        vec_indices_decreased_1 = sort_b_based_on_a(vec_indices[..., 1:2], indices_flat)
                        vec_indices_decreased_2 = sort_b_based_on_a(vec_indices[..., 2:3], indices_flat)

                        n_x = int(x.shape[-1]*2)
                        row_vec_indices_0 = vec_indices_decreased_0 // n_x
                        col_vec_indices_0 = vec_indices_decreased_0 % n_x 
                        row_vec_indices_1 = vec_indices_decreased_1 // n_x  
                        col_vec_indices_1 = vec_indices_decreased_1 % n_x 
                        row_vec_indices_2 = vec_indices_decreased_2 // n_x  
                        col_vec_indices_2 = vec_indices_decreased_2 % n_x 

                        row_b_0 = row_vec_indices_0 // 2
                        col_b_0 = col_vec_indices_0 // 2
                        row_b_1 = row_vec_indices_1 // 2
                        col_b_1 = col_vec_indices_1 // 2
                        row_b_2 = row_vec_indices_2 // 2
                        col_b_2 = col_vec_indices_2 // 2

                        b_index_0 = row_b_0 * int(n_x/2) + col_b_0
                        b_index_1 = row_b_1 * int(n_x/2) + col_b_1
                        b_index_2 = row_b_2 * int(n_x/2) + col_b_2
                        

                        vec_indices = torch.cat((b_index_0, b_index_1, b_index_2), -1)

                    if (self.top_k == 4):    

                        vec_indices_decreased_0 = sort_b_based_on_a(vec_indices[..., 0:1], indices_flat)
                        vec_indices_decreased_1 = sort_b_based_on_a(vec_indices[..., 1:2], indices_flat)
                        vec_indices_decreased_2 = sort_b_based_on_a(vec_indices[..., 2:3], indices_flat)
                        vec_indices_decreased_3 = sort_b_based_on_a(vec_indices[..., 3:4], indices_flat)

                        n_x = int(x.shape[-1]*2)
                        row_vec_indices_0 = vec_indices_decreased_0 // n_x 
                        col_vec_indices_0 = vec_indices_decreased_0 % n_x
                        row_vec_indices_1 = vec_indices_decreased_1 // n_x  
                        col_vec_indices_1 = vec_indices_decreased_1 % n_x
                        row_vec_indices_2 = vec_indices_decreased_2 // n_x  
                        col_vec_indices_2 = vec_indices_decreased_2 % n_x 
                        row_vec_indices_3 = vec_indices_decreased_3 // n_x  
                        col_vec_indices_3 = vec_indices_decreased_3 % n_x 

                        row_b_0 = row_vec_indices_0 // 2
                        col_b_0 = col_vec_indices_0 // 2
                        row_b_1 = row_vec_indices_1 // 2
                        col_b_1 = col_vec_indices_1 // 2
                        row_b_2 = row_vec_indices_2 // 2
                        col_b_2 = col_vec_indices_2 // 2
                        row_b_3 = row_vec_indices_3 // 2
                        col_b_3 = col_vec_indices_3 // 2

                        b_index_0 = row_b_0 * int(n_x/2) + col_b_0
                        b_index_1 = row_b_1 * int(n_x/2) + col_b_1
                        b_index_2 = row_b_2 * int(n_x/2) + col_b_2
                        b_index_3 = row_b_3 * int(n_x/2) + col_b_3  
                        

                        vec_indices = torch.cat((b_index_0, b_index_1, b_index_2, b_index_3), -1)         

                if (self.i_layer == 3) and (self.d == 0):
                    indices_flat = indices.view(B, -1)

                    if (self.top_k == 1):
                        vec_indices_decreased_0 = sort_b_based_on_a(vec_indices[..., 0:1], indices_flat)

                        n_x = int(x.shape[-1]*2)
                        row_vec_indices_0 = vec_indices_decreased_0 // n_x
                        col_vec_indices_0 = vec_indices_decreased_0 % n_x

                        row_b_0 = row_vec_indices_0 // 2
                        col_b_0 = col_vec_indices_0 // 2

                        b_index_0 = row_b_0 * int(n_x/2) + col_b_0

                        vec_indices = b_index_0

                    if (self.top_k == 2):
                        vec_indices_decreased_0 = sort_b_based_on_a(vec_indices[..., 0:1], indices_flat)
                        vec_indices_decreased_1 = sort_b_based_on_a(vec_indices[..., 1:2], indices_flat)

                        n_x = int(x.shape[-1]*2)
                        row_vec_indices_0 = vec_indices_decreased_0 // n_x
                        col_vec_indices_0 = vec_indices_decreased_0 % n_x
                        row_vec_indices_1 = vec_indices_decreased_1 // n_x 
                        col_vec_indices_1 = vec_indices_decreased_1 % n_x

                        row_b_0 = row_vec_indices_0 // 2
                        col_b_0 = col_vec_indices_0 // 2
                        row_b_1 = row_vec_indices_1 // 2
                        col_b_1 = col_vec_indices_1 // 2

                        b_index_0 = row_b_0 * int(n_x/2) + col_b_0
                        b_index_1 = row_b_1 * int(n_x/2) + col_b_1

                        vec_indices = torch.cat((b_index_0, b_index_1), -1)    

                    if (self.top_k == 3):    

                        vec_indices_decreased_0 = sort_b_based_on_a(vec_indices[..., 0:1], indices_flat)
                        vec_indices_decreased_1 = sort_b_based_on_a(vec_indices[..., 1:2], indices_flat)
                        vec_indices_decreased_2 = sort_b_based_on_a(vec_indices[..., 2:3], indices_flat)

                        n_x = int(x.shape[-1]*2)
                        row_vec_indices_0 = vec_indices_decreased_0 // n_x
                        col_vec_indices_0 = vec_indices_decreased_0 % n_x
                        row_vec_indices_1 = vec_indices_decreased_1 // n_x 
                        col_vec_indices_1 = vec_indices_decreased_1 % n_x 
                        row_vec_indices_2 = vec_indices_decreased_2 // n_x 
                        col_vec_indices_2 = vec_indices_decreased_2 % n_x

                        row_b_0 = row_vec_indices_0 // 2
                        col_b_0 = col_vec_indices_0 // 2
                        row_b_1 = row_vec_indices_1 // 2
                        col_b_1 = col_vec_indices_1 // 2
                        row_b_2 = row_vec_indices_2 // 2
                        col_b_2 = col_vec_indices_2 // 2

                        b_index_0 = row_b_0 * int(n_x/2) + col_b_0
                        b_index_1 = row_b_1 * int(n_x/2) + col_b_1
                        b_index_2 = row_b_2 * int(n_x/2) + col_b_2
                        

                        vec_indices = torch.cat((b_index_0, b_index_1, b_index_2), -1)

                    if (self.top_k == 4):    

                        vec_indices_decreased_0 = sort_b_based_on_a(vec_indices[..., 0:1], indices_flat)
                        vec_indices_decreased_1 = sort_b_based_on_a(vec_indices[..., 1:2], indices_flat)
                        vec_indices_decreased_2 = sort_b_based_on_a(vec_indices[..., 2:3], indices_flat)
                        vec_indices_decreased_3 = sort_b_based_on_a(vec_indices[..., 3:4], indices_flat)

                        n_x = int(x.shape[-1]*2)
                        row_vec_indices_0 = vec_indices_decreased_0 // n_x
                        col_vec_indices_0 = vec_indices_decreased_0 % n_x 
                        row_vec_indices_1 = vec_indices_decreased_1 // n_x  
                        col_vec_indices_1 = vec_indices_decreased_1 % n_x 
                        row_vec_indices_2 = vec_indices_decreased_2 // n_x  
                        col_vec_indices_2 = vec_indices_decreased_2 % n_x 
                        row_vec_indices_3 = vec_indices_decreased_3 // n_x  
                        col_vec_indices_3 = vec_indices_decreased_3 % n_x 

                        row_b_0 = row_vec_indices_0 // 2
                        col_b_0 = col_vec_indices_0 // 2
                        row_b_1 = row_vec_indices_1 // 2
                        col_b_1 = col_vec_indices_1 // 2
                        row_b_2 = row_vec_indices_2 // 2
                        col_b_2 = col_vec_indices_2 // 2
                        row_b_3 = row_vec_indices_3 // 2
                        col_b_3 = col_vec_indices_3 // 2

                        b_index_0 = row_b_0 * int(n_x/2) + col_b_0
                        b_index_1 = row_b_1 * int(n_x/2) + col_b_1
                        b_index_2 = row_b_2 * int(n_x/2) + col_b_2
                        b_index_3 = row_b_3 * int(n_x/2) + col_b_3  
                        

                        vec_indices = torch.cat((b_index_0, b_index_1, b_index_2, b_index_3), -1)         

                x_reshaped = x.view(B, D, -1).transpose(1, 2)    

                sorted_features = torch.stack(tuple(
                                    torch.gather(x_reshaped, 1, vec_indices[..., i].unsqueeze(-1).expand(-1, -1, x_reshaped.shape[-1]))
                                    for i in range(self.top_k)
                                ), dim=1)
                
                flip_sorted_features = torch.stack(tuple(
                                    torch.gather(x_reshaped, 1, vec_indices[..., i].unsqueeze(-1).expand(-1, -1, x_reshaped.shape[-1])).flip(dims=(1,))
                                    for i in range(self.top_k)
                                ), dim=1)

                x = torch.cat((sorted_features, flip_sorted_features), dim=1).permute(0, 1, 3, 2)  

        #################### CROSS SCAN ####################
        xs = x

        if no_einsum: 
            x_dbl = F.conv1d(xs.contiguous().view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.contiguous().view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
        
        y: torch.Tensor = self.CrossMerge(ys, vec_indices)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                us=xs, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y, H=H, W=W,
            ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) # (B, L, C)
        y = out_norm(y)

        return (y.to(x.dtype) if to_dtype else y), vec_indices

    def forwardv2(self, x: torch.Tensor, indices: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        y, vec_indices = self.forward_core(x, indices)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out, vec_indices

# support: xv1a,xv2a,xv3a; 
# postfix: _cpos;_ocov;_ocov2;_ca,_ca1;_act;_mul;_onsigmoid,_onsoftmax,_ondwconv3,_onnone;
class SS2Dv3:
    def __initxv__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_inner = d_inner
        k_group = 4
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        self.forward = self.forwardxv

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        if out_norm_none:
            self.out_norm = nn.Identity()
        elif out_norm_cnorm:
            self.out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            self.out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            self.out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = LayerNorm(d_inner)

        # in proj =======================================
        self.omul, forward_type = checkpostfix("_mul", forward_type)
        self.oact, forward_type = checkpostfix("_act", forward_type)
        self.f_omul = nn.Identity() if self.omul else None
        self.out_act = nn.GELU() if self.oact else nn.Identity()

        mode = forward_type[:4]
        assert mode in ["xv1a", "xv2a", "xv3a"]

        self.forward = partial(self.forwardxv, mode=mode)
        self.dts_dim = dict(xv1a=self.dt_rank, xv2a=self.d_inner, xv3a=4 * self.dt_rank)[mode]
        d_inner_all = d_inner + self.dts_dim + 8 * d_state
        self.in_proj = Linear(d_model, d_inner_all, bias=bias)
        
        # conv =======================================
        self.cpos = False
        self.iconv = False
        self.oconv = False
        self.oconv2 = False
        if self.with_dconv:
            cact, forward_type = checkpostfix("_ca", forward_type)
            cact1, forward_type = checkpostfix("_ca1", forward_type)
            self.cact = nn.SiLU() if cact else nn.Identity()
            self.cact = nn.GELU() if cact1 else self.cact
                
            self.oconv2, forward_type = checkpostfix("_ocov2", forward_type)
            self.oconv, forward_type = checkpostfix("_ocov", forward_type)
            self.cpos, forward_type = checkpostfix("_cpos", forward_type)
            self.iconv = (not self.oconv) and (not self.oconv2)

            if self.iconv:
                self.conv2d = nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    groups=d_model,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                )
            if self.oconv:
                self.oconv2d = nn.Conv2d(
                    in_channels=d_inner,
                    out_channels=d_inner,
                    groups=d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                )
            if self.oconv2:
                self.conv2d = nn.Conv2d(
                    in_channels=d_inner_all,
                    out_channels=d_inner_all,
                    groups=d_inner_all,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                )

        # out proj =======================================
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))
        else:
            raise NotImplementedError


        if forward_type.startswith("xv2"):
            del self.dt_projs_weight
            self.dt_projs_weight = None

    def forwardxv(self, x: torch.Tensor, **kwargs):
        B, (H, W) = x.shape[0], (x.shape[2:4] if self.channel_first else x.shape[1:3])
        L = H * W
        dt_projs_weight = self.dt_projs_weight
        A_logs = self.A_logs
        dt_projs_bias = self.dt_projs_bias
        force_fp32 = False
        delta_softplus = True
        out_norm = self.out_norm
        to_dtype = True
        Ds = self.Ds

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        def selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus):
            return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, 1, True)

        if self.iconv:
            x = self.cact(self.conv2d(x)) # (b, d, h, w)
        elif self.cpos:
            x = x + self.conv2d(x) # (b, d, h, w)

        x = self.in_proj(x)
        
        if self.oconv2:
            x = self.conv2d(x) # (b, d, h, w)

        us, dts, Bs, Cs = x.split([self.d_inner, self.dts_dim, 4 * self.d_state, 4 * self.d_state], dim=(1 if self.channel_first else -1))

        _us = us
        if self.channel_first:
            Bs, Cs = Bs.view(B, 4, -1, H, W), Cs.view(B, 4, -1, H, W)
            us = CrossScanTriton.apply(us.contiguous()).view(B, -1, L)
            Bs = CrossScanTriton1b1.apply(Bs.contiguous()).view(B, 4, -1, L)
            Cs = CrossScanTriton1b1.apply(Cs.contiguous()).view(B, 4, -1, L)

            if self.dts_dim == self.dt_rank:
                dts = CrossScanTriton.apply(dts.contiguous()).view(B, -1, L)
                dts = F.conv1d(dts, dt_projs_weight.view(4 * self.d_inner, self.dt_rank, 1), None, groups=4)
            elif self.dts_dim == self.d_inner:
                dts = CrossScanTriton.apply(dts.contiguous()).view(B, -1, L)
            elif self.dts_dim == 4 * self.dt_rank:
                dts = dts.view(B, 4, -1, H, W)
                dts = CrossScanTriton1b1.apply(dts.contiguous()).view(B, -1, L)
                dts = F.conv1d(dts, dt_projs_weight.view(4 * self.d_inner, self.dt_rank, 1), None, groups=4)

        else:
            Bs, Cs = Bs.view(B, H, W, 4, -1), Cs.view(B, H, W, 4, -1)
            us = CrossScanTritonF.apply(us.contiguous(), self.channel_first).view(B, -1, L)
            Bs = CrossScanTriton1b1F.apply(Bs.contiguous(), self.channel_first).view(B, 4, -1, L)
            Cs = CrossScanTriton1b1F.apply(Cs.contiguous(), self.channel_first).view(B, 4, -1, L)

            if self.dts_dim == self.dt_rank:
                dts = CrossScanTritonF.apply(dts.contiguous(), self.channel_first).view(B, -1, L)
                dts = F.conv1d(dts, dt_projs_weight.view(4 * self.d_inner, self.dt_rank, 1), None, groups=4)
            elif self.dts_dim == self.d_inner:
                dts = CrossScanTritonF.apply(dts.contiguous(), self.channel_first).view(B, -1, L)
            elif self.dts_dim == 4 * self.dt_rank:
                dts = dts.view(B, H, W, 4, -1)
                dts = CrossScanTriton1b1F.apply(dts.contiguous(), self.channel_first).view(B, -1, L)
                dts = F.conv1d(dts, dt_projs_weight.view(4 * self.d_inner, self.dt_rank, 1), None, groups=4)

        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float) # (K * c)

        if force_fp32:
            us, dts, Bs, Cs = to_fp32(us, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            us, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, 4, -1, H, W)
            
        if self.channel_first:    
            y: torch.Tensor = CrossMergeTriton.apply(ys).view(B, -1, H, W)
        else:
            y: torch.Tensor = CrossMergeTritonF.apply(ys, self.channel_first).view(B, H, W, -1)
        y = out_norm(y)
        
        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                us=us, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y,
            ))

        y = (y.to(x.dtype) if to_dtype else y)
        
        y = self.out_act(y)
        
        if self.omul:
            y = y * _us

        if self.oconv:
            y = y + self.cact(self.oconv2d(_us))

        out = self.dropout(self.out_proj(y))
        return out


class SS2D(nn.Module, mamba_init, SS2Dv0, SS2Dv2, SS2Dv3):
    def __init__(
        self,
        # basic dims ===========
        i_layer,
        d, 
        d_model=256,
        d_state=1,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=1, # < 2 means no conv 
        conv_bias=False,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v05_noz",
        channel_first=False,
        patch_size=16,
        top_k=4,
        knn=5,
        alpha=100,
        ambiguity=False,
        binary=False,
        k_group=8,
        division_rate=16, 
        mode="RFN",
        dimension="INCREASE",
        csms6s_mode="NORMAL",
        # ======================
        **kwargs,
    ):
        super().__init__()
        kwargs.update(
            i_layer=i_layer, d=d, d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first, patch_size=patch_size, top_k=top_k, 
            knn=knn, alpha=alpha, ambiguity=ambiguity, binary=binary, k_group=k_group, division_rate=division_rate, 
            mode=mode, dimension=dimension, csms6s_mode=csms6s_mode,
        )
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
        else:
            self.__initv2__(**kwargs)


# =====================================================
class VSSBlock(nn.Module):
    def __init__(
        self,
        i_layer,
        d,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 1,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 1,
        ssm_conv_bias=False,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v05_noz",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        patch_size=16,
        top_k=4,
        knn=5,
        alpha=100,
        ambiguity=False,
        binary=False,
        k_group=8,
        division_rate=16, 
        mode="RFN",
        dimension="INCREASE",
        csms6s_mode="NORMAL",
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        self.i_layer = i_layer
        self.d = d 
        self.top_k = top_k

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            # if i_layer < 3:
            self.op = SS2D(
                i_layer,
                d,
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
                patch_size=patch_size,
                top_k=top_k,
                knn=knn,
                alpha=alpha,
                ambiguity=ambiguity,
                binary=binary,
                k_group=k_group,
                division_rate=division_rate, 
                mode=mode,
                dimension=dimension, 
                csms6s_mode=csms6s_mode,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first) 

    #def _forward(self, input: torch.Tensor):
    def _forward(self, input: torch.Tensor): 

        if (self.i_layer == 1) and (self.d == 0):
            x = input[:, :-1]
            indices = input[:, -1]

        elif (self.i_layer == 2) and (self.d == 0):
            x = input[:, :-1]
            indices = input[:, -1]    

        elif (self.i_layer == 3) and (self.d == 0):
            x = input[:, :-1]
            indices = input[:, -1]       

        else:    
            x = input
            indices = 0
        if self.ssm_branch:

            if self.post_norm:

                y = self.op(x)

                if (self.i_layer == 0) and (self.d == 0):
                    B, D, H, W = x.shape
                    eigen_vec, vector_indices = self.op.CrossScan(x)     
                    x_reshaped = x.view(B, D, -1).transpose(1, 2)
                    sorted_features = torch.stack(tuple(
                                        torch.gather(x_reshaped, 1, vector_indices[..., i].unsqueeze(-1).expand(-1, -1, x_reshaped.shape[-1]))
                                        for i in range(self.top_k)
                                    ), dim=1)
                    
                    flip_sorted_features = torch.stack(tuple(
                                        torch.gather(x_reshaped, 1, vector_indices[..., i].unsqueeze(-1).expand(-1, -1, x_reshaped.shape[-1])).flip(dims=(1,))
                                        for i in range(self.top_k)
                                    ), dim=1)

                    x = torch.cat((sorted_features, flip_sorted_features), dim=1).permute(0, 1, 3, 2)
                    x = self.op.CrossMerge(x.view(B, -1, D, H, W)).reshape(B, D, H, W)

                x = x + self.drop_path(self.norm(y))

            else:

                y, vector_indices = self.op(self.norm(x), indices) 

                x = x + self.drop_path(y)

        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN

        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSM(nn.Module):
    def __init__(
        self, 
        patch_size=16, 
        in_chans=3, 
        num_classes=100, 
        depths=[2, 2, 5, 2], 
        dims=[256, 512, 1024, 2048], 
        # dims=[512, 1024, 2048, 4096], 
        # =========================
        ssm_d_state=1,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=1,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v05_noz",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.2, 
        patch_norm=True, 
        norm_layer="ln2d", # "BN", "LN2D"
        downsample_version: str = "v3", # "v1", "v2", "v3"
        patchembed_version: str = "v2", # "v1", "v2"
        use_checkpoint=False,  
        # =========================
        posembed=False,
        imgsize=224,
        top_k=4,
        knn=5,
        alpha=100,
        ambiguity=False,
        binary=False,
        k_group=8,
        division_rate=16, 
        mode="RFN",
        dimension="INCREASE",
        csms6s_mode="NORMAL",
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mode = mode
        self.csms6s_mode = csms6s_mode
        self.dimension = dimension
        
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            if (self.dimension[0] == "KEEP"):
                dims = [int(dims * 1 ** i_layer) for i_layer in range(self.num_layers)]
            else:
                dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        if (self.mode == "org_VMAMBA"): #vmamba_patchifying

            _make_patch_embed = dict(
                v1=self._make_patch_embed,  
                v2=self._make_patch_embed_v2,
            ).get(patchembed_version, None)
            self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first) 

        elif (self.mode == "RFN"):

            self.model_patchification = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)   
            for param in self.model_patchification.parameters():
                param.requires_grad = False
            self.model_patchification.eval()     

        else:
            print("mode is not defined!")
            
        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                i_layer,
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                patch_size=patch_size,
                top_k=top_k,
                knn=knn,
                alpha=alpha,
                ambiguity=ambiguity,    
                binary=binary,
                k_group=k_group,
                division_rate=division_rate, 
                mode = mode,
                dimension=dimension, 
                csms6s_mode=csms6s_mode,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def patchification(self, input_image):
        x = self.model_patchification.conv1(input_image)
        x = self.model_patchification.bn1(x)
        x = self.model_patchification.relu(x)
        x = self.model_patchification.maxpool(x)

        x = self.model_patchification.layer1(x)
        x = self.model_patchification.layer2(x)
        x = self.model_patchification.layer3(x)

        return x

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):  
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )
    
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            # nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(dim, out_dim, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=1, stride=2, return_indices=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        i_layer,
        dim=96,  
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_size= 16,
        top_k= 4,
        knn=5,
        alpha=100,
        ambiguity=False,
        binary=False,
        k_group =8,
        division_rate=16, 
        mode="IAM",
        dimension="KEEP",
        csms6s_mode="NORMAL",
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                i_layer,
                d,
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                patch_size=patch_size,
                top_k=top_k,
                knn=knn,
                alpha=alpha,
                ambiguity=ambiguity,
                binary=binary,
                k_group=k_group,
                division_rate=division_rate, 
                mode=mode,
                dimension=dimension, 
                csms6s_mode=csms6s_mode,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):

        if (self.mode == "org_VMAMBA"):
            x = self.patch_embed(x)

        elif (self.mode == "RFN"):
            
            # RFN module
            
            D = 256
            B, C, H, W = x.shape

            rotations = [0, 90, 180, 270]
            inverse_rotations = [-angle for angle in rotations]

            x_rotated = [TF.rotate(x, angle) for angle in rotations]
            x_rotated = torch.stack(x_rotated).view(-1, C, H, W)

            x_cnn = self.patchification(x_rotated)
            x_cnn = x_cnn.view(len(rotations), B, D, int(H/self.patch_size), int(W/self.patch_size)) 

            x_rotated_back = [TF.rotate(x_cnn[i], angle) for i, angle in enumerate(inverse_rotations)]
            x, _ = torch.max(torch.stack(x_rotated_back), dim=0) 

        else:
            print("mode is not defined!")

        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.classifier(x)     
        return x

    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
            "prim::PythonOp.SelectiveScanOflex": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
            "prim::PythonOp.SelectiveScanCore": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
            "prim::PythonOp.SelectiveScanNRow": partial(selective_scan_flop_jit, flops_fn=flops_selective_scan_fn, verbose=verbose),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4], align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
