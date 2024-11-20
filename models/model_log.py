import math
import warnings
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from timm.models.layers import DropPath
from torch import einsum, nn


################################################################################ Sep 24
def np2th(weights, conv=False):
    if conv:
        weights = rearrange(weights, "h w i o -> o i h w")
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def exists(val):
    return val is not None


def find_max_n_in_lowest_scale(loader):
    max_n = 0
    try:
        for batch in loader:
            bag_features, scales, label, event_time, c = batch
            max_n = max(max_n, bag_features[-1].size(1))
        return max_n
    except Exception as e:
        print(f"WARNING: there was an error running through all the datapoints in the loader for maxn" f"\n :::::{e}")
        return max_n


def moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.0):
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
    assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]
    pos_emb = rearrange(pos_emb, "b (h w) d -> b d h w", h=h, w=w, d=embed_dim)
    return pos_emb


class PPEG(nn.Module):
    def __init__(self, dim=2048):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class PPEGnct(nn.Module):
    def __init__(self, dim=2048):
        super(PPEGnct, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, c_pos_type="", qkv_bias=False, attn_drop=0.0, proj_drop=0.0, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.c_pos_type = c_pos_type
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.scale_embeddings = nn.Parameter(torch.zeros(1, 1, dim, device=self.device))
        nn.init.trunc_normal_(self.scale_embeddings, std=0.02)

        assert self.c_pos_type in {"", "sinu", "learn"}, f"Incorrect Positional encoding: {self.c_pos_type}"

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape

        _H, _W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        cxt_H, cxt_W = int(np.ceil(np.sqrt(M))), int(np.ceil(np.sqrt(M)))

        if self.c_pos_type == "sinu":
            # only pad to nearest higher square
            pos_emb = build_2d_sincos_posemb(h=_H, w=_W, embed_dim=self.dim).to(self.device)
            pos_emb = nn.Parameter(pos_emb, requires_grad=False)
            x_pos_emb = rearrange(pos_emb, "b d nh nw -> b (nh nw) d")

            pos_emb = F.interpolate(pos_emb, size=(cxt_H, cxt_W), mode="bilinear", align_corners=False)
            cxt_pos_emb = rearrange(pos_emb, "b d nh nw -> b (nh nw) d")

            x = x + x_pos_emb[:, :N, ...]
            context = context + cxt_pos_emb[:, :M, ...]
        elif self.c_pos_type == "learn":
            x_pos_emb = nn.Parameter(torch.zeros(1, int(_H * _W), self.dim, device=self.device))
            nn.init.trunc_normal_(x_pos_emb, std=0.02)

            cxt_pos_emb = nn.Parameter(torch.zeros(1, int(cxt_H * cxt_W), self.dim, device=self.device))
            nn.init.trunc_normal_(cxt_pos_emb, std=0.02)

            x = x + x_pos_emb[:, :N, ...]
            context = context + cxt_pos_emb[:, :M, ...]
        elif self.c_pos_type == "square_pad":
            x_add_length = _H * _W - N
            x = torch.cat([x, x[:, :x_add_length, :]], dim=1)

            cxt_add_length = cxt_H * cxt_W - M
            context = torch.cat([context, context[:, :cxt_add_length, :]], dim=1)

            N = _H * _W
            M = cxt_H * cxt_W
        else:
            x_pos_emb = nn.Parameter(torch.zeros(1, int(_H * _W), self.dim, device=self.device), requires_grad=False)
            cxt_pos_emb = nn.Parameter(torch.zeros(1, int(cxt_H * cxt_W), self.dim, device=self.device), requires_grad=False)

            x = x + x_pos_emb[:, :N, ...]
            context = context + cxt_pos_emb[:, :M, ...]

        x_scale_emb = repeat(self.scale_embeddings, "() () d -> b n d", b=B, n=N)
        context_scale_emb = repeat(self.scale_embeddings, "() () d -> b n d", b=B, n=M)

        x = x + x_scale_emb
        context = context + context_scale_emb

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, configs):
        super().__init__()
        hidden_dim = int(configs["dim"] * configs["mlp_ratio"])
        self.net = nn.Sequential(
            nn.Linear(configs["dim"], hidden_dim),
            nn.GELU(),
            nn.Dropout(configs["proj_drop_rate"]),
            nn.Linear(hidden_dim, configs["dim"]),
        )

    def forward(self, x):
        return self.net(x)


class NystromAttention(nn.Module):
    def __init__(
        self, dim, dim_head=64, heads=8, num_landmarks=256, pinv_iterations=6, residual=True, residual_conv_kernel=33, eps=1e-8, dropout=0.0
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None, return_attn=True):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, "b n -> b () n")
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = "... (n l) d -> ... n d"
        q_landmarks = reduce(q, landmark_einops_eq, "sum", l=l)
        k_landmarks = reduce(k, landmark_einops_eq, "sum", l=l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, "... (n l) -> ... n", "sum", l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        # similarities

        einops_eq = "... i d, ... j d -> ... i j"
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out = out + self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out


class Nystromformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        configs,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        attn_values_residual=True,
        attn_values_residual_conv_kernel=33,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        NystromAttention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            num_landmarks=num_landmarks,
                            pinv_iterations=pinv_iterations,
                            residual=attn_values_residual,
                            residual_conv_kernel=attn_values_residual_conv_kernel,
                            dropout=attn_dropout,
                        ),
                        nn.LayerNorm(dim),
                        FeedForward(configs),
                    ]
                )
            )

    def forward(self, x, mask=None):
        attn_weights = []
        for norm_1, attn, norm_2, ff in self.layers:
            x = norm_1(x)
            x_attn, weights = attn(x, mask=mask)
            attn_weights.append(weights)
            x = x + x_attn
            x = norm_2(x)
            x = ff(x) + x
        return x, attn_weights


class Transformer(nn.Module):
    def __init__(self, configs, visualize=False):
        super().__init__()
        depth = configs["num_layers"]
        dim = configs["dim"]
        num_heads = configs["num_heads"]
        dim_head = dim // num_heads
        dropout = configs["attn_drop_rate"]

        self.pos_embed = configs["pos_embed"]

        self.vis = visualize
        self.pre_layers = Nystromformer(
            dim=dim, depth=depth, configs=configs, dim_head=dim_head, heads=num_heads, num_landmarks=dim // 2, attn_dropout=dropout
        )

        if self.pos_embed:
            self.post_layers = Nystromformer(
                dim=dim,
                depth=depth,
                configs=configs,
                dim_head=dim_head,
                heads=num_heads,
                num_landmarks=dim // 2,
                attn_dropout=dropout,
            )

            if configs["no_class_token"]:
                self.pos_embedding = PPEGnct(dim=dim)
            else:
                self.pos_embedding = PPEG(dim=dim)

    def forward(self, x, _H=None, _W=None):
        attn_weights = []
        x, weights = self.pre_layers(x)
        attn_weights.append(weights)

        if self.pos_embed:
            x = self.pos_embedding(x, _H, _W)
            x, weights = self.post_layers(x)
            attn_weights.append(weights)
        return x, attn_weights


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class FirstAttn(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, backbone_dim, dropout_rate=0.25, n_classes=4):
        super(FirstAttn, self).__init__()

        self.backbone_dim = backbone_dim
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.c_pos_type = ""  # '' or 'sinu' or 'learn'
        self.attn_drop_rate = 0.1
        self.proj_drop_rate = 0.0
        self.coattn_attention_dropout = 0.0
        self.coattn_dropout = 0.0
        self.mlp_ratio = 2.0
        self.vis = False
        self.n_classes = n_classes

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.dropout_emb = nn.Dropout(self.dropout_rate)

        if self.backbone_dim == self.embed_dim:
            self.pre_coattn_fc_1 = nn.Identity()
            self.pre_coattn_fc_2 = nn.Identity()
        else:
            self.pre_coattn_fc_1 = nn.Linear(self.backbone_dim, self.embed_dim)
            self.pre_coattn_fc_2 = nn.Linear(self.backbone_dim, self.embed_dim)

        self.pre_coattn_norm_1 = nn.LayerNorm(self.embed_dim)
        self.pre_coattn_norm_2 = nn.LayerNorm(self.embed_dim)

        self.coattn = CrossAttention(self.embed_dim, num_heads=self.num_heads, c_pos_type=self.c_pos_type, qkv_bias=True)

        self.post_coattn_norm = nn.LayerNorm(self.embed_dim)

        transformer_configs = {
            "num_layers": num_layers,
            "coattn_heads": self.num_heads,
            "dim": self.embed_dim,
            "num_heads": self.num_heads,
            "proj_drop_rate": self.proj_drop_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "qkv_bias": True,
            "mlp_ratio": self.mlp_ratio,
            "tf_type": "full",
            "drop_path": 0.0,
            "no_class_token": True,
            "pos_embed": True,
        }

        transformer_configs["no_class_token"] = False
        transformer_configs["pos_embed"] = True

        self.transformer = Transformer(transformer_configs)

        self.norm = nn.LayerNorm(self.embed_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.n_classes),
        )

        self.apply(init_weight)

    def save(self, path: str | Path) -> None:
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path: str | Path) -> None:
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def forward(self, x5_patch_features, x10_patch_features, x20_patch_features):
        co_attn_weights = []

        x10 = self.pre_coattn_fc_1(x10_patch_features)
        x10 = self.pre_coattn_norm_1(x10)

        x20 = self.pre_coattn_fc_2(x20_patch_features)
        x20 = self.pre_coattn_norm_2(x20)

        x, weight = self.coattn(x=x10, context=x20)
        x = x10 + x
        x = self.post_coattn_norm(x)

        if self.vis:
            co_attn_weights.append(weight)

        b, n, _ = x.shape

        x, _H, _W = self.square_pad(x)

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout_emb(x)
        x, attn_weights = self.transformer(x, _H, _W)

        x = self.norm(x)[:, 0]

        logits = self.mlp_head(x)

        logits.unsqueeze(0)

        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        surv_y_hat = torch.topk(logits, 1, dim=1)[1]
        attention_scores = {"coattn": co_attn_weights, "path": attn_weights}
        return hazards, surv, surv_y_hat, logits, attention_scores

    def square_pad(self, n_bag_features):
        H = n_bag_features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        n_bag_features = torch.cat([n_bag_features, n_bag_features[:, :add_length, :]], dim=1)
        return n_bag_features, _H, _W
