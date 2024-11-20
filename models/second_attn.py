import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath
from torch import nn


def np2th(weights, conv=False):
    if conv:
        weights = rearrange(weights, "h w i o -> o i h w")
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def find_max_n_in_lowest_scale(loader):
    max_n = 0
    try:
        for batch in loader:
            bag_features, scales, label, event_time, c = batch
            max_n = max(max_n, bag_features[-1].size(1))
        return max_n
    except Exception as e:
        # this is cause rn we only have 10 out of 259 cases in the df
        print(f"WARNING: there was an error running through all the datapoints in the loader for maxn" f"\n :::::{e}")
        return max_n


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
        elif self.c_pos_type == "learn":
            x_pos_emb = nn.Parameter(torch.zeros(1, int(_H * _W), self.dim, device=self.device))
            nn.init.trunc_normal_(x_pos_emb, std=0.02)

            cxt_pos_emb = nn.Parameter(torch.zeros(1, int(cxt_H * cxt_W), self.dim, device=self.device))
            nn.init.trunc_normal_(cxt_pos_emb, std=0.02)
        else:
            x_pos_emb = nn.Parameter(torch.zeros(1, int(_H * _W), self.dim, device=self.device), requires_grad=False)
            cxt_pos_emb = nn.Parameter(
                torch.zeros(1, int(cxt_H * cxt_W), self.dim, device=self.device), requires_grad=False
            )

        x_scale_emb = repeat(self.scale_embeddings, "() () d -> b n d", b=B, n=N)
        context_scale_emb = repeat(self.scale_embeddings, "() () d -> b n d", b=B, n=M)

        x = x + x_scale_emb + x_pos_emb[:, :N, ...]
        context = context + context_scale_emb + cxt_pos_emb[:, :M, ...]

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


class Attention(nn.Module):
    def __init__(self, configs, visualize):
        super().__init__()
        proj_drop = configs["proj_drop_rate"]
        attn_drop = configs["attn_drop_rate"]
        self.visualize = visualize

        dim = configs["dim"]
        self.heads = configs["num_heads"]
        dim_head = dim // self.heads

        self.scale = dim_head**0.5
        self.attend = nn.Softmax(dim=-1)
        self.attend_drop = nn.Dropout(attn_drop)
        self.to_Uqkv = nn.Linear(dim, 3 * dim, bias=configs["qkv_bias"])

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop),
        )

    def forward(self, x):
        qkv = self.to_Uqkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.attend_drop(attn)
        weights = attn if self.visualize else None
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out, weights


class Block(nn.Module):
    def __init__(self, configs, vis):
        super().__init__()
        self.configs = configs
        dim = configs["dim"]
        self.drop_path = DropPath(configs["drop_path"]) if configs["drop_path"] > 0.0 else nn.Identity()
        self.attention = PreNorm(dim, Attention(configs, vis))
        self.mlp = PreNorm(dim, FeedForward(configs))

    def forward(self, x, nx=None):
        x_prime, weight = self.attention(x) if self.configs["tf_type"] == "full" else self.attention(x, nx=nx)
        x = x + self.drop_path(x_prime)
        x = x + self.drop_path(self.mlp(x))
        return x, weight


class Transformer(nn.Module):
    def __init__(self, configs, visualize=False):
        super().__init__()
        depth = configs["num_layers"]
        dim = configs["dim"]

        self.vis = visualize
        self.layers = nn.ModuleList()
        for d in range(depth):
            self.layers.append(Block(configs, self.vis))

        if configs["no_class_token"]:
            self.pos_embedding = PPEGnct(dim=dim)
        else:
            self.pos_embedding = PPEG(dim=dim)

    def forward(self, x, _H=None, _W=None):
        attn_weights = []
        for i, block in enumerate(self.layers):
            x, weights = block(x, nx=_H)
            if _H and i == 0:
                x = self.pos_embedding(x, _H, _W)
            if self.vis:
                attn_weights.append(weights)
        return x, attn_weights


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        # nn.init.xavier_normal_(m.weight)
        # nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class SecondAttn(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, backbone_dim, dropout_rate=0.2, n_classes=4):
        super(SecondAttn, self).__init__()

        self.backbone_dim = backbone_dim
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.c_pos_type = ""  # '' or 'sinu' or 'learn'
        self.attn_drop_rate = 0.1
        self.emb_drop_rate = 0.15
        self.proj_drop_rate = 0.0
        self.pre_coattn_fc_dropout = 0.0
        self.coattn_attention_dropout = 0.0
        self.coattn_dropout = 0.0
        self.mlp_ratio = 2.0
        self.dim = self.embed_dim
        self.coattn_norm = True
        self.coattn_shrink = False
        self.vis = False
        self.surv_pool = "cls"
        self.n_classes = n_classes

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.dropout_emb = nn.Dropout(self.emb_drop_rate)

        if self.backbone_dim == self.embed_dim:
            self.pre_coattn_fc_1 = nn.Identity()
            self.pre_coattn_fc_2 = nn.Identity()
        else:
            self.pre_coattn_fc_1 = nn.Linear(self.backbone_dim, self.embed_dim)
            self.pre_coattn_fc_2 = nn.Linear(self.backbone_dim, self.embed_dim)

        self.pre_coattn_norm_1 = nn.LayerNorm(self.dim)
        self.pre_coattn_norm_2 = nn.LayerNorm(self.dim)

        self.coattn = CrossAttention(
            self.embed_dim, num_heads=self.num_heads, c_pos_type=self.c_pos_type, qkv_bias=True
        )

        transformer_configs = {
            "num_layers": num_layers,
            "coattn_heads": self.num_heads,
            "dim": self.embed_dim,
            "num_heads": self.num_heads,
            "proj_drop_rate": self.proj_drop_rate,
            "attn_drop_rate": self.proj_drop_rate,
            "qkv_bias": True,
            "mlp_ratio": self.mlp_ratio,
            "tf_type": "full",
            "drop_path": 0.0,
            "no_class_token": True
        }

        self.x10_transformer = Transformer(transformer_configs)
        self.x20_transformer = Transformer(transformer_configs)

        transformer_configs["no_class_token"] = False

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
        x_scale = [x10_patch_features, x20_patch_features]
        co_attn_weights = []
        x = x_scale[0]  # B, N, C

        x = self.pre_coattn_fc_1(x)

        x, _H, _W = self.square_pad(x)
        x, attn_weights = self.x10_transformer(x, _H, _W)

        for idx_, low in enumerate(x_scale[1:]):
            low = self.pre_coattn_fc_2(low)

            low, _H, _W = self.square_pad(low)
            low, attn_weights = self.x20_transformer(low, _H, _W)

            if self.coattn_norm:
                x = self.pre_coattn_norm_1(x)  #    [idx_](x)
                low = self.pre_coattn_norm_2(low)  #[idx_ + 1](low)
            if not self.coattn_shrink:
                x, weight = self.coattn(x=x, context=low)
            else:
                x, weight = self.coattn(x=low, context=x)
            if self.vis:
                co_attn_weights.append(weight)

        b, n, _ = x.shape

        x, _H, _W = self.square_pad(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout_emb(x)
        x, attn_weights = self.transformer(x, _H, _W)

        x = self.norm(x)[:, 0]

        logits = self.mlp_head(x)

        logits.unsqueeze(0)

        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        surv_y_hat = torch.topk(logits, 1, dim=1)[1]
        attention_scores = {'coattn': co_attn_weights, 'path': attn_weights}
        return hazards, surv, surv_y_hat, logits, attention_scores

    def square_pad(self, n_bag_features):
        H = n_bag_features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        n_bag_features = torch.cat([n_bag_features, n_bag_features[:, :add_length, :]], dim=1)
        return n_bag_features, _H, _W
