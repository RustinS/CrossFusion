from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn


class PPEG(nn.Module):
    def __init__(self, dim=2048, class_token=True):
        super(PPEG, self).__init__()

        self.class_token = class_token

        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        if self.class_token:
            cls_token, feat_token = x[:, 0], x[:, 1:]
            cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
            x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
            x = x.flatten(2).transpose(1, 2)
            x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        else:
            cnn_feat = x.transpose(1, 2).view(B, C, H, W)
            x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
            x = x.flatten(2).transpose(1, 2)

        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias).to(self.device)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias).to(self.device)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias).to(self.device)

        self.out_proj = nn.Linear(dim, dim).to(self.device)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.proj_dropout = nn.Dropout(proj_drop)

        self.scale_embeddings = nn.Parameter(torch.zeros(1, 1, dim, device=self.device))
        nn.init.trunc_normal_(self.scale_embeddings, std=0.02)

    def forward(self, x, context, attn_weights=False):
        B, N, C = x.shape
        _, M, _ = context.shape

        x = x + self.scale_embeddings.repeat(B, N, 1)
        context = context + self.scale_embeddings.repeat(B, M, 1)

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(context).view(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(context).view(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout.p)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)

        if attn_weights:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))
            attn_weights = F.softmax(attn, dim=-1)
            attn_weights = self.attn_dropout(attn_weights).detach()
        else:
            attn_weights = None

        x = self.out_proj(attn_output)
        x = self.proj_dropout(x)

        return x, attn_weights


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
            nn.Dropout(configs["ffn_drop_rate"]),
            nn.Linear(hidden_dim, configs["dim"]),
            nn.Dropout(configs["ffn_drop_rate"]),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, attn_dropout=0.0, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.num_heads = heads
        self.head_dim = dim // heads
        assert dim % heads == 0, "Embedding dimension must be divisible by number of heads."

        self.q_proj = nn.Linear(dim, dim, bias=True).to(self.device)
        self.k_proj = nn.Linear(dim, dim, bias=True).to(self.device)
        self.v_proj = nn.Linear(dim, dim, bias=True).to(self.device)

        self.out_proj = nn.Linear(dim, dim).to(self.device)

        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, attn_weights=False):
        B, N, C = x.size()

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout.p)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)

        if attn_weights:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))
            attn_weights = F.softmax(attn, dim=-1)
            attn_weights = self.attn_dropout(attn_weights).detach()
        else:
            attn_weights = None

        return self.out_proj(attn_output), attn_weights


class Block(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        configs,
        heads=8,
        attn_dropout=0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.LayerNorm(dim),
                        MultiHeadAttention(dim, heads=heads, attn_dropout=attn_dropout),
                        nn.LayerNorm(dim),
                        FeedForward(configs),
                    ]
                )
            )

    def forward(self, x, attn_weights=False):
        for norm_1, attn, norm_2, ff in self.layers:
            x = norm_1(x)
            x_attn, attn_w = attn(x, attn_weights=attn_weights)
            x = x + x_attn
            x = norm_2(x)
            x = ff(x) + x
        return x, attn_w


class Transformer(nn.Module):
    def __init__(self, configs, visualize=False):
        super().__init__()
        depth = configs["num_layers"]
        dim = configs["dim"]
        num_heads = configs["num_heads"]
        attn_dropout = configs["attn_drop_rate"]

        self.vis = visualize
        self.pre_layers = Block(
            dim=dim,
            depth=depth,
            configs=configs,
            heads=num_heads,
            attn_dropout=attn_dropout,
        )

        self.post_layers = Block(
            dim=dim,
            depth=depth,
            configs=configs,
            heads=num_heads,
            attn_dropout=attn_dropout,
        )

        self.ppeg = PPEG(dim=dim, class_token=configs["class_token"])

    def forward(self, x, _H=None, _W=None, attn_weights=False):
        x, pre_attn_w = self.pre_layers(x, attn_weights=attn_weights)
        x = self.ppeg(x, _H, _W)
        x, post_attn_w = self.post_layers(x, attn_weights=attn_weights)
        return x, {"pre": pre_attn_w, "post": post_attn_w}


class MultiScaleConv(nn.Module):
    def __init__(self, in_dim, out_dim, groups_dim):
        super(MultiScaleConv, self).__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 7, 1, 7 // 2, groups=groups_dim)
        self.proj1 = nn.Conv2d(in_dim, out_dim, 5, 1, 5 // 2, groups=groups_dim)
        self.proj2 = nn.Conv2d(in_dim, out_dim, 3, 1, 3 // 2, groups=groups_dim)

        self.dim_change = in_dim != out_dim
        if self.dim_change:
            self.x_proj = nn.Conv2d(in_dim, out_dim, 1, 1, 0, groups=groups_dim)

    def forward(self, x):
        x_conv = self.proj(x) + self.proj1(x) + self.proj2(x)
        if self.dim_change:
            x = self.x_proj(x)
        x = x + x_conv
        return x


class ConvProcessor(nn.Module):
    def __init__(
        self,
        dim=256,
        num_conv_layers=2,
        dropout_prob=0.2,
        activation="relu",
        use_se=True,
    ):
        super(ConvProcessor, self).__init__()
        self.dim = dim
        self.conv_channels = dim // 2
        self.num_conv_layers = num_conv_layers
        self.use_se = use_se

        if activation.lower() == "relu":
            self.activation_fn = nn.ReLU(inplace=True)
        elif activation.lower() == "leakyrelu":
            self.activation_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation.lower() == "gelu":
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.conv3d = nn.Conv3d(
            in_channels=3,
            out_channels=1,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )

        conv_layers = []
        in_channels = dim

        conv_layers.append(
            MultiScaleConv(
                in_dim=in_channels,
                out_dim=self.conv_channels,
                groups_dim=self.conv_channels,
            )
        )

        conv_layers.append(self.activation_fn)
        conv_layers.append(nn.Dropout2d(p=dropout_prob))

        self.conv_block = nn.Sequential(*conv_layers)

        self.linear = nn.Sequential(
            nn.Linear(self.conv_channels, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x1, x2, x3, H, W):
        B, _, dim = x1.size()

        x1 = x1.permute(0, 2, 1).view(B, dim, H, W)
        x2 = x2.permute(0, 2, 1).view(B, dim, H, W)
        x3 = x3.permute(0, 2, 1).view(B, dim, H, W)

        x = torch.stack([x1, x2, x3], dim=1)

        x = self.conv3d(x).squeeze(1)

        x = self.conv_block(x)

        x = x.view(B, self.conv_channels, H * W)

        x = x.permute(0, 2, 1).contiguous()

        x = self.linear(x)

        return x


def square_pad(n_bag_features):
    N = n_bag_features.shape[1]
    _H, _W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
    add_length = _H * _W - N
    n_bag_features = torch.cat([n_bag_features, n_bag_features[:, :add_length, :]], dim=1)
    return n_bag_features, _H, _W


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class CrossFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, backbone_dim, dropout_rate=0.3, n_classes=4):
        super(CrossFusion, self).__init__()

        self.attn_weights = False
        self.backbone_dim = backbone_dim
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.attn_drop_rate = 0.2
        self.proj_drop_rate = 0.2
        self.cross_attn_dropout = 0.0
        self.cross_attn_proj_dropout = 0.0
        self.mlp_ratio = 2.0
        self.vis = False
        self.n_classes = n_classes

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        nn.init.normal_(self.cls_token, std=1e-6)

        self.dropout_emb = nn.Dropout(self.dropout_rate)

        if self.backbone_dim == self.embed_dim:
            self.c_lin_proj = nn.Identity()
            self.s_lin_proj = nn.Identity()
            self.f_lin_proj = nn.Identity()
        else:
            self.c_lin_proj = nn.Sequential(nn.Linear(self.backbone_dim, self.embed_dim), nn.LayerNorm(self.embed_dim))
            self.s_lin_proj = nn.Sequential(nn.Linear(self.backbone_dim, self.embed_dim), nn.LayerNorm(self.embed_dim))
            self.f_lin_proj = nn.Sequential(nn.Linear(self.backbone_dim, self.embed_dim), nn.LayerNorm(self.embed_dim))

        self.f_cross_attn = CrossAttention(
            self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=True,
            attn_drop=self.cross_attn_dropout,
            proj_drop=self.cross_attn_proj_dropout,
        )
        self.f_cross_attn_ln = nn.LayerNorm(self.embed_dim)
        self.f_cross_attn_ffn = FeedForward({"dim": self.embed_dim, "mlp_ratio": self.mlp_ratio, "ffn_drop_rate": self.dropout_rate})

        self.c_cross_attn = CrossAttention(
            self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=True,
            attn_drop=self.cross_attn_dropout,
            proj_drop=self.cross_attn_proj_dropout,
        )
        self.c_cross_attn_ln = nn.LayerNorm(self.embed_dim)
        self.c_cross_attn_ffn = FeedForward({"dim": self.embed_dim, "mlp_ratio": self.mlp_ratio, "ffn_drop_rate": self.dropout_rate})

        transformer_configs = {
            "num_layers": num_layers,
            "coattn_heads": self.num_heads,
            "dim": self.embed_dim,
            "num_heads": self.num_heads,
            "proj_drop_rate": self.proj_drop_rate,
            "attn_drop_rate": self.attn_drop_rate,
            "qkv_bias": True,
            "mlp_ratio": self.mlp_ratio,
            "class_token": False,
            "ffn_drop_rate": self.dropout_rate,
        }

        self.c_transformer = Transformer(transformer_configs)
        self.c_transformer_ln = nn.LayerNorm(self.embed_dim)

        self.f_transformer = Transformer(transformer_configs)
        self.f_transformer_ln = nn.LayerNorm(self.embed_dim)

        self.s_transformer = Transformer(transformer_configs)
        self.s_transformer_ln = nn.LayerNorm(self.embed_dim)

        transformer_configs["class_token"] = True

        self.final_transformer = Transformer(transformer_configs)
        self.final_ln = nn.LayerNorm(self.embed_dim)

        self.conv_processor = ConvProcessor(dim=self.embed_dim, dropout_prob=self.dropout_rate)

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

    @staticmethod
    def apply_cab(x, context, cross_attn, cross_attn_ln, cross_attn_ffn, attn_weights=False):
        x_cross, cross_attn_w = cross_attn(x=x, context=context, attn_weights=attn_weights)
        x_cross = x + x_cross
        x_cross = cross_attn_ln(x_cross)
        x_cross = x_cross + cross_attn_ffn(x_cross)
        return x_cross, cross_attn_w

    @staticmethod
    def apply_pt(x, transfomer_block, norm_layer, attn_weights=False):
        x, _H, _W = square_pad(x)
        x, attn_w = transfomer_block(x, _H, _W, attn_weights=attn_weights)
        x = norm_layer(x)
        return x, _H, _W, attn_w

    def forward(self, x5, x10, x20, attn_weights=False):
        self.attn_weights = attn_weights
        attn_scores = {}
        b, _, _ = x10.shape

        x_c = self.c_lin_proj(x5)
        x_s = self.s_lin_proj(x10)
        x_f = self.f_lin_proj(x20)

        # Coarse
        x_c, coarse_cross_attn_w = self.apply_cab(
            x=x_s,
            context=x_c,
            cross_attn=self.c_cross_attn,
            cross_attn_ln=self.c_cross_attn_ln,
            cross_attn_ffn=self.c_cross_attn_ffn,
            attn_weights=attn_weights,
        )
        x_c, _H, _W, coarse_pt_attn_w = self.apply_pt(
            x=x_c, transfomer_block=self.c_transformer, norm_layer=self.c_transformer_ln, attn_weights=attn_weights
        )

        attn_scores["coarse_cross_attn_w"] = coarse_cross_attn_w
        attn_scores["coarse_pt_attn_w"] = coarse_pt_attn_w

        # Fine
        x_f, fine_cross_attn_w = self.apply_cab(
            x=x_s,
            context=x_f,
            cross_attn=self.f_cross_attn,
            cross_attn_ln=self.f_cross_attn_ln,
            cross_attn_ffn=self.f_cross_attn_ffn,
            attn_weights=attn_weights,
        )
        x_f, _H, _W, fine_pt_attn_w = self.apply_pt(
            x=x_f, transfomer_block=self.f_transformer, norm_layer=self.f_transformer_ln, attn_weights=attn_weights
        )

        attn_scores["fine_cross_attn_w"] = fine_cross_attn_w
        attn_scores["fine_pt_attn_w"] = fine_pt_attn_w

        # Source
        x_s, _H, _W, source_pt_attn_w = self.apply_pt(
            x=x_s, transfomer_block=self.s_transformer, norm_layer=self.s_transformer_ln, attn_weights=attn_weights
        )

        attn_scores["source_pt_attn_w"] = source_pt_attn_w

        # Conv Processor
        x_fused = self.conv_processor(x_s, x_f, x_c, _H, _W)

        # Final Pad-Transformer
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b).to(x_fused.dtype)
        x_fused = torch.cat((cls_tokens, x_fused), dim=1)

        x_fused, final_attn_w = self.final_transformer(x_fused, _H, _W, attn_weights=attn_weights)

        attn_scores["final_attn_w"] = final_attn_w

        # MLP Head

        logits = self.mlp_head(self.final_ln(x_fused)[:, 0])

        logits.unsqueeze(0)

        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        y_hat = torch.topk(logits, 1, dim=1)[1]
        return hazards, S, y_hat, logits, attn_scores
