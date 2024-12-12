from math import ceil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from torch import einsum, nn
from xformers.ops import memory_efficient_attention


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

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape

        x = x + self.scale_embeddings.repeat(B, N, 1)
        context = context + self.scale_embeddings.repeat(B, M, 1)

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        q = q.reshape(B * self.num_heads, N, self.head_dim)
        k = k.reshape(B * self.num_heads, M, self.head_dim)
        v = v.reshape(B * self.num_heads, M, self.head_dim)

        attn_output = memory_efficient_attention(q, k, v, p=self.attn_dropout.p)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.out_proj(attn_output)
        x = self.proj_dropout(x)

        return x


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

    def forward(self, x):
        B, N, C = x.size()

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = q.reshape(B * self.num_heads, N, self.head_dim)
        k = k.reshape(B * self.num_heads, N, self.head_dim)
        v = v.reshape(B * self.num_heads, N, self.head_dim)

        attn_output = memory_efficient_attention(q, k, v, p=self.attn_dropout.p)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        return self.out_proj(attn_output)


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

    def forward(self, x):
        for norm_1, attn, norm_2, ff in self.layers:
            x = norm_1(x)
            x_attn = attn(x)
            x = x + x_attn
            x = norm_2(x)
            x = ff(x) + x
        return x


class Transformer(nn.Module):
    def __init__(self, configs, visualize=False):
        super().__init__()
        depth = configs["num_layers"]
        dim = configs["dim"]
        num_heads = configs["num_heads"]
        attn_dropout = configs["attn_drop_rate"]

        self.pos_embed = configs["pos_embed"]

        self.vis = visualize
        self.pre_layers = Block(
            dim=dim,
            depth=depth,
            configs=configs,
            heads=num_heads,
            attn_dropout=attn_dropout,
        )

        if self.pos_embed:
            self.post_layers = Block(
                dim=dim,
                depth=depth,
                configs=configs,
                heads=num_heads,
                attn_dropout=attn_dropout,
            )

            if configs["no_class_token"]:
                self.pos_embedding = PPEGnct(dim=dim)
            else:
                self.pos_embedding = PPEG(dim=dim)

    def forward(self, x, _H=None, _W=None):
        x = self.pre_layers(x)

        if self.pos_embed:
            x = self.pos_embedding(x, _H, _W)
            x = self.post_layers(x)
        return x


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


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(B, C)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


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
        # self.conv_channels = dim
        # self.conv_channels = dim * 3
        self.conv_channels = dim // 2
        self.num_conv_layers = num_conv_layers
        self.use_se = use_se

        # Activation function
        if activation.lower() == "relu":
            self.activation_fn = nn.ReLU(inplace=True)
        elif activation.lower() == "leakyrelu":
            self.activation_fn = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation.lower() == "gelu":
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 3D convolution to process stacked inputs
        self.conv3d = nn.Conv3d(
            in_channels=3,  # Three inputs stacked along this dimension
            out_channels=1,  # Reduce to a single map
            kernel_size=(1, 1, 1),  # Collapse along the stacked dimension
            stride=(1, 1, 1),
            padding=(0, 0, 0),
        )

        # conv_layers = []
        # in_channels = dim

        # for i in range(num_conv_layers):
        #     conv_layers.append(
        #         MultiScaleConv(
        #             in_dim=in_channels,
        #             out_dim=self.conv_channels,
        #             groups_dim=dim,
        #         )
        #     )
        #     conv_layers.append(nn.GroupNorm(num_groups=32, num_channels=self.conv_channels))
        #     conv_layers.append(self.activation_fn)
        #     conv_layers.append(nn.Dropout2d(p=dropout_prob))
        #     in_channels = self.conv_channels

        conv_layers = []
        in_channels = dim

        conv_layers.append(
            MultiScaleConv(
                in_dim=in_channels,
                out_dim=self.conv_channels,
                groups_dim=self.conv_channels,
            )
        )
        # conv_layers.append(nn.GroupNorm(num_groups=self.conv_channels, num_channels=self.conv_channels))
        conv_layers.append(self.activation_fn)
        conv_layers.append(nn.Dropout2d(p=dropout_prob))

        self.conv_block = nn.Sequential(*conv_layers)

        # if self.use_se:
        #     self.se_block = SEBlock(self.conv_channels)

        # self.linear = nn.Sequential(
        #     nn.Linear(self.conv_channels, dim),
        #     self.activation_fn,
        #     nn.Dropout(p=dropout_prob),
        # )
        # if self.use_se:
        #     self.se_block = SEBlock(self.conv_channels)

        # if self.use_se:
        #     self.se_block = SEBlock(self.dim)

        self.linear = nn.Sequential(
            nn.Linear(self.conv_channels, dim),
            nn.LayerNorm(dim),
            # self.activation_fn,
            # nn.Dropout(p=dropout_prob),
        )

    def forward(self, x1, x2, x3, H, W):
        """
        Args:
            x1, x2, x3: Each is a tensor of shape [B, N, dim]
        """
        # Reshape each input: [B, N, dim] -> [B, dim, H, W]
        B, N, dim = x1.size()
        H = W = int(N**0.5)  # Assuming N = H * W
        x1 = x1.permute(0, 2, 1).view(B, dim, H, W)
        x2 = x2.permute(0, 2, 1).view(B, dim, H, W)
        x3 = x3.permute(0, 2, 1).view(B, dim, H, W)

        # Stack along a new dimension: [B, 3, dim, H, W]
        x = torch.stack([x1, x2, x3], dim=1)

        # Apply 3D convolution to reduce [B, 3, dim, H, W] -> [B, dim, H, W]
        x = self.conv3d(x).squeeze(1)  # Squeeze out the reduced dimension (size 1)

        # Pass through 2D convolutional layers
        x = self.conv_block(x)

        # Apply SE block if enabled
        # if self.use_se:
        #     x = self.se_block(x)

        x = x.view(B, self.conv_channels, H * W)  # [B, conv_channels, N]
        # x = x.view(B, self.dim, H * W)  # [B, conv_channels, N]

        x = x.permute(0, 2, 1).contiguous()  # [B, N, conv_channels]

        x = self.linear(x)  # [B, N, dim]

        return x


def square_pad(n_bag_features):
    H = n_bag_features.shape[1]
    _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
    add_length = _H * _W - H
    n_bag_features = torch.cat([n_bag_features, n_bag_features[:, :add_length, :]], dim=1)
    return n_bag_features, _H, _W


class CoCoFusionXX(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, backbone_dim, dropout_rate=0.4, n_classes=4):
        super(CoCoFusionXX, self).__init__()

        self.backbone_dim = backbone_dim
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.attn_drop_rate = 0.2
        self.proj_drop_rate = 0.2
        self.coattn_attention_dropout = 0.0
        self.coattn_proj_dropout = 0.0
        self.mlp_ratio = 2.0
        self.vis = False
        self.n_classes = n_classes

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        nn.init.normal_(self.cls_token, std=1e-6)

        self.dropout_emb = nn.Dropout(self.dropout_rate)

        if self.backbone_dim == self.embed_dim:
            self.coarse_pre_coattn = nn.Identity()
            self.source_pre_coattn = nn.Identity()
            self.fine_pre_coattn = nn.Identity()
        else:
            self.coarse_pre_coattn = nn.Sequential(nn.Linear(self.backbone_dim, self.embed_dim), nn.LayerNorm(self.embed_dim))
            self.source_pre_coattn = nn.Sequential(nn.Linear(self.backbone_dim, self.embed_dim), nn.LayerNorm(self.embed_dim))
            self.fine_pre_coattn = nn.Sequential(nn.Linear(self.backbone_dim, self.embed_dim), nn.LayerNorm(self.embed_dim))

        self.fine_coattn = CrossAttention(
            self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=True,
            attn_drop=self.coattn_attention_dropout,
            proj_drop=self.coattn_proj_dropout,
        )
        self.fine_coattn_norm = nn.LayerNorm(self.embed_dim)
        self.fine_coattn_ffn = FeedForward({"dim": self.embed_dim, "mlp_ratio": self.mlp_ratio, "ffn_drop_rate": self.dropout_rate})

        self.coarse_coattn = CrossAttention(
            self.embed_dim,
            num_heads=self.num_heads,
            qkv_bias=True,
            attn_drop=self.coattn_attention_dropout,
            proj_drop=self.coattn_proj_dropout,
        )
        self.coarse_coattn_norm = nn.LayerNorm(self.embed_dim)
        self.coarse_coattn_ffn = FeedForward({"dim": self.embed_dim, "mlp_ratio": self.mlp_ratio, "ffn_drop_rate": self.dropout_rate})

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
            "ffn_drop_rate": self.dropout_rate,
        }

        transformer_configs["no_class_token"] = True

        self.coarse_transformer = Transformer(transformer_configs)
        self.coarse_norm = nn.LayerNorm(self.embed_dim)

        self.fine_transformer = Transformer(transformer_configs)
        self.fine_norm = nn.LayerNorm(self.embed_dim)

        self.plain_transformer = Transformer(transformer_configs)
        self.plain_norm = nn.LayerNorm(self.embed_dim)

        transformer_configs["no_class_token"] = False

        self.final_transformer = Transformer(transformer_configs)
        self.final_norm = nn.LayerNorm(self.embed_dim)

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
    def apply_coattn(x, x_context, coattn_layer, coattn_norm, coattn_ffn):
        x_co = coattn_layer(x=x, context=x_context)
        x_co = x + x_co
        x_co = coattn_norm(x_co)
        x_co = x_co + coattn_ffn(x_co)
        return x_co

    @staticmethod
    def apply_pad_transformer(x, transformer_layer, norm_layer, dropout):
        x, _H, _W = square_pad(x)
        # x = dropout(x)
        x = transformer_layer(x, _H, _W)
        x = norm_layer(x)
        return x, _H, _W

    def forward(self, x5_patch_features, x10_patch_features, x20_patch_features):
        b, n, _ = x10_patch_features.shape

        coarse_features = self.coarse_pre_coattn(x5_patch_features)
        source_features = self.source_pre_coattn(x10_patch_features)
        fine_features = self.fine_pre_coattn(x20_patch_features)

        # x5 -> x10
        x_coarse_co = self.apply_coattn(
            source_features, coarse_features, self.coarse_coattn, self.coarse_coattn_norm, self.coarse_coattn_ffn
        )
        x_coarse_co, _H, _W = self.apply_pad_transformer(x_coarse_co, self.coarse_transformer, self.coarse_norm, self.dropout_emb)

        # x20 -> x10
        x_fine_co = self.apply_coattn(source_features, fine_features, self.fine_coattn, self.fine_coattn_norm, self.fine_coattn_ffn)
        x_fine_co, _H, _W = self.apply_pad_transformer(x_fine_co, self.fine_transformer, self.fine_norm, self.dropout_emb)

        # x10
        source_features, _H, _W = self.apply_pad_transformer(source_features, self.plain_transformer, self.plain_norm, self.dropout_emb)

        # Conv
        x = self.conv_processor(source_features, x_fine_co, x_coarse_co, _H, _W)

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b).to(x.dtype)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.final_transformer(x, _H, _W)

        x = self.final_norm(x)[:, 0]

        # Head

        logits = self.mlp_head(x)

        logits.unsqueeze(0)

        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        surv_y_hat = torch.topk(logits, 1, dim=1)[1]
        attention_scores = {}
        return hazards, surv, surv_y_hat, logits, attention_scores

    def square_pad(self, n_bag_features):
        H = n_bag_features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        n_bag_features = torch.cat([n_bag_features, n_bag_features[:, :add_length, :]], dim=1)
        return n_bag_features, _H, _W
