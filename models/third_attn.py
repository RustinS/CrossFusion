import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def get_sinusoidal_positional_encoding(n_positions, d_model):
    position = torch.arange(0, n_positions, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(n_positions, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  # Add batch dimension
    return pe


# class ThirdAttn(nn.Module):
#     def __init__(self, clip_model, embed_dim, num_heads, num_layers, dropout_rate=0.2, top_k=15, do_finetune=False):
#         super(ThirdAttn, self).__init__()

#         self.clip_model = clip_model
#         if do_finetune:
#             for param in self.clip_model.parameters():
#                 param.requires_grad = True
#         else:
#             for param in self.clip_model.parameters():
#                 param.requires_grad = False

#         self.feature_enhancer = nn.Sequential(
#             nn.Linear(512, embed_dim),
#             nn.LayerNorm(embed_dim),
#             nn.GELU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(embed_dim, embed_dim),
#             nn.LayerNorm(embed_dim),
#             nn.GELU(),
#         )

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim,
#             nhead=num_heads,
#             batch_first=True,
#             dropout=dropout_rate,
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         self.attention_pool1 = nn.Linear(embed_dim, 1)
#         self.attention_softmax1 = nn.Softmax(dim=1)

#         self.attention_pool2 = nn.Linear(embed_dim, 1)
#         self.attention_softmax2 = nn.Softmax(dim=1)

#         self.text_fc = nn.Sequential(
#             nn.Linear(512, embed_dim),
#             nn.LayerNorm(embed_dim),
#             nn.GELU(),
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(embed_dim * 3, embed_dim),
#             nn.LayerNorm(embed_dim),
#             nn.GELU(),
#             nn.Linear(embed_dim, 1),
#         )
#         self.top_k = top_k

#     def save(self, path: str | Path) -> None:
#         state_dict = self.state_dict()
#         torch.save(state_dict, path)

#     def load(self, path: str | Path) -> None:
#         state_dict = torch.load(path)
#         self.load_state_dict(state_dict)

#     def forward(self, patch_images, text_ids):
#         batch_size, num_patches, c, h, w = patch_images.size()

#         patch_images = patch_images.view(-1, c, h, w)
#         patch_features = self.clip_model.encode_image(patch_images)
#         patch_features = patch_features.view(batch_size, num_patches, -1)

#         patch_features = self.feature_enhancer(patch_features)

#         positional_encoding = get_sinusoidal_positional_encoding(num_patches, patch_features.size(-1)).to(
#             patch_features.device
#         )
#         patch_features += positional_encoding

#         transformer_output = self.transformer_encoder(patch_features)

#         # First attention mechanism
#         attention_scores1 = self.attention_pool1(transformer_output).squeeze(-1)
#         attention_weights1 = self.attention_softmax1(attention_scores1)
#         weighted_sum1 = (transformer_output * attention_weights1.unsqueeze(-1)).sum(dim=1)

#         # Select top-K patches
#         top_k_indices = torch.topk(attention_weights1, self.top_k, dim=1).indices
#         top_k_patches = torch.gather(
#             transformer_output, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, transformer_output.size(-1))
#         )

#         # Second attention mechanism
#         attention_scores2 = self.attention_pool2(top_k_patches).squeeze(-1)
#         attention_weights2 = self.attention_softmax2(attention_scores2)
#         weighted_sum2 = (top_k_patches * attention_weights2.unsqueeze(-1)).sum(dim=1)

#         text_features = self.clip_model.encode_text(text_ids)
#         text_features = self.text_fc(text_features)

#         # Concatenate first layer, second layer weighted attention features, and text features
#         combined_features = torch.cat((weighted_sum1, weighted_sum2, text_features), dim=1)
#         output = self.classifier(combined_features)

#         return output


class ThirdAttn(nn.Module):
    def __init__(self, clip_model, embed_dim, num_heads, num_layers, dropout_rate=0.2):
        super(ThirdAttn, self).__init__()

        self.clip_model = clip_model
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # self.x5_feature_enhancer = self._build_feature_enhancer(embed_dim, dropout_rate)
        self.x10_feature_enhancer = self._build_feature_enhancer(embed_dim, dropout_rate)
        self.x20_feature_enhancer = self._build_feature_enhancer(embed_dim, dropout_rate)
        self.text_enhancer = self._build_feature_enhancer(embed_dim, dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout_rate, activation=F.gelu
        )

        self.x20_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.x10_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.x5_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.pre_text_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.scale_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.text_to_patch_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # Initialize the attention pooling mechanism with queries
        self.attn_pool_queries = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn_pool = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # self.fc = nn.Linear(embed_dim, 1)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 1),
            # nn.LayerNorm(64),
            # nn.GELU(),
            # nn.Dropout(dropout_rate),
            # nn.Linear(64, 1),
        )

        self.reset_params()

    def _build_feature_enhancer(self, embed_dim, dropout_rate):
        return nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            # nn.GELU(),
            # nn.Dropout(dropout_rate),
        )

    def save(self, path: str | Path) -> None:
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path: str | Path) -> None:
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def reset_params(self):
        """
        Function to initialize the parameters
        """
        from torch.nn import init

        for name, module in self.named_modules():
            if name.startswith("clip_model"):
                continue  # Skip initialization for clip_model layers
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.MultiheadAttention):
                init.xavier_uniform_(module.in_proj_weight)
                init.constant_(module.in_proj_bias, 0.0)
                init.xavier_uniform_(module.out_proj.weight)
                init.constant_(module.out_proj.bias, 0.0)
            elif isinstance(module, nn.TransformerEncoderLayer):
                init.xavier_uniform_(module.linear1.weight)
                init.constant_(module.linear1.bias, 0.0)
                init.xavier_uniform_(module.linear2.weight)
                init.constant_(module.linear2.bias, 0.0)
                init.xavier_uniform_(module.self_attn.in_proj_weight)
                init.constant_(module.self_attn.in_proj_bias, 0.0)
                init.xavier_uniform_(module.self_attn.out_proj.weight)
                init.constant_(module.self_attn.out_proj.bias, 0.0)

    def forward(self, x5_patch_features, x10_patch_features, x20_patch_features, text_ids):
        batch_size, x20_num_patches, _ = x20_patch_features.size()
        _, x10_num_patches, _ = x10_patch_features.size()
        _, x5_num_patches, _ = x5_patch_features.size()

        # x5 = self.x5_feature_enhancer(x5_patch_features)
        x10 = self.x10_feature_enhancer(x10_patch_features)
        x20 = self.x20_feature_enhancer(x20_patch_features)
        text_features = self.clip_model.encode_text(text_ids)
        text_features = self.text_enhancer(text_features)

        # Positional Encoding for patches
        # x5 += get_sinusoidal_positional_encoding(x5_num_patches, x5.size(-1)).to(x5.device)
        x10 += get_sinusoidal_positional_encoding(x10_num_patches, x10.size(-1)).to(x10.device)
        x20 += get_sinusoidal_positional_encoding(x20_num_patches, x20.size(-1)).to(x20.device)

        # text_features = self.text_embedding(text_features)
        # x5_patch_features = self.x5_transformer_encoder(x5)
        x10_patch_features = self.x10_transformer_encoder(x10)
        x20_patch_features = self.x20_transformer_encoder(x20)

        combined_x, scale_attn_weights = self.scale_attn(x10, x20, x20)

        # combined_x = self.pre_text_transformer_encoder(combined_x)

        # combined_features = torch.cat((patch_features, text_features.unsqueeze(1)), dim=1)

        # text_queries = text_features.unsqueeze(1)
        # combined_x, combined_features_weights = self.text_to_patch_attn(combined_x, text_queries, text_queries)

        # Transformer encoder
        transformer_output = self.final_transformer_encoder(combined_x)

        # Attention pooling
        queries = self.attn_pool_queries.expand(batch_size, -1, -1)  # Prepare queries for batch
        attn_output, attn_weights = self.attn_pool(queries, transformer_output, transformer_output)

        # Final output
        output = self.fc(attn_output.squeeze(1))
        return output
