import math
import pdb

import numpy as np
import torch
import torch.nn.functional as F
from cuml import KMeans
from einops import rearrange, reduce
from torch import einsum, nn


class AMIL_layer(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.1, activation=None):
        super(AMIL_layer, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        if activation == "sigmoid":
            self.attention_c = nn.Sequential(nn.Linear(D, 1), nn.Sigmoid())
        elif activation == "tanh":
            self.attention_c = nn.Sequential(nn.Linear(D, 1), nn.Tanh())
        elif activation is None:
            self.attention_c = nn.Linear(D, 1)
        else:
            raise NotImplementedError

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        a = self.attention_a(x)
        b = self.attention_b(x)
        A_without_softmax = a.mul(b)
        A_without_softmax = self.attention_c(A_without_softmax).squeeze(dim=2)  # N x n_classes
        A = F.softmax(A_without_softmax, dim=1).unsqueeze(dim=1)
        h = torch.bmm(A, x).squeeze(dim=1)
        return h, A.squeeze(dim=1), A_without_softmax


class SoftFilterLayer(nn.Module):
    def __init__(self, dim, hidden_size, deep=1) -> None:
        super().__init__()
        layers = []
        for i in range(deep):
            layers.append(nn.Linear(dim, hidden_size))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.layers(x)
        h = torch.mul(x, logits)
        return h, logits


class SoftFilterLayerSNN(nn.Module):
    def __init__(self, dim, hidden_size, deep=1) -> None:
        super().__init__()
        layers = []
        for i in range(deep):
            layers.append(nn.Linear(dim, hidden_size, bias=False))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        h = self.layers(x)
        h = torch.mul(x, h)
        return h


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, head_size, dropout_prob=0.25):
        super(SelfAttention, self).__init__()
        innder_dim = num_heads * head_size
        self.num_heads = num_heads
        self.head_size = head_size

        self.to_qkv = nn.Linear(dim, innder_dim * 3)
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.to_out = nn.Sequential(nn.Linear(innder_dim, dim), nn.Dropout(dropout_prob))

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v))
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        out = torch.matmul(attention_probs, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=self.num_heads)
        out = self.to_out(out)
        return out


class ClusterLocalAttention(nn.Module):
    def __init__(self, hidden_size=384, n_cluster=None, cluster_size=None, feature_weight=0, dropout_rate=0.1) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_cluster = n_cluster
        self.cluster_size = cluster_size
        self.num_heads = 4
        self.feature_weight = feature_weight
        self.atten = SelfAttention(dim=hidden_size, num_heads=self.num_heads, head_size=hidden_size // self.num_heads)

        self.head_size = hidden_size // self.num_heads
        innder_dim = self.num_heads * self.head_size
        self.to_qkv = nn.Linear(hidden_size, innder_dim * 3)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.to_out = nn.Sequential(nn.Linear(innder_dim, hidden_size), nn.Dropout(dropout_rate))

    def forward(self, x, coords=None, cluster_label=None, return_patch_label=False):
        B, L, C = x.shape
        if cluster_label is not None:
            labels = cluster_label.numpy()[0]
        else:
            if self.n_cluster == None and self.cluster_size is not None:
                n_cluster = L // self.cluster_size
                if n_cluster == 0:
                    n_cluster = 1
            elif self.n_cluster is not None:
                n_cluster = self.n_cluster
            else:
                raise NotImplementedError
            if self.feature_weight == -1:
                np.random.seed(0)
                labels = np.random.randint(0, n_cluster, size=L)
            else:
                with torch.no_grad():
                    coords = coords.squeeze()
                    kmeans = KMeans(n_clusters=n_cluster, init="k-means++", tol=1e-4, max_iter=5, random_state=1)
                    coords = (coords - torch.mean(coords)) / torch.std(coords)
                    if self.feature_weight != 0:
                        feats = x.squeeze()
                        feats = (feats - torch.mean(feats)) / torch.std(feats)
                        feats = feats * self.feature_weight
                        coords = coords * (1 - self.feature_weight)
                        coords = torch.cat([feats, coords], dim=1)
                    labels = kmeans.fit_predict(coords)
                    labels = labels.get()
        index = np.argsort(labels, kind="stable").tolist()
        index_ori = np.argsort(index, kind="stable").tolist()
        x = x[:, index]
        window_sizes = np.bincount(labels).tolist()
        # Prevent large cluster size after clustering
        window_sizes_new = []
        for size in window_sizes:
            if size >= self.cluster_size * 2:
                num_splits = size // self.cluster_size
                quotient = size // num_splits
                remainder = size % num_splits
                result = [quotient + 1 if i < remainder else quotient for i in range(num_splits)]
                window_sizes_new.extend(result)
            else:
                window_sizes_new.append(size)
        window_sizes = window_sizes_new
        now = 0
        qs, ks, vs = self.to_qkv(x).chunk(3, dim=-1)
        qs, ks, vs = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (qs, ks, vs))
        h = torch.zeros_like(qs)
        for i in range(len(window_sizes)):
            q, k, v = qs[:, :, now : now + window_sizes[i]], ks[:, :, now : now + window_sizes[i]], vs[:, :, now : now + window_sizes[i]]
            attention_scores = torch.matmul(q, k.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.head_size)
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.attn_dropout(attention_probs)
            out = torch.matmul(attention_probs, v)
            h[:, :, now : now + window_sizes[i]] = out
            now += window_sizes[i]
        h = rearrange(h, "b h n d -> b n (h d)", h=self.num_heads)
        h = self.to_out(h) + x
        if return_patch_label:
            return h, labels
        else:
            return h, None


class SCMIL(nn.Module):
    def __init__(
        self,
        n_classes=4,
        input_size=384,
        hidden_size=None,
        deep=1,
        n_cluster=64,
        cluster_size=64,
        feature_weight=-1,
        as_backbone=False,
        dropout_rate=0.1,
        with_softfilter=False,
        use_filter_branch=False,
        with_cssa=True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.deep = deep
        self.n_cluster = n_cluster
        self.cluster_size = cluster_size
        self.feature_weight = feature_weight
        self.as_backbone = as_backbone
        self.with_softfilter = with_softfilter
        self.with_cssa = with_cssa
        if self.with_softfilter:
            use_filter_branch = True
        self.use_filter_branch = use_filter_branch

        if self.with_softfilter:
            self.softfilter = SoftFilterLayer(dim=input_size, hidden_size=hidden_size, deep=1)

        if hidden_size == None:
            hidden_size = input_size
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.GELU(), nn.Dropout(dropout_rate))
        if with_cssa:
            layers = []
            for i in range(deep):
                layers.append(
                    ClusterLocalAttention(
                        hidden_size=hidden_size,
                        n_cluster=n_cluster,
                        cluster_size=cluster_size,
                        feature_weight=feature_weight,
                        dropout_rate=dropout_rate,
                    )
                )
            self.attens = nn.Sequential(*layers)
        self.amil = AMIL_layer(hidden_size, hidden_size, dropout=dropout_rate)
        if not self.as_backbone:
            self.classifier = nn.Linear(hidden_size, n_classes)

        self.iter = 0

    def forward(self, x5_patches, x10_patches, x20_patches, coords=None, cluster_label=None, return_IS=False, return_patch_label=False):
        x = x20_patches
        self.iter += 1
        res = {}
        if self.with_softfilter:
            x, logits = self.softfilter(x)
            if return_IS:
                res["IS"] = logits.squeeze().cpu().detach().numpy()
        if self.use_filter_branch:
            idx = torch.where(logits.squeeze() > 0.5)[0]
            h = x[:, idx]
            coords = coords[:, idx]
            idx2 = torch.where(logits.squeeze() <= 0.5)[0]
            h_app = x[:, idx2]
        else:
            h = self.fc1(x)
        if self.with_cssa:
            if h.shape[1] > 1:
                for atten in self.attens:
                    h, patch_label = atten(h, coords, cluster_label, return_patch_label)
        if return_patch_label:
            patch_label_all = np.zeros(len(logits.squeeze())) - 1
            patch_label_all[idx.cpu().numpy()] = patch_label
            res["patch_label"] = patch_label_all
        if return_IS or return_patch_label:
            return res
        if self.use_filter_branch:
            h = torch.cat([h, h_app], dim=1)
        h, att, _ = self.amil(h)
        if self.as_backbone:
            return h
        logits = self.classifier(h)
        logits.unsqueeze(0)

        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        surv_y_hat = torch.topk(logits, 1, dim=1)[1]
        attention_scores = {}
        return hazards, surv, surv_y_hat, logits, attention_scores
