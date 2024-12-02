from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----> Attention module
class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


# ----> Attention Gated module
class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)  # [N, 256]
        b = self.attention_b(x)  # [N, 256]
        A = a.mul(b)  # torch.mul(a, b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class AMIL(nn.Module):
    def __init__(self, backbone_dim, n_classes, gate=False):
        super(AMIL, self).__init__()
        fc = [nn.Linear(backbone_dim, 512), nn.ReLU()]  # 1024->512
        if gate:
            attention_net = Attn_Net_Gated(L=512, D=256, n_classes=1)
        else:
            attention_net = Attn_Net(L=512, D=256, n_classes=1)

        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(512, n_classes)

    def save(self, path: str | Path) -> None:
        state_dict = self.state_dict()
        torch.save(state_dict, path)

    def load(self, path: str | Path) -> None:
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def forward(self, x5_patches, x10_patches, x20_patches):
        h = x20_patches
        h = h.squeeze(0)  # [n, 1024]

        # ---->Attention
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        h = torch.mm(A, h)

        # ---->predict output
        logits = self.classifiers(h)  # [B, n_classes]

        logits.unsqueeze(0)

        hazards = torch.sigmoid(logits)
        surv = torch.cumprod(1 - hazards, dim=1)
        surv_y_hat = torch.topk(logits, 1, dim=1)[1]
        attention_scores = {}
        return hazards, surv, surv_y_hat, logits, attention_scores
