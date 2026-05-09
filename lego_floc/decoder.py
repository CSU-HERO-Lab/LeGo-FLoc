import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthMlpDecoder(nn.Module):
    def __init__(self, d_min=0.1, d_max=20, d_hyp=-0.2, D=128, input_dim=128):
        super().__init__()
        self.head = nn.Linear(input_dim, D)
        self.d_min = d_min
        self.d_max = d_max
        self.d_hyp = d_hyp
        self.D = D
        self.input_dim = input_dim

    def _depth_values(self, device):
        return torch.linspace(
            self.d_min ** self.d_hyp,
            self.d_max ** self.d_hyp,
            self.D,
            device=device,
        ) ** (1 / self.d_hyp)

    def predict_distribution(self, features):
        d_vals = self._depth_values(features.device)
        logits = self.head(features)
        prob = F.softmax(logits, dim=-1)
        pred_d = torch.sum(prob * d_vals.view(1, 1, -1), dim=-1)
        return pred_d, prob, logits, d_vals

    def forward(self, features):
        pred_d, _, _, _ = self.predict_distribution(features)
        return pred_d
