import torch
import torch.nn as nn


class FusionSelector(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_experts=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, summary_features):
        return self.mlp(summary_features)


class FrozenDualExpertFusion(nn.Module):
    def __init__(self, expert_a, expert_b, selector_hidden_dim=64):
        super().__init__()
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.expert_a.requires_grad_(False)
        self.expert_b.requires_grad_(False)
        self.expert_a.eval()
        self.expert_b.eval()
        self.selector = FusionSelector(input_dim=2, hidden_dim=selector_hidden_dim, num_experts=2)

    @staticmethod
    def _decoder_for(expert):
        if hasattr(expert, 'decoder'):
            return expert.decoder
        if hasattr(expert, 'f3mlp_decoder'):
            return expert.f3mlp_decoder
        raise AttributeError('Expert model has no compatible decoder.')

    def _predict_expert(self, expert, obs_img):
        with torch.no_grad():
            features = expert._encode(obs_img)
            decoder = self._decoder_for(expert)
            pred, prob, _, d_vals = decoder.predict_distribution(features)
        return pred, prob, d_vals

    def forward(self, obs_img):
        pred_a, prob_a, d_vals = self._predict_expert(self.expert_a, obs_img)
        pred_b, prob_b, _ = self._predict_expert(self.expert_b, obs_img)

        selector_in = torch.stack((pred_a.mean(dim=-1), pred_b.mean(dim=-1)), dim=-1)
        weights = self.selector(selector_in)

        probs = torch.stack((prob_a, prob_b), dim=1)
        fused_prob = (probs * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        fused_pred = torch.sum(fused_prob * d_vals.view(1, 1, -1), dim=-1)

        return {
            'pred': fused_pred,
            'prob': fused_prob,
            'weights': weights,
            'expert_a_pred': pred_a,
            'expert_b_pred': pred_b,
        }
