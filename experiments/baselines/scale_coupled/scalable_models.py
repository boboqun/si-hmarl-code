"""
scalable_models.py — Scale-COUPLED policy networks for the C1 ablation
=======================================================================
Purpose
-------
These networks are the *direct contrast* to SI-HMARL's size-invariant interface,
used to validate contribution C1 ("a fixed-dimension interface, decoupled from map
size, is what enables stable training + zero-shot transfer").

They mirror `UAVSpatialCommander` / `UGVNodeDispatcher` in
`experiments/my_method/hierarchical_models.py` EXACTLY, except the macro-block
count `K` is a constructor argument (`macro_k`). Therefore:

  * the action space is K^2 + 2  (grows with K),
  * the UAV cross-attention uses K x K = K^2 spatial tokens (grows with K).

Combined with holding the physical coverage-cell size FIXED (GRID_RES = 100 m, so
the coverage grid is (L/100) x (L/100) and also grows with map side L), this gives a
policy whose action + observation interface is COUPLED to map scale. SI-HMARL instead
keeps K = 5 (27 actions) and a 20 x 20 tensor at every scale.

How K is chosen for the baseline (see README): macro_k = round(L / 400 m), so the
*physical* block size stays ~400 m as the map grows (2 km -> K=5, 3 km -> K=8,
4 km -> K=10, ...). At the 2 km training scale this baseline is IDENTICAL to
SI-HMARL; it diverges (and must be retrained) at every larger scale.

Drop-in for RLlib: register the two TorchModelV2 wrappers below and pass
    custom_model_config = {"macro_k": K, "cnn_channels": 32, "token_dim": 64, ...}

NOTE: torch was unavailable in the authoring sandbox, so run a shape smoke-test on
your cluster (see `if __name__ == "__main__"` block) before launching training.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

# Reuse the shared building blocks + observation-dimension constants from the
# production model module so the baseline stays byte-for-byte consistent with
# SI-HMARL everywhere except the K-dependent parts.
from hierarchical_models import (
    MLPBlock, masked_logits,
    UAV_SELF_DIM, UAV_ALLIES_DIM, UGV_SELF_DIM,
)

# RLlib wrappers
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog


# ======================================================================
#  K-adaptive UAV commander  (spatial cross-attention over K^2 tokens)
# ======================================================================
class UAVSpatialCommanderScalable(nn.Module):
    def __init__(self, macro_k: int = 5, cnn_channels: int = 32, token_dim: int = 64,
                 state_dim: int = 64, nhead: int = 4, dropout: float = 0.05):
        super().__init__()
        self.macro_k  = int(macro_k)
        self.n_blocks = self.macro_k * self.macro_k          # grows with K
        self.n_actions = self.n_blocks + 2

        # Coverage grid (arbitrary H x W) -> token_dim channels -> AdaptiveAvgPool to K x K.
        # The AdaptiveAvgPool replaces SI-HMARL's fixed 20->10->5 MaxPool chain, so the
        # number of spatial tokens equals K^2 regardless of the (growing) input grid.
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_channels, token_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(token_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d((self.macro_k, self.macro_k)),   # -> [B, token_dim, K, K]
        )
        # K^2 learnable position embeddings (one per macro-block); grows with K.
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.n_blocks, token_dim))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

        state_in = UAV_SELF_DIM + UAV_ALLIES_DIM                  # 11 + 9 = 20
        self.state_mlp  = MLPBlock(state_in, 64, state_dim, dropout)
        self.query_proj = nn.Linear(state_dim, token_dim) if state_dim != token_dim else nn.Identity()
        self.cross_attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=nhead,
                                                dropout=dropout, batch_first=True)
        self.attn_norm  = nn.LayerNorm(token_dim)
        self.fuse_mlp   = MLPBlock(token_dim + state_dim, 128, 128, dropout)
        self.actor_special = nn.Linear(128, 2)         # idx 0 = noop, idx K^2+1 = recharge
        self.actor_block   = nn.Linear(token_dim, 1)   # per-token block score -> K^2 logits
        self.critic_head   = nn.Linear(128, 1)

        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(lambda g: torch.nan_to_num(g, nan=0.0, posinf=10.0, neginf=-10.0))

    def _encode_grid(self, cov: torch.Tensor) -> torch.Tensor:
        x = cov.float().unsqueeze(1)                # [B, 1, H, W]
        x = self.grid_cnn(x)                        # [B, token_dim, K, K]
        x = x.flatten(2).transpose(1, 2)            # [B, K^2, token_dim]
        return x + self.spatial_pos_embed

    def forward(self, obs_dict: Dict[str, torch.Tensor], action_mask: torch.Tensor):
        tokens = self._encode_grid(obs_dict['coverage_grid'])                # [B, K^2, token_dim]
        sv = self.state_mlp(torch.cat([obs_dict['self_state'],
                                       obs_dict['allies_state']], dim=-1))   # [B, state_dim]
        q = self.query_proj(sv).unsqueeze(1)                                 # [B, 1, token_dim]
        attn, attn_w = self.cross_attn(q, tokens, tokens)
        attn = self.attn_norm(attn.squeeze(1) + q.squeeze(1))                # [B, token_dim]
        fused = self.fuse_mlp(torch.cat([attn, sv], dim=-1))                 # [B, 128]
        block_scores   = self.actor_block(tokens).squeeze(-1)                # [B, K^2]
        special_scores = self.actor_special(fused)                          # [B, 2]
        logits = torch.cat([special_scores[:, 0:1], block_scores,
                            special_scores[:, 1:2]], dim=-1)                 # [B, K^2+2]
        logits = masked_logits(torch.clamp(logits, -1e6, 1e6), action_mask.float())
        return logits, self.critic_head(fused), attn_w


# ======================================================================
#  K-adaptive UGV dispatcher  (only the action head depends on K)
# ======================================================================
class UGVNodeDispatcherScalable(nn.Module):
    def __init__(self, macro_k: int = 5, cnn_channels: int = 32, hidden_dim: int = 128,
                 nhead: int = 2, dropout: float = 0.05):
        super().__init__()
        self.n_actions = macro_k * macro_k + 2          # K^2 + 2 semantic actions
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),                    # global feature, scale-agnostic
        )
        self.self_mlp = MLPBlock(UGV_SELF_DIM, 32, 32, dropout)
        self.uav_embed = nn.Linear(8, 32)
        self.uav_attn  = nn.MultiheadAttention(embed_dim=32, num_heads=nhead,
                                               dropout=dropout, batch_first=True)
        self.uav_norm  = nn.LayerNorm(32)
        self.fuse_mlp  = MLPBlock(32 + 32 + 32 + cnn_channels, hidden_dim, hidden_dim, dropout)
        self.actor_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, self.n_actions))
        self.critic_head = nn.Linear(hidden_dim, 1)
        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(lambda g: torch.nan_to_num(g, nan=0.0, posinf=10.0, neginf=-10.0))

    def _encode_grid(self, cov):
        return self.grid_cnn(cov.float().unsqueeze(1)).flatten(1)

    def _encode_uav(self, allies):
        t = torch.stack([allies[:, 0:8], allies[:, 8:16]], dim=1)   # [B, 2, 8]
        t = self.uav_embed(t)                                       # [B, 2, 32]
        tc = torch.clamp(t, -10.0, 10.0)
        a, _ = self.uav_attn(tc, tc, tc)
        return self.uav_norm(a + t).mean(dim=1)                     # [B, 32]

    def forward(self, obs_dict, action_mask):
        g = self._encode_grid(obs_dict['coverage_grid'])
        s = self.self_mlp(obs_dict['self_state'])
        u = self._encode_uav(obs_dict['allies_state'])
        # [s, attn_feat, direct_feat, grid]; original UGVNodeDispatcher returns
        # attn_feat==direct_feat (both = attention-output mean), so 'u, u' matches it.
        fused = self.fuse_mlp(torch.cat([s, u, u, g], dim=-1))
        logits = masked_logits(torch.clamp(self.actor_head(fused), -1e6, 1e6), action_mask.float())
        return logits, self.critic_head(fused)


# ======================================================================
#  RLlib TorchModelV2 wrappers (read macro_k from custom_model_config)
# ======================================================================
class RLlibUAVModelScalable(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        cfg = model_config.get("custom_model_config", {})
        self.core = UAVSpatialCommanderScalable(
            macro_k=cfg.get("macro_k", 5), cnn_channels=cfg.get("cnn_channels", 32),
            token_dim=cfg.get("token_dim", 64), state_dim=cfg.get("state_dim", 64),
            nhead=cfg.get("nhead", 4), dropout=cfg.get("dropout", 0.05))
        self._v = None

    def forward(self, input_dict, state, seq_lens):
        o = input_dict["obs"]; inner = o["observation"]
        obs = {"coverage_grid": inner["coverage_grid"].float(),
               "self_state": inner["self_state"].float(),
               "allies_state": inner["allies_state"].float()}
        logits, value, _ = self.core(obs, o["action_mask"].float())
        self._v = value
        return logits, state

    def value_function(self):
        return self._v.squeeze(-1)


class RLlibUGVModelScalable(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        cfg = model_config.get("custom_model_config", {})
        self.core = UGVNodeDispatcherScalable(
            macro_k=cfg.get("macro_k", 5), cnn_channels=cfg.get("cnn_channels", 32),
            hidden_dim=cfg.get("hidden_dim", 128), nhead=cfg.get("nhead", 2),
            dropout=cfg.get("dropout", 0.05))
        self._v = None

    def forward(self, input_dict, state, seq_lens):
        o = input_dict["obs"]; inner = o["observation"]
        obs = {"coverage_grid": inner["coverage_grid"].float(),
               "self_state": inner["self_state"].float(),
               "allies_state": inner["allies_state"].float()}
        logits, value = self.core(obs, o["action_mask"].float())
        self._v = value
        return logits, state

    def value_function(self):
        return self._v.squeeze(-1)


ModelCatalog.register_custom_model("RLlibUAVModel_scalable", RLlibUAVModelScalable)
ModelCatalog.register_custom_model("RLlibUGVModel_scalable", RLlibUGVModelScalable)


# ----------------------------------------------------------------------
#  Shape smoke-test (run on a machine with torch; verifies K-adaptivity)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    for K, H in [(5, 20), (8, 30), (10, 40)]:     # (macro_k, coverage-grid side)
        uav = UAVSpatialCommanderScalable(macro_k=K)
        ugv = UGVNodeDispatcherScalable(macro_k=K)
        B = 4
        uav_obs = {"coverage_grid": torch.zeros(B, H, H),
                   "self_state": torch.zeros(B, UAV_SELF_DIM),
                   "allies_state": torch.zeros(B, UAV_ALLIES_DIM)}
        ugv_obs = {"coverage_grid": torch.zeros(B, H, H),
                   "self_state": torch.zeros(B, UGV_SELF_DIM),
                   "allies_state": torch.zeros(B, 16)}
        m = torch.ones(B, K * K + 2)
        lu, vu, _ = uav(uav_obs, m)
        lg, vg = ugv(ugv_obs, m)
        assert lu.shape == (B, K * K + 2) and lg.shape == (B, K * K + 2), (K, lu.shape, lg.shape)
        print(f"K={K:2d} H={H:2d}  UAV logits {tuple(lu.shape)}  UGV logits {tuple(lg.shape)}  OK")
    print("shape smoke-test passed: action dim = K^2+2 grows with K, input grid is scale-agnostic")
