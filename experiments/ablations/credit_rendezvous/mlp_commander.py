"""
mlp_commander.py — 交叉注意力消融 (CA-ablation) 的 UAV 宏策略
============================================================
本模块提供 UAVSpatialCommander 的「无空间交叉注意力」对照版本 MLPCommander：
将 coverage_grid 直接展平后与 self/allies 状态拼接，仅经普通 MLP 主干输出
27 维语义动作 logits。其输入/输出与 UAVSpatialCommander 完全一致，可被
hierarchical_train_v2.py 中的 RLlib 包装器（经 run_ablation_matrix.py 的
RLlibUAVModel_MLP）直接调用。

目的（对应审稿意见 M3）：隔离「空间交叉注意力」这一贡献点本身的增益——
在动作空间、奖励、会合机制全部相同的前提下，只把宏策略主干从
Cross-Attention 换成等参数量级的 MLP，从而把贡献 1 的效果单独量化出来。
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

# 复用 my_method/hierarchical_models.py 的工具与维度常量，保证严格对齐
_MY_METHOD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "my_method"))
if _MY_METHOD not in sys.path:
    sys.path.insert(0, _MY_METHOD)

from hierarchical_models import (  # noqa: E402
    masked_logits,
    GRID_H, GRID_W,
    UAV_SELF_DIM, UAV_ALLIES_DIM, UAV_ACT_DIM,
)


class MLPCommander(nn.Module):
    """
    UAV 宏策略的「无空间注意力」消融版本。

    构造参数与 UAVSpatialCommander 保持同名（cnn_channels / token_dim /
    state_dim / nhead / dropout），未使用的参数被安全忽略，从而可以直接复用
    同一套 custom_model_config，无需修改训练配置。

    forward(obs_dict, action_mask) -> (logits[B,27], value[B,1], None)
        obs_dict = {
            "coverage_grid": [B, 20, 20],
            "self_state":    [B, 11],
            "allies_state":  [B, 9],
        }
    第三个返回值占位为 None（对齐 UAVSpatialCommander 的 attn_w），
    使 RLlib 包装器 `logits, value, _ = core(...)` 解包保持兼容。
    """

    def __init__(
        self,
        cnn_channels: int = 32,
        token_dim: int = 64,
        state_dim: int = 64,
        nhead: int = 4,
        dropout: float = 0.05,
        hidden_dim: int = 128,
    ):
        super().__init__()
        grid_dim = GRID_H * GRID_W                          # 20×20 = 400
        in_dim = grid_dim + UAV_SELF_DIM + UAV_ALLIES_DIM   # 400 + 11 + 9 = 420

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.actor_head = nn.Linear(hidden_dim, UAV_ACT_DIM)   # 27 维动作
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        cov = obs_dict["coverage_grid"].float()
        B = cov.shape[0]
        cov_flat = cov.reshape(B, -1)                           # [B, 400]
        x = torch.cat(
            [cov_flat, obs_dict["self_state"].float(), obs_dict["allies_state"].float()],
            dim=-1,
        )                                                       # [B, 420]
        h = self.trunk(x)
        logits = masked_logits(self.actor_head(h), action_mask.float())
        value = self.critic_head(h)
        return logits, value, None
