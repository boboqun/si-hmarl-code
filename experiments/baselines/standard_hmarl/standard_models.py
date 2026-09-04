"""
hierarchical_models.py
======================
分层多智能体强化学习 (Hierarchical MARL) 策略网络

包含两个独立网络：
  ① UAVSpatialCommander  —— 空间交叉注意力，动作与区块空间对齐
  ② UGVNodeDispatcher    —— 节点调度网络，关注 UAV 电量状态

约束：
  - 纯 PyTorch，无 RLlib/Gymnasium 依赖
  - 与 HierarchicalEnv.py 观测结构严格对应
  - 每个网络暴露统一的 forward(obs_dict, action_mask) 接口
  - Actor/Critic 共享主干，分头输出

环境观测结构（来自 HierarchicalEnv.py）：
  UAV: observation = {
    "coverage_grid": uint8  (20, 20)
    "self_state":    float32 (7,)
    "allies_state":  float32 (9,)
  }
  action_mask: int8 (27,)   [0=noop, 1~25=block, 26=recharge]

  UGV: observation = {
    "coverage_grid": uint8  (20, 20)
    "self_state":    float32 (5,)
    "allies_state":  float32 (12,)
  }
  action_mask: int8 (MAX_NODES=400,)

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# ──────────────────────────────────────────────
# 全局超参数（与 HierarchicalEnv 保持一致）
# ──────────────────────────────────────────────

GRID_H       = 20        # coverage_grid 高度
GRID_W       = 20        # coverage_grid 宽度
MACRO_ROWS   = 5         # 宏区块行数
MACRO_COLS   = 5         # 宏区块列数
NUM_BLOCKS   = MACRO_ROWS * MACRO_COLS   # = 25

UAV_ACT_DIM  = 27        # UAV 动作维度
UGV_ACT_DIM  = 27        # UGV 语义动作维度 (0=停靠, 1~25=宏区块, 26=接驳)

UAV_SELF_DIM    = 7   # [x, y, battery, is_busy, task_block, is_returning, is_swapping]
UAV_ALLIES_DIM  = 9   # UGV(3) + other_UAV(6: x,y,bat,is_returning,is_swapping,task_block)
UGV_SELF_DIM    = 5   # [x, y, speed, target_norm, is_moving]
UGV_ALLIES_DIM  = 12  # 2 × UAV(6: x,y,bat,is_returning,is_swapping,task_block)


# ─────────────────────────────────────────────────────────────
#  工具：带掩码的 Softmax（动作采样用）
# ─────────────────────────────────────────────────────────────

def masked_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """
    将 action_mask == 0 的位置填充 -1e9，保证这些动作的概率接近 0。

    Args:
        logits      : [B, act_dim]
        action_mask : [B, act_dim]  (1=合法, 0=非法)
    Returns:
        masked logits [B, act_dim]
    """
    fill_val = torch.finfo(logits.dtype).min / 2   # 比 -1e9 更安全
    return logits.masked_fill(action_mask == 0, fill_val)


# ─────────────────────────────────────────────────────────────
#  共用模块：轻量 MLP Block
# ─────────────────────────────────────────────────────────────

class MLPBlock(nn.Module):
    """带 LayerNorm + GELU 的线性层堆叠，用于特征提取。"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ================================================================
#  模型一：UAVSpatialCommander
#  空间交叉注意力策略网络
# ================================================================

class UAVSpatialCommander(nn.Module):
    """
    UAV 高级指挥官网络。

    核心思想：coverage_grid → CNN → 25 个空间 Token，
              self_state + allies_state → MLP → Query Vector，
              Cross-Attention(Query, 25 Tokens) → 动作选择。

    动作空间：Discrete(27) = {0:noop, 1~25:25个区块, 26:返回充电}
    其中区块动作 1~25 与 25 个空间 Token 完美对齐。

    Args:
        cnn_channels  : CNN 中间通道数
        token_dim     : 每个空间 Token 的维度（= CNN 输出通道数）
        state_dim     : 全局状态嵌入维度（Query 维度）
        nhead         : Multi-Head Attention 头数
        dropout       : Dropout 率
    """

    def __init__(
        self,
        cnn_channels: int  = 32,
        token_dim:    int  = 64,
        state_dim:    int  = 64,
        nhead:        int  = 4,
        dropout:      float = 0.05,
    ):
        super().__init__()

        # ── A. Coverage Grid → 25 个空间 Token ────────────────────
        # 20×20 → Conv(3,3,s1) → MaxPool(2) → 10×10
        #       → Conv(3,3,s1) → MaxPool(2) → 5×5
        # 输出形状: [B, token_dim, 5, 5] → reshape → [B, 25, token_dim]
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.GELU(),
            nn.MaxPool2d(2),                                # 20→10
            nn.Conv2d(cnn_channels, token_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(token_dim),
            nn.GELU(),
            nn.MaxPool2d(2),                                # 10→5
        )
        # 5×5=25 个空间 Token，维度 token_dim
        # 附加可学习的位置编码（25 个区块各一套）
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, NUM_BLOCKS, token_dim))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

        # ── B. 全局状态 → Query Vector ─────────────────────────────────
        # self_state(7) + allies_state(8) = 15 维
        state_input_dim = UAV_SELF_DIM + UAV_ALLIES_DIM   # = 15
        self.state_mlp = MLPBlock(state_input_dim, 64, state_dim, dropout)

        # ── C. Query 向量维度对齐：state_dim → token_dim ─────────
        # 若二者不等，用线性投影对齐（Cross-Attention 要求 Q/K/V 同维）
        self.query_proj = nn.Linear(state_dim, token_dim) \
            if state_dim != token_dim else nn.Identity()

        # ── D. Multi-Head Cross-Attention ──────────────────────────
        # Query: [B, 1, token_dim]  (全局状态向量)
        # Key/Value: [B, 25, token_dim]  (25 个空间 Token)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm  = nn.LayerNorm(token_dim)

        # ── E. 融合层 ───────────────────────────────────────────────
        # 将 cross-attn 的聚合输出与全局状态拼接，送入下游头
        fused_dim = token_dim + state_dim
        self.fuse_mlp = MLPBlock(fused_dim, 128, 128, dropout)

        # ── F. Actor Head：输出 [B, 27] logits ────────────────────
        # 27 = [noop] + [block_1 ~ block_25] + [recharge]
        # 注意：block logits (25维) 直接来自注意力输出（空间对齐），
        # noop 和 recharge 来自融合特征 MLP
        self.actor_special = nn.Linear(128, 2)    # noop (idx=0) + recharge (idx=26)
        self.actor_block   = nn.Linear(token_dim, 1)   # 每个 token → 该区块得分

        # ── G. Critic Head：输出 [B, 1] value ─────────────────────
        self.critic_head = nn.Linear(128, 1)

    def _encode_grid(self, coverage_grid: torch.Tensor) -> torch.Tensor:
        """
        coverage_grid: [B, 20, 20] (uint8 → float)
        Return: spatial_tokens [B, 25, token_dim]
        """
        B = coverage_grid.size(0)
        x = coverage_grid.float().unsqueeze(1)           # [B, 1, 20, 20]
        x = self.grid_cnn(x)                             # [B, token_dim, 5, 5]
        x = x.flatten(2).transpose(1, 2)                 # [B, 25, token_dim]
        x = x + self.spatial_pos_embed                   # 叠加位置编码
        return x

    def _encode_state(self, self_state: torch.Tensor,
                      allies_state: torch.Tensor) -> torch.Tensor:
        """
        Return: query_vec [B, state_dim]
        """
        x = torch.cat([self_state, allies_state], dim=-1)   # [B, 16]
        return self.state_mlp(x)                             # [B, state_dim]

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_dict: {
                "coverage_grid": [B, 20, 20]  float/uint8
                "self_state":    [B, 6]        float32
                "allies_state":  [B, 7]        float32
            }
            action_mask: [B, 27]  int8 (1=合法, 0=非法)

        Returns:
            logits  : [B, 27]   屏蔽后的 logits（可直接 softmax 得概率）
            value   : [B, 1]    状态价值估计
            attn_w  : [B, 1, 25] 注意力权重（可视化用）
        """
        B = obs_dict['coverage_grid'].shape[0]

        # A. 空间 Token
        spatial_tokens = self._encode_grid(obs_dict['coverage_grid'])  # [B, 25, token_dim]

        # B. 全局状态 → Query
        state_vec = self._encode_state(
            obs_dict['self_state'], obs_dict['allies_state'])           # [B, state_dim]
        query = self.query_proj(state_vec).unsqueeze(1)                 # [B, 1, token_dim]

        # C. Cross-Attention
        attn_out, attn_w = self.cross_attn(
            query, spatial_tokens, spatial_tokens)                      # attn_out: [B, 1, token_dim]
        attn_out = self.attn_norm(attn_out.squeeze(1) + query.squeeze(1))  # residual + norm [B, token_dim]

        # D. 融合
        fused = torch.cat([attn_out, state_vec], dim=-1)               # [B, token_dim+state_dim]
        fused = self.fuse_mlp(fused)                                    # [B, 128]

        # E. Actor: 构造 [B, 27] logits
        #    idx 0     : noop
        #    idx 1~25  : 每个空间 Token 的得分（空间对齐）
        #    idx 26    : recharge
        block_scores = self.actor_block(spatial_tokens).squeeze(-1)    # [B, 25]
        special_scores = self.actor_special(fused)                     # [B, 2]

        logits = torch.cat([
            special_scores[:, 0:1],   # noop   → idx 0
            block_scores,             # block  → idx 1~25
            special_scores[:, 1:2],   # recharge → idx 26
        ], dim=-1)                                                      # [B, 27]

        # F. Action Masking
        mask = action_mask.float()
        logits = masked_logits(logits, mask)

        # G. Critic
        value = self.critic_head(fused)                                 # [B, 1]

        return logits, value, attn_w

    def get_action_and_value(
        self,
        obs_dict: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作并返回 log_prob 和 entropy。

        Returns:
            action   : [B]   采样/贪心动作
            log_prob : [B]   该动作的 log 概率
            entropy  : [B]   策略熵
            value    : [B, 1] 价值估计
        """
        logits, value, _ = self.forward(obs_dict, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ================================================================
#  模型二：UGVNodeDispatcher
#  节点调度网络（关注 UAV 电量，预判调度需求）
# ================================================================

class UGVNodeDispatcher(nn.Module):
    """
    UGV 节点调度网络。

    核心思想：
      - 从 self_state + allies_state 提取特征
      - 用 Self-Attention 关注两架 UAV 的状态（尤其是电量 / is_busy）
      - coverage_grid 用 CNN 全局池化后融合
      - 输出 27 维语义 logits (0=停靠, 1~25=宏区块, 26=接驳)

    Args:
        cnn_channels : CNN 中间通道数
        hidden_dim   : MLP 隐藏维度
        nhead        : Attention 头数
        dropout      : Dropout 率
    """

    def __init__(
        self,
        cnn_channels: int  = 32,
        hidden_dim:   int  = 128,
        nhead:        int  = 2,
        dropout:      float = 0.05,
    ):
        super().__init__()

        # ── A. Coverage Grid → 全局特征向量 ───────────────────────
        # 20×20 → CNN → GlobalAvgPool → [B, cnn_channels]
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.GELU(),
            nn.MaxPool2d(2),                                # 20→10
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),                        # →[B, cnn_channels, 1, 1]
        )
        grid_feat_dim = cnn_channels

        # ── B. Self State MLP ──────────────────────────────────────
        self.self_mlp = MLPBlock(UGV_SELF_DIM, 32, 32, dropout)

        # ── C. UAV 状态 Self-Attention ─────────────────────────────────
        # allies_state: [B, 12] = uav_0(6) + uav_1(6)
        # 每个 UAV Token: [x, y, battery, is_returning, is_swapping, task_block]
        UAV_TOKEN_DIM = 6   # 更新：支持 task_block 意图共享
        self.uav_embed = nn.Linear(UAV_TOKEN_DIM, 32)               # 每个 UAV Token 嵌入
        self.uav_attn  = nn.MultiheadAttention(
            embed_dim=32, num_heads=nhead, dropout=dropout, batch_first=True)
        self.uav_norm  = nn.LayerNorm(32)

        # ── D. 融合 MLP ─────────────────────────────────────────────
        fused_in = 32 + 32 + 32 + grid_feat_dim   # self + uav_attn + uav_direct + grid
        self.fuse_mlp = MLPBlock(fused_in, hidden_dim, hidden_dim, dropout)

        # ── E. Actor Head：输出 [B, 27] 语义 logits ───────────────────
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, UGV_ACT_DIM),
        )

        # ── F. Critic Head ──────────────────────────────────────────
        self.critic_head = nn.Linear(hidden_dim, 1)

    def _encode_grid(self, coverage_grid: torch.Tensor) -> torch.Tensor:
        x = coverage_grid.float().unsqueeze(1)       # [B, 1, 20, 20]
        x = self.grid_cnn(x)                         # [B, C, 1, 1]
        return x.flatten(1)                          # [B, C]

    def _encode_uav_allies(self, allies_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        allies_state: [B, 10] = [uav0_x, uav0_y, uav0_bat, uav0_is_returning, uav0_is_swapping,
                                  uav1_x, uav1_y, uav1_bat, uav1_is_returning, uav1_is_swapping]
        返回:
            attn_feat  : [B, 32]  注意力融合后的特征
            direct_feat: [B, 32]  共用注意力输出（可扩展）
        """
        # 分拆两架 UAV 的 Token（每架 5 维）
        uav0 = allies_state[:, 0:6]   # [B, 6]
        uav1 = allies_state[:, 6:12]  # [B, 6]
        tokens = torch.stack([uav0, uav1], dim=1)     # [B, 2, 5]
        tokens = self.uav_embed(tokens)               # [B, 2, 32]

        attn_out, _ = self.uav_attn(tokens, tokens, tokens)  # [B, 2, 32]
        attn_out = self.uav_norm(attn_out + tokens)           # residual + norm
        attn_feat = attn_out.mean(dim=1)                      # [B, 32] 平均池化

        direct_feat = attn_feat
        return attn_feat, direct_feat

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_dict: {
                "coverage_grid": [B, 20, 20]
                "self_state":    [B, 5]
                "allies_state":  [B, 8]
            }
            action_mask: [B, 27]  (1=合法语义动作, 0=禁用)

        Returns:
            logits : [B, 27]
            value  : [B, 1]
        """
        # A. 覆盖率特征
        grid_feat = self._encode_grid(obs_dict['coverage_grid'])      # [B, 32]

        # B. UGV 自身状态
        self_feat = self.self_mlp(obs_dict['self_state'])             # [B, 32]

        # C. UAV 状态注意力
        attn_feat, direct_feat = self._encode_uav_allies(
            obs_dict['allies_state'])                                  # [B, 32] × 2

        # D. 融合
        fused = torch.cat([self_feat, attn_feat, direct_feat, grid_feat], dim=-1)
        fused = self.fuse_mlp(fused)                                  # [B, hidden_dim]

        # E. Actor
        logits = self.actor_head(fused)                               # [B, 27]
        logits = masked_logits(logits, action_mask.float())

        # F. Critic
        value = self.critic_head(fused)                               # [B, 1]

        return logits, value

    def get_action_and_value(
        self,
        obs_dict: Dict[str, torch.Tensor],
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs_dict, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ================================================================
#  Mock 测试
# ================================================================

def _make_uav_obs(B: int = 4) -> Dict[str, torch.Tensor]:
    return {
        'coverage_grid': torch.randint(0, 2, (B, GRID_H, GRID_W)).float(),
        'self_state':    torch.rand(B, UAV_SELF_DIM),
        'allies_state':  torch.rand(B, UAV_ALLIES_DIM),
    }


def _make_ugv_obs(B: int = 4) -> Dict[str, torch.Tensor]:
    return {
        'coverage_grid': torch.randint(0, 2, (B, GRID_H, GRID_W)).float(),
        'self_state':    torch.rand(B, UGV_SELF_DIM),
        'allies_state':  torch.rand(B, UGV_ALLIES_DIM),
    }


if __name__ == '__main__':
    torch.manual_seed(42)
    B = 4   # Batch size

    print("=" * 70)
    print("  Hierarchical MARL 策略网络 Mock 测试")
    print("=" * 70)

    # ── 测试一：UAVSpatialCommander ───────────────────────────────
    print("\n[1/2] UAVSpatialCommander")
    print("-" * 70)

    uav_model = UAVSpatialCommander(
        cnn_channels=32, token_dim=64, state_dim=64, nhead=4)
    uav_params = sum(p.numel() for p in uav_model.parameters())
    print(f"  可训练参数量: {uav_params:,}")

    uav_obs = _make_uav_obs(B)

    # 场景A：is_busy=True，只有 noop(0) 和 recharge(26) 合法
    mask_busy = torch.zeros(B, UAV_ACT_DIM, dtype=torch.int8)
    mask_busy[:, 0]  = 1   # noop
    mask_busy[:, 26] = 1   # recharge

    logits_busy, val_busy, attn_w = uav_model(uav_obs, mask_busy)
    prob_busy = torch.softmax(logits_busy, dim=-1)

    print("\n  [场景A] is_busy=True，掩码只开放 noop(0) 和 recharge(26)：")
    for i in range(B):
        legal_sum   = prob_busy[i, [0, 26]].sum().item()
        illegal_sum = prob_busy[i, 1:26].sum().item()
        print(f"    sample[{i}]: P(noop)={prob_busy[i,0]:.4f}  "
              f"P(recharge)={prob_busy[i,26]:.4f}  "
              f"P(blocks)={illegal_sum:.2e}  P(legal)={legal_sum:.6f}")
    assert all(
        prob_busy[i, 1:26].sum().item() < 1e-4 for i in range(B)
    ), "❌ 掩码失效：被屏蔽的区块动作仍有概率！"
    print("  [✓] 非法区块动作概率 < 1e-4 → 掩码生效")

    # 场景B：is_busy=False，全部动作合法
    mask_free = torch.ones(B, UAV_ACT_DIM, dtype=torch.int8)
    logits_free, val_free, _ = uav_model(uav_obs, mask_free)
    prob_free = torch.softmax(logits_free, dim=-1)

    print("\n  [场景B] is_busy=False，全部 27 个动作合法：")
    for i in range(B):
        topk_v, topk_i = prob_free[i].topk(3)
        print(f"    sample[{i}]: Top-3 动作: "
              + ", ".join([f"act{topk_i[k].item()}={topk_v[k].item():.3f}"
                            for k in range(3)]))

    # 注意力权重形状验证
    assert attn_w.shape == (B, 1, NUM_BLOCKS), \
        f"❌ attn_w shape 错误: {attn_w.shape}"
    print(f"\n  注意力权重 shape: {tuple(attn_w.shape)}  ✓  (B, 1, 25)")

    # 动作采样
    action, log_p, ent, _ = uav_model.get_action_and_value(uav_obs, mask_busy)
    assert action.shape == (B,), f"action shape: {action.shape}"
    print(f"  采样动作: {action.tolist()}  (应均为 0 或 26)")
    assert all(a in [0, 26] for a in action.tolist()), \
        "❌ 采样到非法动作！"
    print("  [✓] 所有采样动作均为合法动作")

    print(f"  log_prob: {log_p.tolist()}")
    print(f"  entropy:  {ent.tolist()}")
    print(f"  value:    {val_busy.squeeze(-1).tolist()}")

    # ── 测试二：UGVNodeDispatcher ─────────────────────────────────
    print("\n[2/2] UGVNodeDispatcher")
    print("-" * 70)

    ugv_model = UGVNodeDispatcher(cnn_channels=32, hidden_dim=128, nhead=2)
    ugv_params = sum(p.numel() for p in ugv_model.parameters())
    print(f"  可训练参数量: {ugv_params:,}")

    ugv_obs = _make_ugv_obs(B)

    # 场景 A：模拟 FSM-2 换电中，只有 0 号动作（停靠）合法
    mask_ugv = torch.zeros(B, UGV_ACT_DIM, dtype=torch.int8)
    mask_ugv[:, 0] = 1  # 只开放 0 号：停靠待命

    logits_ugv, val_ugv = ugv_model(ugv_obs, mask_ugv)
    prob_ugv = torch.softmax(logits_ugv, dim=-1)

    print(f"\n  [场景 A] 语义动作测试，掩码只开放 0 (停靠)：")
    for i in range(B):
        legal_prob  = prob_ugv[i, 0].item()
        illegal_sum = prob_ugv[i, 1:].sum().item()
        print(f"    sample[{i}]: P(停靠)={legal_prob:.6f}  P(其他)={illegal_sum:.2e}")
    assert all(
        prob_ugv[i, 1:].sum().item() < 1e-4 for i in range(B)
    ), "❌ 掩码失效：被屏蔽的语义动作仍有概率！"
    print("  [✓] 语义动作掩码生效！")

    action_ugv, log_p_ugv, ent_ugv, _ = ugv_model.get_action_and_value(
        ugv_obs, mask_ugv)
    assert all(a.item() == 0 for a in action_ugv), "❌ 采样到非停靠动作！"
    print(f"  采样动作: {action_ugv.tolist()}  (应均为 0)")
    print(f"  entropy:  {ent_ugv.tolist()}")
    print(f"  value:    {val_ugv.squeeze(-1).tolist()}")
    print("  [✓] 所有采样均为停靠动作")


    # ── 总结 ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  UAVSpatialCommander 参数量 : {uav_params:,}")
    print(f"  UGVNodeDispatcher   参数量 : {ugv_params:,}")
    print(f"  合计参数量          : {uav_params + ugv_params:,}")
    print("=" * 70)
    print("  [✓] 全部测试通过！")
    print("=" * 70)
