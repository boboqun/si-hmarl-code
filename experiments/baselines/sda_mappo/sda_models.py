"""
SDA learning baseline (Chen et al., IEEE RA-L 2026) — adapted encoder
=====================================================================
"A Multi-UAV Cooperative Coverage Method Based on Sparse Dual-Attention RL"
(原文 SDA-MATD3：MATD3 + Entity Attention Module(EAM) + 动态稀疏激活(sparsemax/softmax 混合)
 + Interaction Attention Module(IAM)).

⚠️ Chen 的任务（连续加速度、覆盖动目标+避障、无能量/无 UGV）与本文差别很大，不能直接跑其代码。
本实现采用 baseline_addition_plan.md 推荐的**轻量版 "SDA-MAPPO"**：与 mappo_flat 基线
**完全相同的 env / obs / 动作 / 训练超参**，仅把 FlatModelBase 的「CNN特征 ⊕ self ⊕ allies → MLP」
融合，替换为 Chen 的**稀疏实体注意力编码器(EAM)**，从而隔离"注意力编码器"这一变量，得到一个
独立的、近期的、学习类注意力 MARL 对照（论文表内标 `SDA-MATD3*`，adapted）。

实体集映射（扁平 obs → Chen 的实体集）：
  query   = self_state                  （本机；Chen 用 own(pos,vel,action) 作 query）
  K/V 集  = { grid_token(覆盖栅格 CNN 特征 → "目标/前沿"实体),
              allies_token(队友 UAV + UGV) }
  稀疏激活: w = λ·sparsemax(score) + (1-λ)·softmax(score)，λ 随训练线性升（见文末 ramp 注释）。

接口与 flat_models 完全一致：注册 RLlibSDAUAVModel / RLlibSDAUGVModel 即可在 train_flat 管线中替换。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


# ── Martins & Astudillo (2016) sparsemax ──────────────────────────────
def sparsemax(z, dim=-1):
    z = z - z.amax(dim=dim, keepdim=True)              # 数值稳定
    zs, _ = torch.sort(z, dim=dim, descending=True)
    n = z.shape[dim]
    rng = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    shp = [1] * z.dim(); shp[dim] = n
    rng = rng.view(shp)
    cssv = zs.cumsum(dim) - 1
    cond = (1 + rng * zs) > cssv
    k = cond.to(z.dtype).sum(dim=dim, keepdim=True).clamp(min=1.0)
    tau = cssv.gather(dim, (k.long() - 1)) / k
    return torch.clamp(z - tau, min=0.0)


class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
    def forward(self, x): return self.net(x)


class EntityAttention(nn.Module):
    """EAM：self 作 query，对实体集 (K/V) 做多头注意力 + 动态稀疏激活。"""
    def __init__(self, d, heads=4):
        super().__init__()
        assert d % heads == 0
        self.d, self.h, self.dh = d, heads, d // heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        # 稀疏混合系数 λ：训练中线性升（见文末）；默认 0.5 给稳态混合
        self.register_buffer("sparsity_lambda", torch.tensor(0.5))

    def forward(self, query, entities):          # query [B,d], entities [B,N,d]
        B, N, _ = entities.shape
        q = self.q(query).view(B, self.h, 1, self.dh)
        k = self.k(entities).view(B, N, self.h, self.dh).transpose(1, 2)   # [B,h,N,dh]
        v = self.v(entities).view(B, N, self.h, self.dh).transpose(1, 2)
        score = (q * k).sum(-1) / (self.dh ** 0.5)                          # [B,h,N]
        lam = self.sparsity_lambda
        w = lam * sparsemax(score, dim=-1) + (1.0 - lam) * F.softmax(score, dim=-1)
        ctx = (w.unsqueeze(-1) * v).sum(2).reshape(B, self.d)               # [B,d]
        return ctx


class SDAModelBase(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 self_dim, allies_dim):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        cfg = model_config.get("custom_model_config", {})
        ch   = cfg.get("cnn_channels", 32)
        d    = cfg.get("hidden_dim", 128)
        heads = cfg.get("heads", 4)
        drop = cfg.get("dropout", 0.05)

        # 覆盖栅格 CNN（与 flat 基线一致，保证公平）
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), nn.GroupNorm(1, ch), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(1, ch), nn.GELU(),
            nn.AdaptiveAvgPool2d(1))

        self.grid_proj   = nn.Linear(ch, d)          # 实体 token: 覆盖/前沿
        self.allies_proj = nn.Linear(allies_dim, d)  # 实体 token: 队友 + UGV
        self.self_proj   = nn.Linear(self_dim, d)     # query
        self.eam         = EntityAttention(d, heads)
        self.fuse        = MLPBlock(2 * d, d, d, drop)
        self.actor_head  = nn.Linear(d, num_outputs)
        self.critic_head = nn.Linear(d, 1)
        self._cur_value = None

        # 稳定初始化：防止第 1 轮就出 NaN
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # actor/critic head 用小尺度初始化，避免初始输出过大
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["observation"]
        cov = obs["coverage_grid"].float().unsqueeze(1)
        st  = obs["self_state"].float()
        al  = obs["allies_state"].float()

        g  = self.grid_cnn(cov).flatten(1)
        sq = self.self_proj(st)
        entities = torch.stack([self.grid_proj(g), self.allies_proj(al)], dim=1)  # [B,2,d]
        ctx = self.eam(sq, entities)

        # NaN 保护：sparsemax 数值不稳定时可能产出 NaN，用零向量替代防止梯度爆炸
        if torch.isnan(ctx).any():
            ctx = torch.where(torch.isnan(ctx), torch.zeros_like(ctx), ctx)
        if torch.isnan(sq).any():
            sq = torch.where(torch.isnan(sq), torch.zeros_like(sq), sq)

        fused = self.fuse(torch.cat([sq, ctx], dim=-1))

        # 再次检查 fused
        if torch.isnan(fused).any():
            fused = torch.where(torch.isnan(fused), torch.zeros_like(fused), fused)

        logits = self.actor_head(fused)
        # 钳位 logits 防止极端值导致 Normal 分布参数爆炸
        logits = logits.clamp(-10.0, 10.0)
        self._cur_value = self.critic_head(fused)
        return logits, state

    def value_function(self):
        return self._cur_value.squeeze(-1)


class RLlibSDAUAVModel(SDAModelBase):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name,
                         self_dim=7, allies_dim=8)


class RLlibSDAUGVModel(SDAModelBase):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name,
                         self_dim=5, allies_dim=10)


# ── λ 线性 ramp（Chen 式动态稀疏化）──────────────────────────────────
# RLlib 模型拿不到全局训练步；如需精确复现 Chen 的"早期 softmax→后期 sparsemax"，
# 在训练脚本里加一个 callback，按 env-step 比例更新每个 worker 上模型的 sparsity_lambda：
#
#   class SparsityRamp(DefaultCallbacks):
#       def on_train_result(self, *, algorithm, result, **kw):
#           frac = min(1.0, result["num_env_steps_sampled"] / TOTAL_STEPS)
#           algorithm.env_runner_group.foreach_env_runner(
#               lambda er: [setattr(m.eam, "sparsity_lambda",
#                                   torch.tensor(frac)) for m in er.module.values()
#                           if hasattr(getattr(m,'eam',None),'sparsity_lambda')])
#
# 不加 callback 时默认 λ=0.5（稳态混合），已是有效的稀疏注意力，可直接训练。
