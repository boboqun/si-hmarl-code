"""
smdp_policy.py  (V6 — 梯度缩放对齐 + Episode 级标准化)
===============================================================
SMDP-MAPPO 核心组件②：决策感知优势归一化（Decision-Aware Advantage Normalization）

V6 关键改进：
    1. Advantage 标准化移至 postprocess（Episode 级，~24 决策），
       不再在 minibatch（~1.5 决策）中做，消除统计崩溃
    2. Actor Loss / Entropy / KL 分母改为 total_count（minibatch 大小），
       对齐与 Critic Loss（.mean()）的梯度缩放
    3. Critic Loss 保持全样本学习（不掩码）

覆写了 PPOTorchPolicy 的两个方法：
    - postprocess_trajectory → 接入 Hybrid GAE
    - loss → 接入决策掩码 + 梯度缩放对齐 + VF-Clip + KL

与 my_method/ 的关系：
    零修改。继承 RLlib 标准 PPOTorchPolicy，仅覆写 postprocess 和 loss。
"""

import numpy as np
import torch
import torch.nn.functional as F

from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing


class SMDPPPOTorchPolicy(PPOTorchPolicy):
    """
    SMDP-MAPPO 自定义 PPO Policy。

    与标准 PPOTorchPolicy 的关键差异：
    A. Advantage 已在 postprocess Episode 级标准化，loss 中直接使用
    B. Actor Loss / Entropy / KL 分母为 total_count（对齐基线梯度缩放）
    C. Critic Loss 在全部样本上计算（不掩码，标准 GAE value targets）

    保留了标准 PPO 的 VF Clipping 和 KL Penalty，确保与基线公平对比。
    """

    def postprocess_trajectory(self, sample_batch,
                                other_agent_batches=None, episode=None):
        """替换默认 GAE 为 Macro-GAE。"""
        # 延迟导入，避免循环依赖
        from smdp_postprocess import smdp_postprocess_fn
        return smdp_postprocess_fn(
            self, sample_batch, other_agent_batches, episode
        )

    def loss(self, model, dist_class, train_batch):
        """
        V6 PPO Loss，核心修改：
        - Advantage 已在 postprocess Episode 级标准化，此处直接使用
        - Actor Loss / Entropy / KL 分母为 total_count（梯度缩放对齐）
        - Critic Loss 在全样本上计算（带 VF Clipping，标准 GAE targets）
        """

        # ═══ 1. 模型前向（必须最先执行，以构建计算图）═══
        logits, state = model(train_batch)
        curr_dist = dist_class(logits, model)

        # ═══ 2. 安全提取决策掩码 ═══
        custom_cfg = self.config.get("model", {}).get("custom_model_config", {})
        use_masking = custom_cfg.get("use_decision_masking", True)

        if use_masking and "decision_mask" in train_batch:
            # RLlib 底层已将 SampleBatch 中的数据转为 GPU Tensor
            decision_mask = train_batch["decision_mask"].float()
        else:
            # 消融关闭或 mask 不可用时，全量参与
            decision_mask = torch.ones_like(
                train_batch[SampleBatch.REWARDS], dtype=torch.float32
            )
        valid_count = torch.clamp(decision_mask.sum(), min=1.0)
        total_count = float(decision_mask.shape[0])  # minibatch 大小

        # ═══ 3. Advantage 直接使用（已在 postprocess Episode 级标准化）═══
        adv = train_batch[Postprocessing.ADVANTAGES]

        # ═══ 4. Actor Loss（决策掩码 + total_count 梯度对齐）═══
        logp = curr_dist.logp(train_batch[SampleBatch.ACTIONS])
        ratio = torch.exp(logp - train_batch[SampleBatch.ACTION_LOGP])

        clip_param = self.config["clip_param"]
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv
        actor_loss_per_sample = -torch.min(surr1, surr2)
        actor_loss = (actor_loss_per_sample * decision_mask).sum() / total_count

        # ═══ 5. Entropy（决策掩码 + total_count 梯度对齐）═══
        entropy = (curr_dist.entropy() * decision_mask).sum() / total_count

        # ═══ 6. Critic Loss（全局，不掩码，带 VF Clipping）═══
        vf_preds_curr = model.value_function()
        vf_targets = train_batch[Postprocessing.VALUE_TARGETS]
        # 读取上一轮 Critic 预测值（用于 clipping 基准）
        vf_preds_old = train_batch.get(SampleBatch.VF_PREDS,
                                        vf_preds_curr.detach())
        vf_clip_param = self.config.get("vf_clip_param", 500.0)

        # 标准 PPO clipped VF loss
        vf_loss_unclipped = (vf_preds_curr - vf_targets).pow(2.0)
        vf_clipped = vf_preds_old + torch.clamp(
            vf_preds_curr - vf_preds_old, -vf_clip_param, vf_clip_param
        )
        vf_loss_clipped = (vf_clipped - vf_targets).pow(2.0)
        vf_loss = torch.max(vf_loss_unclipped, vf_loss_clipped).mean()

        # ═══ 6.5 KL 散度约束（带掩码，维护 Trust Region）═══
        # RLlib 的动态 KL 系数调度器依赖 tower_stats["kl"]
        if SampleBatch.ACTION_DIST_INPUTS in train_batch:
            prev_dist = dist_class(
                train_batch[SampleBatch.ACTION_DIST_INPUTS], model
            )
            mean_kl = (
                prev_dist.kl(curr_dist) * decision_mask
            ).sum() / total_count
        else:
            mean_kl = torch.tensor(0.0, device=logits.device)

        # ═══ 7. 汇总 ═══
        # RLlib 内部动态更新 kl_coeff 属性
        kl_coeff = getattr(self, "kl_coeff",
                           self.config.get("kl_coeff", 0.2))
        # entropy_coeff 由 RLlib 的 schedule 管理
        entropy_coeff = self.entropy_coeff

        total_loss = (
            actor_loss
            + self.config["vf_loss_coeff"] * vf_loss
            - entropy_coeff * entropy
            + kl_coeff * mean_kl
        )

        # ═══ 8. 监控指标（Tensorboard）═══
        # 键名必须与 RLlib PPOTorchPolicy.stats_fn 期望的完全一致
        from ray.rllib.utils.torch_utils import explained_variance
        vf_explained = explained_variance(
            vf_targets, vf_preds_curr
        )
        model.tower_stats.update({
            "total_loss": total_loss.detach(),
            "mean_policy_loss": actor_loss.detach(),
            "mean_vf_loss": vf_loss.detach(),
            "mean_entropy": entropy.detach(),
            "mean_kl_loss": mean_kl.detach(),
            "vf_explained_var": vf_explained.detach(),
            "decision_ratio": (
                valid_count / float(decision_mask.numel())
            ).detach(),
        })

        return total_loss

    def stats_fn(self, train_batch):
        """覆写 stats_fn，在标准 PPO 指标之上增加 decision_ratio。"""
        from ray.rllib.utils.numpy import convert_to_numpy
        return convert_to_numpy({
            "cur_kl_coeff": self.kl_coeff,
            "cur_lr": self.cur_lr,
            "total_loss": torch.mean(
                torch.stack(self.get_tower_stats("total_loss"))),
            "policy_loss": torch.mean(
                torch.stack(self.get_tower_stats("mean_policy_loss"))),
            "vf_loss": torch.mean(
                torch.stack(self.get_tower_stats("mean_vf_loss"))),
            "vf_explained_var": torch.mean(
                torch.stack(self.get_tower_stats("vf_explained_var"))),
            "kl": torch.mean(
                torch.stack(self.get_tower_stats("mean_kl_loss"))),
            "entropy": torch.mean(
                torch.stack(self.get_tower_stats("mean_entropy"))),
            "entropy_coeff": self.entropy_coeff,
            # SMDP-MAPPO 专属指标
            "decision_ratio": torch.mean(
                torch.stack(self.get_tower_stats("decision_ratio"))),
        })

