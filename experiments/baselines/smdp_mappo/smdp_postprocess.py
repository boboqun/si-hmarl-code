"""
smdp_postprocess.py  (V6 — Hybrid GAE + Episode 级标准化)
===========================================================
SMDP-MAPPO 核心组件①：时域解耦（Temporal Decoupling）

V6 架构思想：
    - Critic 生活在物理时钟里，接收标准 GAE 的密集平滑信号
    - Actor  生活在决策时钟里，接收 Macro-GAE 的跨宏动作因果信号

具体实现：
    Phase 0: 调用 RLlib 标准 compute_advantages() → 密集 VALUE_TARGETS 给 Critic
    Phase 1: 前向折叠宏动作 → macro_deltas
    Phase 2: 逆向递归 → macro_advs
    Phase 3: 仅在决策点写入 ADVANTAGES 给 Actor + Episode 级标准化

与 V5 的关键差异：
    1. Critic 的 VALUE_TARGETS 由标准 GAE 提供（低方差），不再用反向 TD(1)（高方差）
    2. Advantage 标准化在 Episode 级完成（~24 个决策点），不在 minibatch 中做
    3. Config B/D 分流处理，保证消融实验公平性

与 my_method/ 的关系：
    零修改。仅通过 sample_batch 中的 obs 和 vf_preds 工作。
"""

import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages


def smdp_postprocess_fn(policy, sample_batch,
                         other_agent_batches=None, episode=None):
    """
    V6 Hybrid GAE 后处理。

    Parameters
    ----------
    policy : Policy
        当前策略实例，用于读取 gamma/lambda 和消融开关。
    sample_batch : SampleBatch
        单个 agent 的完整 episode 轨迹。

    Returns
    -------
    SampleBatch
        增加了 ADVANTAGES, VALUE_TARGETS, decision_mask 三个字段。
    """
    gamma = policy.config.get("gamma", 0.99)
    lam = policy.config.get("lambda", 0.95)

    rewards = sample_batch[SampleBatch.REWARDS]
    vf_preds = sample_batch[SampleBatch.VF_PREDS]
    # 兼容不同 RLlib 版本的 done 键名
    dones = sample_batch.get(
        SampleBatch.TERMINATEDS,
        sample_batch.get(SampleBatch.DONES,
                         np.zeros(len(rewards), dtype=bool))
    )
    T = len(rewards)

    # ── 消融开关（从 custom_model_config 读取）──
    custom_cfg = policy.config.get("model", {}).get("custom_model_config", {})
    use_macro_gae = custom_cfg.get("use_macro_gae", True)
    use_masking = custom_cfg.get("use_decision_masking", True)

    # ══════════════════════════════════════════════════════════════
    #  安全提取 action_mask → 计算合法动作数
    #  防御性双路解包：适配 Dict obs 和 Flatten obs 两种情况
    # ══════════════════════════════════════════════════════════════
    try:
        obs = sample_batch[SampleBatch.OBS]
        if isinstance(obs, dict):
            # 情况 A：RLlib 保持了原始 Dict 结构（NoPreprocessor）
            action_masks = obs["action_mask"]
        else:
            # 情况 B：RLlib 扁平化了 obs（DictFlatteningPreprocessor）
            # 用 RLlib 官方工具还原字典结构
            from ray.rllib.models.modelv2 import restore_original_dimensions
            obs_dict = restore_original_dimensions(
                np.asarray(obs),
                policy.observation_space,
                tensorlib="numpy"
            )
            action_masks = np.asarray(obs_dict["action_mask"])

        n_valids = action_masks.sum(axis=-1).astype(np.int32)

    except Exception as e:
        # ⚠️ 绝不静默降级！提取失败必须暴露
        import warnings
        warnings.warn(
            f"[SMDP-MAPPO] action_mask 提取失败: {e}. "
            f"obs type={type(sample_batch[SampleBatch.OBS])}. "
            f"Macro-GAE 退化为标准 GAE！请检查 RLlib 版本。"
        )
        n_valids = np.full(T, 2, dtype=np.int32)

    # ══════════════════════════════════════════════════════════════
    #  在覆写 n_valids 之前备份真实决策掩码
    #  确保消融配置 C（关 Macro-GAE，开 Masking）不被连带污染
    # ══════════════════════════════════════════════════════════════
    true_decision_mask = (n_valids > 1).astype(np.float32)

    # ══════════════════════════════════════════════════════════════
    #  V6 Phase 0: 标准 GAE → 密集 VALUE_TARGETS（治愈 Critic）
    #  调用 RLlib 官方 compute_advantages()，生成与基线一致的
    #  低方差 λ-加权 TD target，替代 V5 的高方差反向 TD(1)
    # ══════════════════════════════════════════════════════════════
    v_last = 0.0 if dones[-1] else float(vf_preds[-1])
    sample_batch = compute_advantages(
        sample_batch, v_last, gamma, lam,
        use_gae=True, use_critic=True
    )
    # 此时 sample_batch 已有标准 GAE 的 ADVANTAGES 和 VALUE_TARGETS

    # ══════════════════════════════════════════════════════════════
    #  Config A / C 快速返回：不使用 Macro-GAE
    #  标准 GAE 的 ADVANTAGES + Episode 级标准化
    # ══════════════════════════════════════════════════════════════
    if not use_macro_gae:
        adv = sample_batch[Postprocessing.ADVANTAGES].copy()
        if use_masking:
            # Config C: 标准 GAE + 决策点标准化，非决策点归零
            valid_idx = true_decision_mask.astype(bool)
            if valid_idx.sum() > 1:
                m = adv[valid_idx].mean()
                s = adv[valid_idx].std()
                adv[valid_idx] = (adv[valid_idx] - m) / max(s, 1e-8)
            adv[~valid_idx] = 0.0
        else:
            # Config A: 基线全量标准化（完全等价于标准 PPO）
            adv = (adv - adv.mean()) / max(adv.std(), 1e-8)

        sample_batch[Postprocessing.ADVANTAGES] = adv
        sample_batch["decision_mask"] = true_decision_mask
        return sample_batch

    # ══════════════════════════════════════════════════════════════
    #  以下为 Macro-GAE 路径（Config B / D）
    #  消融：关闭 Macro-GAE 时已在上方返回
    # ══════════════════════════════════════════════════════════════

    # 识别决策点（合法动作 > 1 的时间步）
    decision_indices = np.where(n_valids > 1)[0].tolist()

    # 防御：确保首帧为决策点（处理 rollout 从承诺期中段开始的情况）
    if len(decision_indices) == 0 or decision_indices[0] != 0:
        decision_indices.insert(0, 0)
    decision_indices.append(T)  # 哨兵

    # ══════════════════════════════════════════════════════════════
    #  Phase 1: 前向折叠 —— 每个宏动作段的 TD-Error
    # ══════════════════════════════════════════════════════════════
    macro_deltas = []
    macro_discount_factors = []   # γ^τ per macro-action
    macro_lambda_factors = []     # λ^τ per macro-action

    for i in range(len(decision_indices) - 1):
        t_start = decision_indices[i]
        t_end = decision_indices[i + 1]
        tau = t_end - t_start

        # 段内累积折扣奖励
        cum_r = 0.0
        discount = 1.0
        terminated_in_segment = False

        for t in range(t_start, t_end):
            cum_r += discount * rewards[t]
            if dones[t]:
                terminated_in_segment = True
                break
            discount *= gamma

        # 段末引导值
        if terminated_in_segment:
            v_next = 0.0
            gamma_tau = 0.0   # 终止后无未来
        else:
            v_next = float(vf_preds[t_end]) if t_end < T else v_last
            gamma_tau = discount   # = γ^τ（如果中途无终止）

        delta = cum_r + gamma_tau * v_next - float(vf_preds[t_start])

        macro_deltas.append(delta)
        macro_discount_factors.append(gamma_tau)
        # λ^τ 衰减下限：当 τ 很大时（如 1200），λ^1200 → 0
        # 导致 GAE 递归被完全截断，宏动作间失去信用传递
        # 保留最低 5% 的跨宏动作信用，防止 Advantage 退化为孤立 TD
        macro_lambda_factors.append(max(lam ** tau, 0.05))

    # ══════════════════════════════════════════════════════════════
    #  Phase 2: 逆向递归 —— 宏动作级 GAE
    #  A_k = δ_k + γ^τ_k · λ^τ_k · A_{k+1}
    # ══════════════════════════════════════════════════════════════
    n_macros = len(macro_deltas)
    macro_advs = np.zeros(n_macros, dtype=np.float32)
    last_adv = 0.0

    for k in reversed(range(n_macros)):
        macro_advs[k] = (macro_deltas[k]
                         + macro_discount_factors[k]
                         * macro_lambda_factors[k]
                         * last_adv)
        last_adv = macro_advs[k]

    # ══════════════════════════════════════════════════════════════
    #  V6 Phase 3: 决策点覆写 + Episode 级标准化
    #  替代 V5 的 O(N) 反向 TD(1) 循环
    #  VALUE_TARGETS 不动（由 Phase 0 标准 GAE 提供）
    # ══════════════════════════════════════════════════════════════
    advantages = np.zeros(T, dtype=np.float32)

    if use_masking:
        # ── Config D（完整版 SMDP-MAPPO）──
        # 仅在真实决策点写入 Macro-GAE advantage
        for i in range(len(decision_indices) - 1):
            t_start = decision_indices[i]
            if t_start < T and true_decision_mask[t_start]:
                advantages[t_start] = macro_advs[i]

        # 决策点标准化（~24 个样本，统计稳定）
        valid_idx = true_decision_mask.astype(bool)
        if valid_idx.sum() > 1:
            m = advantages[valid_idx].mean()
            s = advantages[valid_idx].std()
            advantages[valid_idx] = (advantages[valid_idx] - m) / max(s, 1e-8)
    else:
        # ── Config B（仅 Macro-GAE，无决策感知归一化）──
        # 将宏动作 advantage 均匀平铺到整个承诺期
        # 然后做全局标准化，模拟"不做决策感知归一化"的效果
        for i in range(len(decision_indices) - 1):
            t_start = decision_indices[i]
            t_end = decision_indices[i + 1]
            advantages[t_start:min(t_end, T)] = macro_advs[i]

        m = advantages.mean()
        s = advantages.std()
        advantages = (advantages - m) / max(s, 1e-8)

    # ── 写入 SampleBatch ──
    sample_batch[Postprocessing.ADVANTAGES] = advantages
    # 🚨 VALUE_TARGETS 保持 Phase 0 标准 GAE 的结果，不覆写！
    sample_batch["decision_mask"] = true_decision_mask

    return sample_batch
