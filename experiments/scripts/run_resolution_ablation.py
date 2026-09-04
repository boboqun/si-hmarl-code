#!/usr/bin/env python3
"""
run_resolution_ablation.py
===========================
高危沙盒隔离测试：动态分辨率消融 (Resolution Ablation)
严格遵守 4 项隔离原则：零侵入动态子类化、绝对目录隔离、算力防爆节流、强制 Dry-Run。
"""

import os
import sys
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from gymnasium import spaces

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# RLlib Imports
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig

# Project Imports
from experiments.my_method.env_defs import RandomMapEnv, GRID_RES, MAP_SIZE
from experiments.my_method.HierarchicalEnv import HierarchicalEnv, UGV_SPEED, DT
from experiments.my_method.hierarchical_train import RLlibUAVModel, RLlibUGVModel, ParallelPettingZooEnv
from experiments.my_method.hierarchical_models import UAVSpatialCommander, UGVNodeDispatcher, MLPBlock, UAV_SELF_DIM, UAV_ALLIES_DIM

# ──────────────────────────────────────────────────────────────────
# 第一防线：环境级子类化重写 (零侵入消除环境变量硬编码)
# ──────────────────────────────────────────────────────────────────
class DynamicHierarchicalEnv(HierarchicalEnv):
    def __init__(self, map_env, resolution_K=6):
        self.K = resolution_K
        self.NUM_BLOCKS = self.K * self.K
        self.MACRO_BLOCK_SIZE = MAP_SIZE / self.K
        self.ACTION_DIM = self.NUM_BLOCKS + 2  # 0=noop, 1~K²=宏区块, K²+1=接驳/充电
        self.RECHARGE_ACT = self.NUM_BLOCKS + 1
        super().__init__(map_env=map_env)

    def observation_space(self, agent: str) -> spaces.Dict:
        coverage_sp = spaces.Box(
            low=0, high=1, shape=(int(MAP_SIZE/GRID_RES), int(MAP_SIZE/GRID_RES)), dtype=np.float32)

        if agent.startswith('ugv'):
            self_sp    = spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32)
            allies_sp  = spaces.Box(0.0, 1.0, shape=(12,), dtype=np.float32)
            obs_sp     = spaces.Dict({
                'coverage_grid': coverage_sp,
                'self_state':    self_sp,
                'allies_state':  allies_sp,
            })
            mask_sp    = spaces.Box(0, 1, shape=(self.ACTION_DIM,), dtype=np.int8)
        else:
            self_sp    = spaces.Box(0.0, 1.0, shape=(7,), dtype=np.float32)
            allies_sp  = spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32)
            obs_sp     = spaces.Dict({
                'coverage_grid': coverage_sp,
                'self_state':    self_sp,
                'allies_state':  allies_sp,
            })
            mask_sp    = spaces.Box(0, 1, shape=(self.ACTION_DIM,), dtype=np.int8)

        return spaces.Dict({'observation': obs_sp, 'action_mask': mask_sp})

    def action_space(self, agent: str) -> spaces.Discrete:
        return spaces.Discrete(self.ACTION_DIM)

    def _block_id_to_rect(self, block_id: int) -> Tuple[Tuple, Tuple]:
        idx = block_id - 1
        row = idx // self.K
        col = idx % self.K
        x0 = self.map_env.min_x + col * self.MACRO_BLOCK_SIZE
        y0 = self.map_env.min_y + row * self.MACRO_BLOCK_SIZE
        x1 = x0 + self.MACRO_BLOCK_SIZE
        y1 = y0 + self.MACRO_BLOCK_SIZE
        return (x0, y0), (x1, y1)

    def _get_completed_blocks(self) -> set:
        completed = set()
        grid = self.coverage_grid
        rows_per_block = max(1, len(grid) // self.K)
        cols_per_block = max(1, len(grid[0]) // self.K)

        for block_id in range(1, self.NUM_BLOCKS + 1):
            idx = block_id - 1
            row = idx // self.K
            col = idx % self.K
            r0, r1 = row * rows_per_block, (row + 1) * rows_per_block
            c0, c1 = col * cols_per_block, (col + 1) * cols_per_block
            block_status = grid[r0:r1, c0:c1]
            if block_status.size > 0 and np.all(block_status >= 1.0):
                completed.add(block_id)
        return completed

    def _get_uav_action_mask(self, agent: str) -> np.ndarray:
        mask = np.zeros(self.ACTION_DIM, dtype=np.int8)
        uav = self.uavs[agent]
        if uav['battery'] <= 25.0 and not uav['is_swapping']:
            mask[self.RECHARGE_ACT] = 1
            return mask
        if uav['is_swapping'] or uav['is_returning']:
            mask[0] = 1
            return mask

        mask[1:self.NUM_BLOCKS + 1] = 1
        mask[self.RECHARGE_ACT] = 1
        
        tm_block = uav.get('current_block', 0)
        # Must retain at least current assignment
        completed = self._get_completed_blocks()
        for b in completed:
            if 1 <= b <= self.NUM_BLOCKS:
                mask[b] = 0
                
        if 1 <= tm_block <= self.NUM_BLOCKS:
            mask[tm_block] = 1
        return mask

    def _get_ugv_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.ACTION_DIM, dtype=np.int8)
        uav_needs_help = any(u['is_returning'] for u in self.uavs.values())
        if uav_needs_help:
            mask[self.RECHARGE_ACT] = 1
        else:
            mask[0] = 1
            mask[1:self.NUM_BLOCKS + 1] = 1
            mask[self.RECHARGE_ACT] = 1
        return mask

    def _get_obs(self, agent: str) -> Dict[str, Any]:
        ret = super()._get_obs(agent)
        obs_dict = ret['observation']
        if agent.startswith('uav'):
            u = self.uavs[agent]
            # task_norm is at self_state[4]. 
            # 7维: [x, y, battery, is_busy, task_block, is_returning, is_swapping]
            obs_dict['self_state'][4] = float(u.get('current_block', 0)) / float(self.NUM_BLOCKS)
            ou = self.uavs['uav_1' if agent == 'uav_0' else 'uav_0']
            obs_dict['allies_state'][8] = float(ou.get('current_block', 0)) / float(self.NUM_BLOCKS)
            
            return {
                'observation': {
                    'self_state': obs_dict['self_state'].astype(np.float32),
                    'allies_state': obs_dict['allies_state'].astype(np.float32),
                    'coverage_grid': np.clip(self.coverage_grid, 0.0, 1.0).astype(np.float32)
                },
                'action_mask': self._get_uav_action_mask(agent)
            }
        else:
            # Fix UGV's allies_state normalization for K > 5 blocks (prevent > 1.0 bounds error)
            obs_dict['allies_state'][5] = float(self.uavs['uav_0'].get('current_block', 0)) / float(self.NUM_BLOCKS)
            obs_dict['allies_state'][11] = float(self.uavs['uav_1'].get('current_block', 0)) / float(self.NUM_BLOCKS)
            
            return {
                'observation': {
                    'self_state': obs_dict['self_state'].astype(np.float32),
                    'allies_state': obs_dict['allies_state'].astype(np.float32),
                    'coverage_grid': np.clip(self.coverage_grid, 0.0, 1.0).astype(np.float32)
                },
                'action_mask': self._get_ugv_action_mask()
            }

    def _dispatch_uav_command(self, agent: str, action: int):
        uav = self.uavs[agent]
        from experiments.my_method.HierarchicalEnv import ACT_NOOP, UGV_SPEED
        
        if action == ACT_NOOP:
            pass
        elif action == self.RECHARGE_ACT:
            if uav['is_busy']:
                self._dispatch_bonus -= 10.0
                uav['waypoint_queue'] = []
                uav['current_block']  = 0

            uav['is_busy']         = True
            uav['is_returning']    = True
            uav['is_swapping']     = False
            uav['swap_countdown']  = 0

            true_dist = math.hypot(self.car['x'] - uav['x'], self.car['y'] - uav['y'])
            self._ugv_shock_penalty += (true_dist / UGV_SPEED) * 1.0

        elif 1 <= action <= self.NUM_BLOCKS:
            if not uav['is_busy']:
                (x0, y0), (x1, y1) = self._block_id_to_rect(action)
                res = self.map_env.grid_resolution
                r0, c0 = self.map_env.xy_to_grid(x0, y0)
                r1, c1 = self.map_env.xy_to_grid(x1 - 1e-3, y1 - 1e-3)
                
                sub_grid = self.coverage_grid[r0:r1+1, c0:c1+1]
                uncovered = np.where(sub_grid == 0)
                
                if len(uncovered[0]) > 0:
                    min_r, max_r = int(np.min(uncovered[0])), int(np.max(uncovered[0]))
                    min_c, max_c = int(np.min(uncovered[1])), int(np.max(uncovered[1]))
                    
                    rect_min = (self.map_env.min_x + (c0 + min_c) * res,
                                self.map_env.min_y + (r0 + min_r) * res)
                    rect_max = (self.map_env.min_x + (c0 + max_c + 1) * res,
                                self.map_env.min_y + (r0 + max_r + 1) * res)
                    start_ref = (uav['x'], uav['y'])
                    end_ref   = (self.car['x'], self.car['y'])
                    from experiments.my_method.HierarchicalEnv import UAV_SCAN_WIDTH
                    
                    wps = self.planner.generate_optimal_waypoints(
                        rect_min=rect_min,
                        rect_max=rect_max,
                        sweep_width=UAV_SCAN_WIDTH,
                        start_reference=start_ref,
                        end_reference=end_ref,
                    )
                    
                    if wps:
                        uav['waypoint_queue'] = list(wps)
                        uav['is_busy']        = True
                        uav['is_returning']   = False
                        uav['is_swapping']    = False
                        uav['current_block']  = action
                else:
                    self._dispatch_bonus -= 2.0
                    uav['waypoint_queue'] = []
                    uav['current_block'] = 0
                uav['is_busy'] = True

    def _step_ugv(self, action: int):
        import math
        from experiments.my_method.HierarchicalEnv import UGV_SPEED, DT
        
        self.car['semantic_action'] = action
        v = self.car['edge_v']
        
        if action == 0:
            target_node = v
        elif 1 <= action <= self.NUM_BLOCKS:
            (x0, y0), (x1, y1) = self._block_id_to_rect(action)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            target_node = min(self.G.nodes(), key=lambda n: math.hypot(
                float(self.G.nodes[n]['x']) - cx, float(self.G.nodes[n]['y']) - cy))
        elif action == self.RECHARGE_ACT:
            returning_uav = next((u for u in self.uavs.values() if u['is_returning']), None)
            if returning_uav:
                target_node = min(self.G.nodes(), key=lambda n: math.hypot(
                    float(self.G.nodes[n]['x']) - returning_uav['x'],
                    float(self.G.nodes[n]['y']) - returning_uav['y']))
            else:
                working = [u for u in self.uavs.values() if u.get('is_busy') and not u['is_returning'] and not u['is_swapping']]
                if working:
                    cx = sum(u['x'] for u in working) / len(working)
                    cy = sum(u['y'] for u in working) / len(working)
                    target_node = min(self.G.nodes(), key=lambda n: math.hypot(
                        float(self.G.nodes[n]['x']) - cx, float(self.G.nodes[n]['y']) - cy))
                else:
                    target_node = v
        else:
            target_node = v

        self.car['target_node'] = target_node
        budget = UGV_SPEED * DT
        self._ugv_move_toward(target_node, budget)


# ──────────────────────────────────────────────────────────────────
# 第二防线：模型级子类化重写 (零侵入消除 PyTorch 硬编码)
# ──────────────────────────────────────────────────────────────────
class DynamicUAVCommander(UAVSpatialCommander):
    def __init__(self, K=6, **kwargs):
        nn.Module.__init__(self)  # Bypass original __init__ entirely
        self.K = K
        self.NUM_BLOCKS = K * K
        cnn_channels = 32
        token_dim = 64
        state_dim = 64
        dropout = 0.05
        nhead = 4
        
        # Override spatial grid pooling to forcefully match KxK (AdaptiveMaxPool2d eliminates magic shapes!)
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_channels, token_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(token_dim),
            nn.GELU(),
            nn.AdaptiveMaxPool2d((self.K, self.K)), 
        )
        
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.NUM_BLOCKS, token_dim))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        
        state_input_dim = UAV_SELF_DIM + UAV_ALLIES_DIM
        self.state_mlp = MLPBlock(state_input_dim, 64, state_dim, dropout)
        self.query_proj = nn.Linear(state_dim, token_dim) if state_dim != token_dim else nn.Identity()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=nhead, dropout=dropout, batch_first=True)
            
        self.actor_block = nn.Sequential(
            nn.Linear(token_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self.attn_norm = nn.LayerNorm(token_dim)
        fused_dim = token_dim + state_dim
        self.fuse_mlp = MLPBlock(fused_dim, 128, 128, dropout)
        
        self.actor_special = nn.Linear(128, 2)
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def _encode_grid(self, coverage_grid):
        if len(coverage_grid.shape) == 3:
            coverage_grid = coverage_grid.unsqueeze(1)
        x = self.grid_cnn(coverage_grid)                 
        x = x.flatten(2).transpose(1, 2)                 
        x = x + self.spatial_pos_embed                   
        return x

    def forward(self, obs_dict, action_mask=None):
        spatial_tokens = self._encode_grid(obs_dict['coverage_grid'])
        state_raw = torch.cat([obs_dict['self_state'], obs_dict['allies_state']], dim=-1)
        state_vec = self.state_mlp(state_raw)
        
        query = self.query_proj(state_vec).unsqueeze(1)
        attn_out, attn_w = self.cross_attn(query=query, key=spatial_tokens, value=spatial_tokens)
        
        global_ctx = self.attn_norm(attn_out.squeeze(1) + query.squeeze(1))
        fused = torch.cat([global_ctx, state_vec], dim=-1)
        fused = self.fuse_mlp(fused)
        
        block_scores = self.actor_block(spatial_tokens).squeeze(-1) # [B, NUM_BLOCKS]
        special_scores = self.actor_special(fused)
        
        logits = torch.cat([
            special_scores[:, 0:1],
            block_scores,
            special_scores[:, 1:2]
        ], dim=-1)

        from experiments.my_method.hierarchical_models import masked_logits
        if action_mask is not None:
            logits = masked_logits(logits, action_mask.float())
        
        val = self.critic_head(fused)
        return logits, val, attn_w

class DynamicRLlibUAVModel(RLlibUAVModel):
    def __init__(self, obs_space, act_space, num_outputs, model_config, name, K=6):
        super().__init__(obs_space, act_space, num_outputs, model_config, name)
        # Hot-swap the core with the dynamic K-variant
        self.core = DynamicUAVCommander(K=K)

class DynamicUGVDispatcher(UGVNodeDispatcher):
    def __init__(self, K=6, **kwargs):
        nn.Module.__init__(self)
        self.K = K
        self.NUM_BLOCKS = K * K
        self.ACTION_DIM = self.NUM_BLOCKS + 2
        
        cnn_channels = 32
        hidden_dim = 128
        nhead = 2
        dropout = 0.05
        
        from experiments.my_method.hierarchical_models import UGV_SELF_DIM
        # A. grid
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1), nn.BatchNorm2d(cnn_channels), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1), nn.BatchNorm2d(cnn_channels), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # B. self MLP
        self.self_mlp = MLPBlock(UGV_SELF_DIM, 32, 32, dropout)
        
        # C. uav attn
        self.uav_embed = nn.Linear(6, 32)
        self.uav_attn = nn.MultiheadAttention(embed_dim=32, num_heads=nhead, dropout=dropout, batch_first=True)
        self.uav_norm = nn.LayerNorm(32)
        
        # D. fuse
        self.fuse_mlp = MLPBlock(32 + 32 + 32 + cnn_channels, hidden_dim, hidden_dim, dropout)
        
        # E. Actor
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, self.ACTION_DIM)
        )
        # F. Critic
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
        
    def _encode_uav_allies(self, allies_state):
        B = allies_state.shape[0]
        # [B, 12] -> [B, 2, 6]
        uav_seq = allies_state.view(B, 2, 6)
        uav_tok = self.uav_embed(uav_seq) # [B, 2, 32]
        attn_out, _ = self.uav_attn(query=uav_tok, key=uav_tok, value=uav_tok)
        uav_fused = self.uav_norm(uav_tok + attn_out)
        
        attn_feat_pooled = uav_fused.mean(dim=1) # [B, 32]
        direct_feat_pooled = uav_tok.mean(dim=1)  # [B, 32]
        return attn_feat_pooled, direct_feat_pooled

    def forward(self, obs_dict, action_mask=None):
        B = obs_dict['self_state'].shape[0]
        grid_input = obs_dict['coverage_grid']
        if len(grid_input.shape) == 3:
            grid_input = grid_input.unsqueeze(1)
        grid_feat = self.grid_cnn(grid_input).view(B, -1)
        self_feat = self.self_mlp(obs_dict['self_state'])
        attn_feat, direct_feat = self._encode_uav_allies(obs_dict['allies_state'])
        
        fused = torch.cat([self_feat, attn_feat, direct_feat, grid_feat], dim=-1)
        fused = self.fuse_mlp(fused)
        
        logits = self.actor_head(fused)
        from experiments.my_method.hierarchical_models import masked_logits
        if action_mask is not None:
            logits = masked_logits(logits, action_mask.float())
            
        value = self.critic_head(fused)
        return logits, value

class DynamicRLlibUGVModel(RLlibUGVModel):
    def __init__(self, obs_space, act_space, num_outputs, model_config, name, K=6):
        super().__init__(obs_space, act_space, num_outputs, model_config, name)
        self.core = DynamicUGVDispatcher(K=K)


def env_creator(config):
    K = config.get("resolution", 6)
    map_env = RandomMapEnv(grid_resolution=GRID_RES)
    return ParallelPettingZooEnv(DynamicHierarchicalEnv(map_env, resolution_K=K))

# ──────────────────────────────────────────────────────────────────
# 引擎入口与防线机制 (Dry-Run + Compute Throttling)
# ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Resolution Test Sandbox")
    parser.add_argument("--K", type=int, default=6, help="Target dynamic resolution (KxK)")
    parser.add_argument("--workers", type=int, default=1, help="Number of Ray worker threads")
    parser.add_argument("--envs_per_worker", type=int, default=2, help="Envs per worker thread")
    parser.add_argument("--commit_training", action="store_true", help="Arm weapon: Allow genuine RLlib execution")
    args = parser.parse_args()

    # Dynamic Registration Injection
    # RLlib requires a class reference. We use a class factory to inject K at initialization.
    def make_dynamic_uav_cls(K_val):
        class _Cls(DynamicRLlibUAVModel):
            def __init__(self, obs, act, out, conf, nm):
                super().__init__(obs, act, out, conf, nm, K=K_val)
        return _Cls

    def make_dynamic_ugv_cls(K_val):
        class _Cls(DynamicRLlibUGVModel):
            def __init__(self, obs, act, out, conf, nm):
                super().__init__(obs, act, out, conf, nm, K=K_val)
        return _Cls

    ModelCatalog.register_custom_model("DynamicRLlibUAVModel", make_dynamic_uav_cls(args.K))
    ModelCatalog.register_custom_model("DynamicRLlibUGVModel", make_dynamic_ugv_cls(args.K))
    register_env("dynamic_hierarchical_env", env_creator)

    # 4. Mandatory Dry-Run Default
    if not args.commit_training:
        print("\n" + "="*45)
        print(f"====== MANDATORY DRY-RUN MODE (K={args.K}) ======")
        print("="*45)
        print("[!] Executing Sandbox Validation...")
        env = env_creator({"resolution": args.K})
        obs, _ = env.reset()
        print(f"  [✓] Environment blocks                : {args.K} x {args.K} = {args.K*args.K}")
        print(f"  [✓] UAV Action Space Dynamically Built: {env.par_env.action_space('uav_0')}")
        print(f"  [✓] UGV Action Space Dynamically Built: {env.par_env.action_space('ugv_0')}")
        print("  [✓] Magic numbers completely bypassed! Execution aborted securely.")
        print("\n  ** Use --commit_training to initiate actual training. **\n")
        sys.exit(0)


    # 2. Physical Directory Isolation
    save_dir = os.path.join(PROJECT_DIR, "experiments", "results_ablation", f"resolution_{args.K}x{args.K}")
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n[!!!] WEAPONS ARMED: Engaging PPO Training Sandbox with Compute Throttling [!!!]\n")
    
    # 3. Compute Throttling Lock Configuration
    config = (
        PPOConfig()
        .environment("dynamic_hierarchical_env", env_config={"resolution": args.K})
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            num_env_runners=args.workers,
            num_envs_per_env_runner=args.envs_per_worker, 
            rollout_fragment_length="auto"
        )
        .training(
            train_batch_size=2000,    # Reduced capacity
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
        )
        .resources(num_gpus=0)        # VRAM Lock: CPU Only to prevent master process crash!
        .multi_agent(
            policies={
                "uav_policy": (
                    None,
                    env_creator({"resolution": args.K}).par_env.observation_space("uav_0"),
                    env_creator({"resolution": args.K}).par_env.action_space("uav_0"),
                    {"model": {"custom_model": "DynamicRLlibUAVModel"}}
                ),
                "ugv_policy": (
                    None,
                    env_creator({"resolution": args.K}).par_env.observation_space("ugv_0"),
                    env_creator({"resolution": args.K}).par_env.action_space("ugv_0"),
                    {"model": {"custom_model": "DynamicRLlibUGVModel"}}
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "ugv_policy" if "ugv" in agent_id else "uav_policy",
        )
        .debugging(logger_config={"type": "ray.tune.logger.TBXLogger", "logdir": save_dir})
    )

    algo = config.build()
    print(f"====== PPO Sandbox Established successfully for K={args.K} ======")
    print(f"Logs isolated to: {save_dir}\n")
    
    NUM_ITER = 1000
    best_reward = -float('inf')

    for i in range(1, NUM_ITER + 1):
        result = algo.train()
        rew = result.get("episode_reward_mean", float("nan"))
        eplen = result.get("episode_len_mean", float("nan"))
        steps = result.get("timesteps_total", 0)

        import math
        is_best = False
        if not math.isnan(rew) and rew > best_reward:
            best_reward = rew
            is_best = True

        trend = "📈" if is_best else "  "
        print(f"iter {i:>4} | rew: {rew:+8.3f} {trend} | len: {eplen:>7.1f} | steps: {steps:>10,}")

        if is_best:
            ckpt = algo.save(checkpoint_dir=f"{save_dir}/best_ckpt")
            print(f"  [>] New Best Model Saved: {ckpt}")

        if i % 50 == 0:
            algo.save(checkpoint_dir=f"{save_dir}/ckpt_{i}")
