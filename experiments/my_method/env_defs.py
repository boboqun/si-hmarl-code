"""
env_defs.py
===========
Ray RLlib 训练用的环境类定义模块。

必须放在独立文件（不能在 __main__ 中定义），
因为 Ray 用 pickle 将环境发到 remote worker 进程，
worker 反序列化时需要从一个可 import 的模块重建类。

包含：
    RandomMapEnv          —— 过程化随机地图生成器（固定物理边界）
    RandomSimulationEnv   —— Zero-Padding 拦截层，覆写 obs/action_space
"""

import math
import random
import functools

import numpy as np
import networkx as nx
from gymnasium import spaces

import os
import sys
# Inject current directory to sys.path for Ray remote worker Unpickling 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SimulationEnv import SimulationEnv

# ── 全局设定（必须与 train.py 中的常量一致）──────────────
MAP_SIZE         = 2000.0  # 物理地图边长（米）
MAX_NODES = 200     # UGV move 动作空间容量上限
GRID_RES  = 100.0   # 单格物理尺寸（米）—— 保持 20×20 矩阵输入
GRID_ROWS = int(MAP_SIZE / GRID_RES)   # = 20
GRID_COLS = int(MAP_SIZE / GRID_RES)   # = 20
NUM_NODES_MIN = 20   # 路网节点数下限（尺度评估按面积倍数放大,默认与原字面量一致）
NUM_NODES_MAX = 50   # 路网节点数上限


# ────────────────────────────────────────────────
#  RandomMapEnv：固定物理画框 + 随机拓扑生成
# ────────────────────────────────────────────────

class RandomMapEnv:
    """
    每次 reset() 时过程化生成随机路网。

    为何必须固定物理边界？
        RLlib 在首次 rollout 时把 obs/action space 的 shape 编译进静态图。
        若后续 reset() 的 coverage_grid 尺寸或 action_mask 长度发生变化，
        会立即报 Shape Mismatch。固定画框 + Zero-Padding 彻底消除此问题。

    接口要求（兼容 SimulationEnv）:
        G_proj, min_x/max_x, min_y/max_y,
        rows, cols, grid_resolution, coverage_grid,
        xy_to_grid(x, y) → (row, col)
    """

    def __init__(self, grid_resolution: float = GRID_RES, seed: int = None):
        self.grid_resolution = grid_resolution

        # 固定物理边界
        self.min_x = 0.0
        self.max_x = MAP_SIZE
        self.min_y = 0.0
        self.max_y = MAP_SIZE

        # 固定栅格尺寸（永不改变）
        self.rows = GRID_ROWS
        self.cols = GRID_COLS
        self.coverage_grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # 首次初始化生成一张地图
        self.regenerate(seed=seed)

    def regenerate(self, seed: int = None):
        """
        生成全新路网拓扑。
        采用 MST + 随机捷径算法，确保 100% 全局连通且符合真实农田机耕道特征。
        """
        rng = random.Random(seed)
        num_nodes = rng.randint(NUM_NODES_MIN, NUM_NODES_MAX)   # 节点数随机化，增加地图多样性
        self.G_proj = self._build_sparse_connected_graph(num_nodes=num_nodes, rng=rng)

        # 重置栅格（尺寸固定，只清零）
        self.coverage_grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

    def _build_sparse_connected_graph(self, num_nodes: int = 40,
                                      rng: random.Random = None) -> nx.Graph:
        """
        生成符合农田机耕道特征的稀疏且 100% 全局连通路网。

        算法：
          1. 随机撤点：在地图范围内随机生成 num_nodes 个节点
          2. MST：构建最小生成树，保证 100% 连通且极度稀疏
          3. 增加随机环路：顺截径成环，类似真实岁道叉路
          4. 边权重：全部采用 length = 甋氏物理距离
        """
        if rng is None:
            rng = random.Random()

        G = nx.Graph()

        # Step 1: 随机撤点
        node_ids = [f"n_{i}" for i in range(num_nodes)]
        coords   = {}
        for nid in node_ids:
            x = rng.uniform(self.min_x, self.max_x)
            y = rng.uniform(self.min_y, self.max_y)
            G.add_node(nid, x=x, y=y)
            coords[nid] = (x, y)

        # Step 2: 构建全连图并提取 MST
        temp_G = nx.Graph()
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                u, v   = node_ids[i], node_ids[j]
                xi, yi = coords[u]
                xj, yj = coords[v]
                dist = math.hypot(xi - xj, yi - yj)
                temp_G.add_edge(u, v, weight=dist)
        mst = nx.minimum_spanning_tree(temp_G, weight='weight')
        for u, v, data in mst.edges(data=True):
            G.add_edge(u, v, length=data['weight'])

        # Step 3: 随机增加局部捷径（形成环路）
        num_shortcuts = max(1, int(num_nodes * 0.15))
        node_list = list(G.nodes())
        for _ in range(num_shortcuts):
            u = rng.choice(node_list)
            xu, yu = coords[u]
            candidates = [n for n in node_list
                          if n != u and not G.has_edge(u, n)]
            if not candidates:
                continue
            # 按距离排序，唃向连接较近的节点
            candidates.sort(key=lambda n: math.hypot(
                xu - coords[n][0], yu - coords[n][1]))
            v = rng.choice(candidates[:3])
            dist = math.hypot(xu - coords[v][0], yu - coords[v][1])
            G.add_edge(u, v, length=dist)

        return G


    def xy_to_grid(self, x: float, y: float):
        """物理坐标 → 栅格索引 (row, col)，自动截断越界。"""
        col = int((x - self.min_x) / self.grid_resolution)
        row = int((y - self.min_y) / self.grid_resolution)
        return max(0, min(row, self.rows - 1)), max(0, min(col, self.cols - 1))


# ────────────────────────────────────────────────
#  RandomSimulationEnv：Zero-Padding 拦截层
# ────────────────────────────────────────────────

class RandomSimulationEnv(SimulationEnv):
    """
    在父类 SimulationEnv 上叠加四层 Padding 拦截，使 obs/action space
    维度在整个训练期间保持固定，满足 RLlib 静态计算图要求。

    注意：observation_space / action_space 不使用 @lru_cache，
    避免 pickle 序列化失败（Ray remote worker 无法序列化 cached_method）。
    """

    def __init__(self, map_env: RandomMapEnv):
        self._random_map = map_env
        # 必须在 super().__init__() 之前初始化缓存属性！
        # 因为父类 __init__ 内部会调用 observation_space() / action_space()，
        # 如果此时缓存属性还未存在，会立即 AttributeError。
        self._ugv_obs_space = None
        self._ugv_act_space = None
        super().__init__(map_env=map_env)

    # ── 拦截层 ①: 固定 observation_space ──────

    def observation_space(self, agent: str):
        """ugv_0 的 action_mask 强制固定为 (MAX_NODES,)。"""
        if agent == "ugv_0":
            if self._ugv_obs_space is None:
                base = super().observation_space(agent)
                new  = dict(base.spaces)
                new["action_mask"] = spaces.Box(
                    low=0, high=1, shape=(MAX_NODES,), dtype=np.int8
                )
                self._ugv_obs_space = spaces.Dict(new)
            return self._ugv_obs_space
        return super().observation_space(agent)

    # ── 拦截层 ②: 固定 action_space ───────────

    def action_space(self, agent: str):
        """ugv_0 的 move 固定为 Discrete(MAX_NODES)。"""
        if agent == "ugv_0":
            if self._ugv_act_space is None:
                base = super().action_space(agent)
                new  = dict(base.spaces)
                new["move"] = spaces.Discrete(MAX_NODES)
                self._ugv_act_space = spaces.Dict(new)
            return self._ugv_act_space
        return super().action_space(agent)

    # ── 拦截层 ③: Zero-Padding action_mask ────

    def _get_obs(self, agent: str):
        """
        将真实 mask（长 N）填入 MAX_NODES 长度的零向量前 N 位。
        Padding 区（N～MAX_NODES）保持 0，视为非法节点。
        """
        obs = super()._get_obs(agent)
        if agent == "ugv_0":
            true_mask = obs["action_mask"]
            padded    = np.zeros(MAX_NODES, dtype=np.int8)
            n         = min(len(true_mask), MAX_NODES)
            padded[:n] = true_mask[:n]
            obs["action_mask"] = padded
        return obs

    def step(self, actions: dict):
        """
        UGV 动作安全拦截层（双重保险）。

        问题背景：
            _densify_graph 在长边上插入虚拟节点（如 'vnode_11_28_19'），
            使得 self.num_nodes 远多于原始节点数。RL 网络输出的 move_idx
            即使在 [0, num_nodes) 范围内，也可能映射到非相邻的虚拟节点，
            直接传入 _move_ugv 就会导致 KeyError。

        修复逻辑（三层检查）：
            ① 强制转 Python int（防止 np.int32 被 dict.get 当成不同 key）
            ② 越界裁剪：clamp 到 [0, num_nodes-1]
            ③ 邻居合法性检查：目标节点必须是 UGV 当前位置的实际邻居，
               否则强制回退到 edge_u（停留原节点，永远合法）
        """
        if "ugv_0" in actions:
            raw = actions["ugv_0"].get("move", 0)
            move_idx = int(raw)   # ① 强制转 Python int（处理 np.int32）

            # ② 越界裁剪
            move_idx = max(0, min(move_idx, self.num_nodes - 1))

            # ③ 邻居合法性检查
            target_node = self.nodes_list[move_idx]
            current_u   = self.car["edge_u"]
            current_v   = self.car["edge_v"]

            # UGV 当前位置的合法邻居 = current_u 的邻居 ∪ current_v 的邻居
            valid_neighbors = set(self.G.neighbors(current_u))
            valid_neighbors.add(current_u)   # 停留在起点也合法
            if current_v != current_u:
                valid_neighbors |= set(self.G.neighbors(current_v))
                valid_neighbors.add(current_v)

            if target_node not in valid_neighbors:
                # 目标节点不合法 → 停留在当前 edge_u
                move_idx = self.node_to_idx.get(current_u, 0)

            actions["ugv_0"]["move"] = move_idx

        return super().step(actions)


    # ── reset: 刷新地图 + 重新初始化 ───────────

    def reset(self, seed=None, options=None):
        """每局 episode 开始前先重新生成随机路网。"""
        # 刷新随机路网
        self._random_map.regenerate(seed=seed)

        # 同步图引用到父类属性
        self.map_env     = self._random_map
        self.G           = self._densify_graph(self._random_map.G_proj, max_len=20.0)
        self.nodes_list  = list(self.G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes_list)}
        self.idx_to_node = {i: n for i, n in enumerate(self.nodes_list)}  # 修复：同步反向映射
        self.num_nodes   = len(self.nodes_list)
        self.grid_rows   = self._random_map.rows
        self.grid_cols   = self._random_map.cols

        # 新地图生成后使缓存失效
        self._ugv_obs_space = None
        self._ugv_act_space = None

        return super().reset(seed=seed, options=options)
