"""
SimulationVisualizer.py
=======================
基于 Pygame 的异构多智能体模拟器 —— 实时可视化界面

功能说明：
    ┌────────────────────────────┬─────────────────┐
    │  地图区域 (800×800)         │  HUD 信息面板   │
    │  - 覆盖率栅格（动态染色）    │  - 时间 / FPS   │
    │  - 路网拓扑（预渲染）        │  - 车辆状态      │
    │  - 无人机扫描带（半透明）    │  - 无人机状态    │
    │  - 车辆 / 无人机 实时位置   │  - 覆盖率进度条  │
    └────────────────────────────┴─────────────────┘

键盘控制：
    SPACE  ── 暂停 / 继续
    R      ── 重置环境
    Q / ESC── 退出
    ↑ / ↓  ── 加速 / 减速（FPS）

依赖：
    pip install pygame
    （SimulationEnv.py 必须位于同一目录）
"""

import sys
import math
import random
import os

# ── 在 pygame.init() 之前设置 SDL HiDPI 提示（macOS Retina 关键）──
os.environ.setdefault('SDL_VIDEO_HIGHDPI', '1')
os.environ.setdefault('SDL_VIDEO_ALLOW_SCREENSAVER', '1')

import pygame
import pygame.gfxdraw
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 导入仿真引擎与 Wrapper
from SimulationEnv import SimulationEnv, _MockMapEnv
from DispatcherWrapper import DispatcherWrapper

# ──────────────────────────────────────────────
#  布局与颜色常数
# ──────────────────────────────────────────────

# ── 超采样倍率设置 ──────────────────────────────────────────
# 所有绘制在 SS_SCALE 倍的离屏 Surface 上完成，最终 smoothscale 缩回真实窗口。
# 这是在 pygame 软件渲染下获得 Retina 级清晰度的最有效方法。
SS_SCALE       = 2               # 超采样倍率（2 = 4× 像素密度）

WIN_W, WIN_H   = 1400, 900       # 逻辑窗口尺寸（CSS pixels）

# 超采样绘制画布尺寸（实际像素数）
CANVAS_W = WIN_W * SS_SCALE
CANVAS_H = WIN_H * SS_SCALE

MAP_W, MAP_H   = 880  * SS_SCALE, 900 * SS_SCALE   # 地图区（超采样空间）
HUD_X          = MAP_W
HUD_W          = CANVAS_W - MAP_W

MAP_PADDING    = 28 * SS_SCALE
DRAW_W         = MAP_W - 2 * MAP_PADDING
DRAW_H         = MAP_H - 2 * MAP_PADDING

# 颜色调色板（现代深色主题，提升对比度）
C_BG           = ( 10,  12,  20)   # 窗口背景（更深）
C_MAP_BG       = ( 16,  20,  32)   # 地图底色
C_GRID_EMPTY   = ( 28,  33,  50)   # 未覆盖栅格
C_GRID_COVERED = ( 40, 180,  60)   # 已覆盖栅格（亮绿）
C_ROAD         = ( 55,  68,  98)   # 路网颜色
C_ROAD_NODE    = ( 90, 105, 145)   # 路网节点

C_CAR          = ( 80, 170, 255)   # 车辆（亮蓝）
C_CAR_AURA     = ( 80, 170, 255,  35)  # 车辆采集圈（带透明）
C_UAV1         = (255, 165,  45)   # 无人机 1（橙）
C_UAV2         = ( 70, 225, 200)   # 无人机 2（青）
C_SWAP_LOCK    = (255,  75,  95)   # 换电锁定状态色

C_HUD_BG       = ( 14,  18,  30)   # HUD 底色（更深）
C_HUD_CARD     = ( 22,  28,  45)   # HUD 卡片背景
C_HUD_LINE     = ( 38,  46,  72)   # HUD 分割线
C_TEXT_MAIN    = (230, 238, 255)   # 主文字（更亮）
C_TEXT_DIM     = (110, 122, 158)   # 次要文字
C_TEXT_WARN    = (255, 125,  55)   # 警告文字
C_BAR_BG       = ( 35,  43,  68)   # 进度条底色
C_BAR_GREEN    = ( 50, 210, 100)   # 进度条绿色
C_BAR_ORANGE   = (255, 165,  45)   # 进度条橙色
C_BAR_RED      = (225,  55,  55)   # 进度条红色
C_TITLE_ACCENT = ( 90, 130, 255)   # 标题强调色

FPS_DEFAULT    = 10               # 默认帧率


# ──────────────────────────────────────────────
#  坐标变换辅助类
# ──────────────────────────────────────────────

class CoordTransform:
    """
    物理坐标（米）→ 屏幕像素坐标 的双向变换器。
    保持纵轴：物理 y 轴朝上，屏幕 y 轴朝下，需要翻转。
    """
    def __init__(self, phys_min_x, phys_min_y, phys_max_x, phys_max_y,
                 screen_x0, screen_y0, screen_w, screen_h):
        self.p_min_x = phys_min_x
        self.p_min_y = phys_min_y
        self.p_range_x = phys_max_x - phys_min_x
        self.p_range_y = phys_max_y - phys_min_y
        self.sx0 = screen_x0
        self.sy0 = screen_y0
        self.sw  = screen_w
        self.sh  = screen_h

        # 缩放比（保持等比，取较小值以防裁切）
        self.scale = min(self.sw / self.p_range_x, self.sh / self.p_range_y)

        # 居中偏移
        self.offset_x = screen_x0 + (screen_w - self.p_range_x * self.scale) / 2
        self.offset_y = screen_y0 + (screen_h - self.p_range_y * self.scale) / 2

    def to_screen(self, px: float, py: float):
        sx = int(self.offset_x + (px - self.p_min_x) * self.scale)
        # Y 轴翻转
        sy = int(self.offset_y + (self.p_range_y - (py - self.p_min_y)) * self.scale)
        return sx, sy

    def length_to_px(self, meters: float) -> int:
        return max(1, int(meters * self.scale))


# ──────────────────────────────────────────────
#  MapEditor 辅助类
# ──────────────────────────────────────────────

class MapEditor:
    """
    启动初期的手绘路网编辑器。
    使用固定的缩放比例尺将屏幕像素映射到物理空间，供底层模拟引擎使用。
    """
    def __init__(self, map_w, map_h, scale_px_to_m=2.0):
        self.map_w = map_w
        self.map_h = map_h
        self.scale = scale_px_to_m  # 1px = X meters
        
        self.nodes = []   # [(sx, sy, px, py), ...]
        self.edges = []   # [(node_idx_u, node_idx_v), ...]
        self.selected_node_idx = None

        # 使用与主界面相同的中文字体列表，确保中文正常显示
        _cn = ['pingfangsc', 'hiraginosansgb', 'songtisc', 'stheitilight',
               'microsoftyahei', 'arialunicode']
        self.font_xl = pygame.font.SysFont(_cn, 32)
        self.font_md = pygame.font.SysFont(_cn, 20)
        self.font_sm = pygame.font.SysFont(_cn, 16)

    def handle_click(self, sx, sy):
        # 点击越界忽略
        if sx < 0 or sx > self.map_w or sy < 0 or sy > self.map_h:
            return

        # 检查是否点击了现有节点 (判断距离阈值，例如 15px)
        clicked_idx = None
        for i, (nx, ny, _, _) in enumerate(self.nodes):
            if math.hypot(sx - nx, sy - ny) < 15:
                clicked_idx = i
                break

        if clicked_idx is not None:
            # 点击了现有节点
            if self.selected_node_idx is None:
                # 之前没选中，现在选中它
                self.selected_node_idx = clicked_idx
            else:
                # 之前有选中，现在意图连线
                if self.selected_node_idx != clicked_idx:
                    edge = tuple(sorted((self.selected_node_idx, clicked_idx)))
                    if edge not in self.edges:
                        self.edges.append(edge)
                # 连线（或取消）后，清空选中状态
                self.selected_node_idx = None
        else:
            # 点击空白处，创建新节点
            px, py = sx * self.scale, sy * self.scale
            self.nodes.append((sx, sy, px, py))
            # 自动选中刚创建的节点，方便接下来的连续点击连线
            self.selected_node_idx = len(self.nodes) - 1

    def draw(self, surface):
        # 背景
        surface.fill(C_MAP_BG)

        # 1. 绘制已连接的边
        for u, v in self.edges:
            sx1, sy1, _, _ = self.nodes[u]
            sx2, sy2, _, _ = self.nodes[v]
            pygame.draw.line(surface, C_ROAD, (sx1, sy1), (sx2, sy2), 3)

        # 2. 绘制节点
        for i, (sx, sy, px, py) in enumerate(self.nodes):
            if i == self.selected_node_idx:
                # 选中状态高亮加大
                pygame.draw.circle(surface, (255, 120, 60), (sx, sy), 10)
                pygame.draw.circle(surface, (255, 255, 255), (sx, sy), 10, 2)
            else:
                pygame.draw.circle(surface, C_ROAD_NODE, (sx, sy), 6)
                pygame.draw.circle(surface, (255, 255, 255), (sx, sy), 6, 1)

        # 3. 绘制提示文字（全部使用 font_md / font_xl，支持中文）
        txt_title = self.font_xl.render("地图编辑器  Map Editor", True, (255, 200, 50))
        surface.blit(txt_title, (20, 18))

        hints = [
            ("鼠标左键点击空白处 → 创建节点",           (200, 210, 230)),
            ("点击已有节点选中，再点另一节点 → 连线",    (200, 210, 230)),
            ("按 [ENTER] 保存路网并进入仿真",           (100, 240, 130)),
        ]
        for i, (text, color) in enumerate(hints):
            surf = self.font_md.render(text, True, color)
            surface.blit(surf, (20, 60 + i * 28))

        if self.selected_node_idx is not None:
            lbl = self.font_md.render(
                f"已选中节点 {self.selected_node_idx}  —  再点击另一节点以连线",
                True, (255, 130, 60)
            )
            surface.blit(lbl, (20, 150))

        # 右下角节点 / 边计数
        stats = self.font_sm.render(
            f"节点: {len(self.nodes)}    边: {len(self.edges)}",
            True, (120, 130, 160)
        )
        surface.blit(stats, (20, self.map_h - 30))

    def build_mock_env(self):
        """将绘制好的节点和边转换为 _MockMapEnv 以供系统使用。"""
        # 如果什么都没画，给一个默认节点防崩
        if not self.nodes:
            cx, cy = self.map_w // 2, self.map_h // 2
            self.nodes.append((cx, cy, cx * self.scale, cy * self.scale))

        import networkx as nx
        from SimulationEnv import _MockMapEnv

        G = nx.Graph()
        for i, (sx, sy, px, py) in enumerate(self.nodes):
            # 将物理坐标存入图中
            G.add_node(i, x=px, y=py)

        for u, v in self.edges:
            n1 = G.nodes[u]
            n2 = G.nodes[v]
            # 边的物理长度 = 欧氏距离
            dist = math.hypot(n1['x'] - n2['x'], n1['y'] - n2['y'])
            G.add_edge(u, v, length=dist)

        # 计算地图物理边界
        pxs = [n[2] for n in self.nodes]
        pys = [n[3] for n in self.nodes]
        min_x, max_x = min(pxs) - 50.0, max(pxs) + 50.0
        min_y, max_y = min(pys) - 50.0, max(pys) + 50.0

        size_x = max_x - min_x
        size_y = max_y - min_y
        max_size = max(size_x, size_y, 100.0)

        # 实例化基于真实边界的 mock 环境（因为包含完整的环境属性约束，比纯净版好）
        env = _MockMapEnv(size=max_size, grid_resolution=2.0)
        env.min_x, env.max_x = min_x, max_x
        env.min_y, env.max_y = min_y, max_y
        # 使用我们的定制图替换默网格图
        env.G_proj = G
        
        # 重构覆盖率栅格尺寸
        env.cols = int(math.ceil((max_x - min_x) / env.grid_resolution))
        env.rows = int(math.ceil((max_y - min_y) / env.grid_resolution))
        env.coverage_grid = np.zeros((env.rows, env.cols), dtype=np.uint8)

        return env


# ──────────────────────────────────────────────
#  SimulationVisualizer 主类
# ──────────────────────────────────────────────

class SimulationVisualizer:
    """
    Pygame 可视化器主类。

    调用方式：
        vis = SimulationVisualizer(map_env)
        vis.run()
    """

    def __init__(self, map_env=None):
        # 根据是否传入外部地图环境决定初始状态
        self.app_state = 'EDITOR' if map_env is None else 'SIMULATION'

        # Pygame 初始化
        pygame.init()
        pygame.display.set_caption("多智能体路径规划模拟器 — VRP-D Visualizer")
        # 安全获取 HiDPI/SCALED 标志（并非所有 pygame 版本都暴露这些常量）
        _HIGHDPI = getattr(pygame, 'ALLOW_HIGHDPI', 0)
        _RESIZABLE = pygame.RESIZABLE
        self.screen = pygame.display.set_mode(
            (WIN_W, WIN_H),
            _HIGHDPI | _RESIZABLE
        )
        # 超采样离屏画布：所有绘制均在此进行，最终 smoothscale 到真实窗口
        self._canvas = pygame.Surface((CANVAS_W, CANVAS_H))
        self.clock   = pygame.time.Clock()
        self._init_fonts()
        self._emoji_cache = {}

        # 状态对象
        if self.app_state == 'EDITOR':
            self.map_editor = MapEditor(MAP_W, MAP_H)
            self.map_env = None
            self.sim = None
            self.tester = None
            self.state = None
            self.paused = True
            self.fps = FPS_DEFAULT
            self.done = False
            self.tick_cnt = 0
            self.reward_acc = 0.0
            self._uav_trails = [[], []]
        else:
            self.map_env = map_env
            self.sim = DispatcherWrapper(SimulationEnv(map_env=map_env))
            self.tester = None
            self.paused = True
            self.fps = FPS_DEFAULT
            self._do_reset()

    # ── 初始化 ──────────────────────────────────────────────────────

    def _init_fonts(self):
        """在超采样坐标系下初始化字体，所有尺寸乘以 SS_SCALE。"""
        # 在超采样空间里字体尺寸须 ×SS_SCALE，最终 smoothscale 缩回后依然清晰
        S = SS_SCALE
        chinese_fonts = ['pingfangsc', 'hiraginosansgb', 'songtisc', 'stheitilight',
                         'microsoftyahei', 'arialunicode']

        # 尝试加载系统高质量 TTF 字体（macOS SF 系列）
        sf_paths = [
            '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
            '/System/Library/Fonts/SFNS.ttf',
            '/System/Library/Fonts/SFNSRounded.ttf',
            '/Library/Fonts/Arial Unicode.ttf',
        ]
        ttf_path = next((p for p in sf_paths if os.path.exists(p)), None)

        def _make(size, bold=False):
            sz = size * S
            if ttf_path:
                try:
                    f = pygame.font.Font(ttf_path, sz)
                    f.bold = bold
                    return f
                except Exception:
                    pass
            return pygame.font.SysFont(chinese_fonts, sz, bold=bold)

        self.font_xl      = _make(24)
        self.font_lg      = _make(18)
        self.font_md      = _make(15)
        self.font_sm      = _make(13)
        self.font_xs      = _make(11)
        self.font_md_bold = _make(15, bold=True)
        self.font_sm_bold = _make(13, bold=True)

        src = ttf_path if ttf_path else 'SysFont'
        print(f"[*] 字体初始化完成 (SS={S}x, src={src})")

    def _get_emoji_surf(self, emoji_char, size=24):
        """
        利用 PIL 渲染彩色 Emoji 并转换为 Pygame Surface。
        使用固定的 160px 安全尺寸渲染以规避 'invalid pixel size' 错误，再缩放。
        """
        cache_key = (emoji_char, size)
        if cache_key in self._emoji_cache:
            return self._emoji_cache[cache_key]

        font_path = "/System/Library/Fonts/Apple Color Emoji.ttc"
        if not os.path.exists(font_path):
            return self.font_md.render(emoji_char, True, (255, 255, 255))

        try:
            # 使用固定安全尺寸 (160) 渲染，这是 Apple Color Emoji 的标准 Strike 之一
            safe_render_size = 160
            pil_font = ImageFont.truetype(font_path, safe_render_size)
            
            # 画布稍大一点确保不被裁切
            canvas_size = int(safe_render_size * 1.2)
            img = Image.new('RGBA', (canvas_size, canvas_size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((0, 0), emoji_char, font=pil_font, embedded_color=True)
            
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
                w, h = img.size
                # 按照高度等比缩放到请求的 size
                new_h = size
                new_w = int(w * (new_h / h))
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            raw_data = img.tobytes("raw", "RGBA")
            pg_surf = pygame.image.fromstring(raw_data, img.size, "RGBA")
            self._emoji_cache[cache_key] = pg_surf
            return pg_surf
        except Exception as e:
            print(f"[!] Emoji 渲染失败 ({emoji_char}): {e}")
            return self.font_md.render(emoji_char, True, (255, 255, 255))

    def _setup_simulation_from_editor(self):
        """用户编辑完成后，构建地图和仿真环境"""
        self.map_env = self.map_editor.build_mock_env()
        self.sim = DispatcherWrapper(SimulationEnv(map_env=self.map_env))
        
        self.app_state = 'SIMULATION'
        self._do_reset()

    def _do_reset(self):
        """重置仿真环境并重建所有图层缓存。"""
        self.obs, info = self.sim.reset()
        self.state = self._decode_obs_to_state(self.obs)
        self.done     = False
        self.tick_cnt = 0
        self.reward_acc = 0.0
        self.paused   = True
        self._uav_trails = [[], []]

        # 重建坐标变换器
        self.tf = CoordTransform(
            self.map_env.min_x, self.map_env.min_y,
            self.map_env.max_x, self.map_env.max_y,
            MAP_PADDING, MAP_PADDING,
            DRAW_W, DRAW_H,
        )

        self._pre_render_roads()

        # 扫描带 Surface（在超采样画布尺寸上创建）
        self._swath_surface = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
        self._swath_surface.fill((0, 0, 0, 0))

        # 新增实例化接管智能体 (已移除)
        self.random_agent = None
        self.tester = None

    def _decode_obs_to_state(self, obs):
        """
        将 PettingZoo / Wrapper 输出的归一化 obs 还原为 UI 需要的带有绝对物理坐标的字典结构。
        格式兼容旧版 Render 的 self.state
        """
        env_u = self.sim.unwrapped
        state_out = {
            't': env_u.t,
            'coverage_pct': np.sum(env_u.coverage_grid) / (env_u.grid_rows * env_u.grid_cols) * 100.0
        }
        
        # 解析 UGV
        if 'ugv_0' in obs:
            car = env_u.car
            car_x = car['x']
            car_y = car['y']
            
            mode = 'work' if car['mode'] == 1 else 'cruise'
            
            status = 'idle'
            if self.sim.current_swapping_uav is not None:
                status = 'swap_locked'
                mode = 'swap_wait'
            elif car['edge_u'] != car['edge_v'] and car['progress'] > 0:
                status = 'moving'
            
            current_node = car['edge_u'] if car['progress'] < 1e-3 else car['edge_v']
            
            state_out['car'] = {
                'x': car_x, 'y': car_y,
                'current_node': current_node,
                'mode': mode, 'status': status,
                'battery_stock': int(car.get('battery_stock', 10)),
                'swap_countdown': int(self.sim.swap_countdown) if self.sim.current_swapping_uav else 0
            }
        
        # 解析 UAVs
        for i, agent_id in enumerate(['uav_0', 'uav_1']):
            if agent_id in obs:
                uav = env_u.uavs[agent_id]
                ux = uav['x']
                uy = uav['y']
                
                mode = 'work' if uav['mode'] == 1 else 'cruise'
                
                status = 'moving' # 默认飞行中
                swap_cd = 0
                
                if self.sim.current_swapping_uav == agent_id:
                    status = 'swap_locked'
                    mode = 'swap'
                    swap_cd = int(self.sim.swap_countdown)
                elif agent_id in self.sim.rescue_queue:
                    # 判断是否已经降落休眠
                    if uav.get('battery_frozen', False) or (env_u.car and math.hypot(env_u.car['x'] - ux, env_u.car['y'] - uy) < 1.0):
                        status = 'idle'
                        mode = 'swap'
                
                battery = int(uav['battery'])
                
                state_out[f'uav_{i+1}'] = {
                    'x': ux, 'y': uy,
                    'mode': mode, 'status': status,
                    'battery': battery, 'swap_countdown': swap_cd
                }
            else:
                # 坠毁兜底
                state_out[f'uav_{i+1}'] = {'x': 0, 'y': 0, 'mode': 'offline', 'status': 'dead', 'battery': 0, 'swap_countdown': 0}
                
        return state_out


    def _pre_render_roads(self):
        """
        将路网绘制到超采样离屏 Surface。
        使用抗锯齿线条 + 更宽的线宽获得清晰路网。
        """
        surf = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))

        G = self.map_env.G_proj
        edges = list(G.edges())

        # 稀疏绘制（最多 3000 条边）
        step = max(1, len(edges) // 3000)
        lw = max(1, SS_SCALE)  # 超采样空间中线宽=2px → 缩小后≈1px 清晰
        for i, (u, v) in enumerate(edges):
            if i % step != 0:
                continue
            nd_u = G.nodes[u]
            nd_v = G.nodes[v]
            sx0, sy0 = self.tf.to_screen(float(nd_u['x']), float(nd_u['y']))
            sx1, sy1 = self.tf.to_screen(float(nd_v['x']), float(nd_v['y']))
            # 使用抗锯齿线
            pygame.draw.aaline(surf, C_ROAD, (sx0, sy0), (sx1, sy1))

        # 在路网上叠加节点圆点（仅采样最多 400 个节点）
        all_nodes = list(G.nodes())
        node_step = max(1, len(all_nodes) // 400)
        for i, n in enumerate(all_nodes):
            if i % node_step != 0:
                continue
            nd = G.nodes[n]
            sx, sy = self.tf.to_screen(float(nd['x']), float(nd['y']))
            r = max(1, SS_SCALE)
            pygame.gfxdraw.filled_circle(surf, sx, sy, r, C_ROAD_NODE)
            pygame.gfxdraw.aacircle(surf, sx, sy, r, C_ROAD_NODE)

        self._road_surface = surf

    # ── 主循环 ──────────────────────────────────────────────────────

    def run(self):
        """主事件循环。"""
        running = True
        while running:
            # =============== EDITOR 阶段处理 ===============
            if self.app_state == 'EDITOR':
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        # 鼠标左键，添加/选择节点
                        self.map_editor.handle_click(event.pos[0], event.pos[1])
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            self._setup_simulation_from_editor()
                        elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False

                if self.app_state == 'EDITOR': # 判断有没有刚刚通过ENTER跳出
                    self.screen.fill(C_BG)
                    self.map_editor.draw(self.screen)
                    pygame.display.flip()
                    self.clock.tick(30)
                    continue

            # =============== SIMULATION 阶段处理 ===============
            # ── 事件处理 ──
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)

            # ── 仿真推进 ──
            if not self.paused and not self.done and getattr(self, 'obs', None) is not None:
                actions = {}
                # 从 action_mask 构建安全的随机探索字典
                for agent in self.sim.agents:
                    act = self.sim.action_space(agent).sample()
                    if agent == 'ugv_0':
                        mask = self.obs[agent]['action_mask']
                        valid_actions = np.where(mask == 1)[0]
                        if valid_actions.size > 0:
                            act['move'] = int(np.random.choice(valid_actions))
                    actions[agent] = act
                    
                self.obs, rewards, terminations, truncations, infos = self.sim.step(actions)
                
                # 使用适配器还原为 UI 可用的物理坐标字典格式
                self.state = self._decode_obs_to_state(self.obs)
                
                # 兼容旧逻辑
                self.done = any(terminations.values())
                self.reward_acc += sum(rewards.values()) if rewards else 0.0
                
                self.tick_cnt += 1
                self._update_trails()
                self._update_swath_surface()

            # ── 绘制 ──
            self._draw_frame()
            # 将超采样画布 smoothscale 到真实屏幕（高质量缩放=抗锯齿效果）
            actual_w, actual_h = self.screen.get_size()
            scaled = pygame.transform.smoothscale(self._canvas, (actual_w, actual_h))
            self.screen.blit(scaled, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()
        sys.exit()

    def _handle_key(self, key) -> bool:
        """处理按键，返回 False 表示退出。"""
        if key in (pygame.K_q, pygame.K_ESCAPE):
            return False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_r:
            self._do_reset()
            self.paused = True
        elif key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
            self.fps = min(60, self.fps + 2)
        elif key in (pygame.K_DOWN, pygame.K_MINUS):
            self.fps = max(1, self.fps - 2)
        elif key == pygame.K_d:
            self._print_diagnostic()
        return True

    def _print_diagnostic(self):
        """按 D 键时，向终端打印完整诊断快照，方便验证仿真逻辑正确性。"""
        s = self.state
        if s is None:
            print("[Diagnostic] 尚无状态（仿真未启动）")
            return
        c  = s['car']
        u1 = s['uav_1']
        u2 = s['uav_2']
        agent_name = "ScriptedTester"
        agent_detail = ""
        if getattr(self, 'tester', None) is None or (
            hasattr(self.tester, 'phase') and self.tester.phase >= 4
        ):
            agent_name = "RandomPatrolAgent"
            agent_detail = f"swap_threshold={RandomPatrolAgent.SWAP_THRESHOLD}"
        else:
            agent_detail = f"phase={self.tester.phase}"

        sep = "─" * 60
        print(f"\n{sep}")
        print(f"[Diagnostic Snapshot]  t={s['t']}  agent={agent_name}({agent_detail})")
        print(f"  Coverage : {s['coverage_pct']:.4f}%   累计奖励={self.reward_acc:+.3f}")
        print(f"  CAR  | ({c['x']:.1f}, {c['y']:.1f})  node={c['current_node']}  "
              f"status={c['status']}  mode={c['mode']}  "
              f"battery_stock={c['battery_stock']}  swap_cd={c['swap_countdown']}")
        print(f"  UAV1 | ({u1['x']:.1f}, {u1['y']:.1f})  "
              f"battery={u1['battery']}/900  status={u1['status']}  "
              f"mode={u1['mode']}  swap_cd={u1['swap_countdown']}")
        print(f"  UAV2 | ({u2['x']:.1f}, {u2['y']:.1f})  "
              f"battery={u2['battery']}/900  status={u2['status']}  "
              f"mode={u2['mode']}  swap_cd={u2['swap_countdown']}")
        print(sep)

    def _update_trails(self):
        """记录无人机坐标轨迹（保留最近 80 帧）。"""
        MAX_TRAIL = 80
        for i, uav_key in enumerate(['uav_1', 'uav_2']):
            u = self.state[uav_key]
            sx, sy = self.tf.to_screen(u['x'], u['y'])
            self._uav_trails[i].append((sx, sy))
            if len(self._uav_trails[i]) > MAX_TRAIL:
                self._uav_trails[i].pop(0)

    def _update_swath_surface(self):
        """
        将当前帧的扫描带半透明矩形绘制到 _swath_surface 上（只追加，不清除历史）。
        当 coverage_pct 超过一定阈值时仍然保留历史，体现地毯式扫描效果。
        """
        for i, uav_key in enumerate(['uav_1', 'uav_2']):
            u = self.state[uav_key]
            if u['status'] not in ('work', 'moving') or u['mode'] != 'work':
                continue
            if len(self._uav_trails[i]) < 2:
                continue
            # 使用前一帧和当前帧坐标绘制扫描带矩形
            sx0, sy0 = self._uav_trails[i][-2]
            sx1, sy1 = self._uav_trails[i][-1]
            self._draw_swath_rect(sx0, sy0, sx1, sy1, i)

    def _draw_swath_rect(self, sx0, sy0, sx1, sy1, uav_idx: int):
        """在 _swath_surface 上绘制一条扫描矩形（半透明叠层）。"""
        dx = sx1 - sx0
        dy = sy1 - sy0
        length = math.hypot(dx, dy)
        if length < 0.5:
            return

        # 扫描带半宽（像素）
        half_w_px = self.tf.length_to_px(25.0)  # 50m / 2

        # 法向量
        nx = -dy / length
        ny =  dx / length

        # 四顶点
        verts = [
            (int(sx0 + nx * half_w_px), int(sy0 + ny * half_w_px)),
            (int(sx1 + nx * half_w_px), int(sy1 + ny * half_w_px)),
            (int(sx1 - nx * half_w_px), int(sy1 - ny * half_w_px)),
            (int(sx0 - nx * half_w_px), int(sy0 - ny * half_w_px)),
        ]
        color = (C_UAV1[0], C_UAV1[1], C_UAV1[2], 55) if uav_idx == 0 \
                else (C_UAV2[0], C_UAV2[1], C_UAV2[2], 55)
        pygame.gfxdraw.filled_polygon(self._swath_surface, verts, color)

    # ── 绘制帧 ──────────────────────────────────────────────────────

    @property
    def _surf(self):
        """绘制目标：超采样离屏画布。"""
        return self._canvas

    def _draw_frame(self):
        """每帧的全部绘制调用序列（在超采样画布上进行）。"""
        C = self._canvas   # 绘制目标：超采样离屏画布
        S = SS_SCALE
        C.fill(C_BG)

        # 1. 地图底色
        pygame.draw.rect(C, C_MAP_BG, (0, 0, MAP_W, MAP_H))

        # 2. 覆盖率栅格
        self._draw_coverage_grid(C)

        # 3. 路网（预渲染层）
        C.blit(self._road_surface, (0, 0))

        # 4. 无人机扫描带历史
        C.blit(self._swath_surface, (0, 0))

        # 5. 无人机轨迹线
        self._draw_trails(C, S)

        # 6. 车辆采集圆
        self._draw_car_aura(C, S)

        # 7. 实体图标
        self._draw_entities(C, S)

        # 8. 地图边框（损色分隔线）
        pygame.draw.rect(C, C_HUD_LINE, (0, 0, MAP_W, MAP_H), S * 1)

        # 9. HUD 面板
        self._draw_hud(C, S)

        # 10. 覆盖线（分隔地图与HUD）
        pygame.draw.line(C, C_HUD_LINE, (HUD_X, 0), (HUD_X, CANVAS_H), S * 2)

        # 11. 暂停覆盖提示
        if self.paused:
            self._draw_pause_overlay(C, S)

    def _draw_coverage_grid(self, C):
        """将 coverage_grid 转换为彩色 Surface 绘制到地图区域。"""
        grid = self.sim.coverage_grid
        rows, cols = grid.shape

        cell_px_w = max(1, int(DRAW_W / cols))
        cell_px_h = max(1, int(DRAW_H / rows))

        cov_rows, cov_cols = np.where(grid == 1)
        for r, c_idx in zip(cov_rows, cov_cols):
            phys_x = self.map_env.min_x + (c_idx + 0.5) * self.map_env.grid_resolution
            phys_y = self.map_env.min_y + (r + 0.5) * self.map_env.grid_resolution
            sx, sy = self.tf.to_screen(phys_x, phys_y)
            rect = pygame.Rect(
                sx - cell_px_w // 2,
                sy - cell_px_h // 2,
                cell_px_w, cell_px_h
            )
            pygame.draw.rect(C, C_GRID_COVERED, rect)

    def _draw_car_aura(self, C, S):
        """绘制车辆采集范围圆（半透明蓝色光晕）。"""
        if self.state is None:
            return
        c = self.state['car']
        sx, sy = self.tf.to_screen(c['x'], c['y'])
        radius_px = self.tf.length_to_px(10.0)

        aura_surf = pygame.Surface((radius_px * 2 + 2, radius_px * 2 + 2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(
            aura_surf, radius_px + 1, radius_px + 1, radius_px, C_CAR_AURA
        )
        C.blit(aura_surf, (sx - radius_px - 1, sy - radius_px - 1))

    def _draw_trails(self, C, S):
        """绘制无人机历史轨迹（渐隐效果，抗锯齿线）。"""
        colors = [C_UAV1, C_UAV2]
        for i, trail in enumerate(self._uav_trails):
            if len(trail) < 2:
                continue
            n = len(trail)
            cl = colors[i]
            for j in range(1, n):
                # 越近越亮
                width = S if j < n // 2 else S * 2
                pygame.draw.aaline(C, cl, trail[j - 1], trail[j])

    def _draw_entities(self, C, S):
        """绘制车辆和两架无人机的图标、标签、状态环。"""
        if self.state is None:
            return

        # ── 车辆 ──
        c = self.state['car']
        csx, csy = self.tf.to_screen(c['x'], c['y'])
        car_color = C_SWAP_LOCK if c['status'] == 'swap_locked' else C_CAR
        car_size = 14 * S
        pygame.draw.rect(
            C, car_color,
            (csx - car_size, csy - car_size, car_size * 2, car_size * 2),
            border_radius=S * 3
        )
        # 十字准星
        pygame.draw.aaline(C, C_MAP_BG, (csx - car_size, csy), (csx + car_size, csy))
        pygame.draw.aaline(C, C_MAP_BG, (csx, csy - car_size), (csx, csy + car_size))
        # 轮廓
        pygame.draw.rect(
            C, C_TEXT_MAIN,
            (csx - car_size, csy - car_size, car_size * 2, car_size * 2),
            S, border_radius=S * 3
        )
        # 标签
        label = self.font_sm.render('CAR', True, C_TEXT_MAIN)
        C.blit(label, (csx + car_size + S * 3, csy - label.get_height() // 2))

        # 换电倒计时環
        if c['status'] == 'swap_locked' and c['swap_countdown'] > 0:
            ratio = 1.0 - c['swap_countdown'] / 120.0
            self._draw_arc_ring(C, csx, csy, car_size + S * 6, ratio, C_SWAP_LOCK, S)

        # ── 无人机 ──
        uav_defs = [
            ('uav_1', C_UAV1, 'U1'),
            ('uav_2', C_UAV2, 'U2'),
        ]
        for key, color, tag in uav_defs:
            u = self.state[key]
            usx, usy = self.tf.to_screen(u['x'], u['y'])
            uav_r = 12 * S

            is_swapping_with_car = (
                u['status'] == 'swap_locked' and
                c['status'] == 'swap_locked' and
                math.hypot(u['x'] - c['x'], u['y'] - c['y']) < 3.0
            )

            if is_swapping_with_car:
                pulse_w = max(S, int(S * 3 * (u['swap_countdown'] % 10) / 10))
                pygame.draw.line(C, (255, 255, 100), (csx, csy), (usx, usy), pulse_w)
                bar_w = 100 * S
                bar_h = 10 * S
                bar_x = (csx + usx) // 2 - bar_w // 2
                bar_y = min(csy, usy) - 30 * S
                ratio = 1.0 - u['swap_countdown'] / 120.0
                pygame.draw.rect(C, C_BAR_BG, (bar_x, bar_y, bar_w, bar_h), border_radius=S * 5)
                if ratio > 0:
                    pygame.draw.rect(C, (255, 215, 0), (bar_x, bar_y, int(bar_w * ratio), bar_h), border_radius=S * 5)
                pygame.draw.rect(C, (255, 255, 255), (bar_x, bar_y, bar_w, bar_h), S, border_radius=S * 5)
                if (u['swap_countdown'] // 5) % 2 == 0:
                    ic_x, ic_y = bar_x - S * 4, bar_y - S * 20
                    pygame.draw.rect(C, (255, 215, 0), (ic_x, ic_y, S * 10, S * 7), border_radius=S)
                    pygame.draw.rect(C, (255, 215, 0), (ic_x + S * 10, ic_y + S * 2, S * 2, S * 3))
                    lbl = self.font_sm.render(" SWAPPING...", True, (255, 215, 0))
                    C.blit(lbl, (ic_x + S * 13, ic_y - S))

            draw_color = C_SWAP_LOCK if u['status'] == 'swap_locked' else color

            # 无人机圆形图标（抗锯齿）
            pygame.gfxdraw.filled_circle(C, usx, usy, uav_r, draw_color)
            pygame.gfxdraw.aacircle(C, usx, usy, uav_r, C_TEXT_MAIN)

            # 内部十字
            pygame.draw.aaline(C, C_MAP_BG, (usx - uav_r, usy), (usx + uav_r, usy))
            pygame.draw.aaline(C, C_MAP_BG, (usx, usy - uav_r), (usx, usy + uav_r))

            # 标签
            label = self.font_sm.render(tag, True, C_TEXT_MAIN)
            C.blit(label, (usx + uav_r + S * 3, usy - label.get_height() // 2))

            if not is_swapping_with_car:
                self._draw_mini_battery(C, usx - 18 * S, usy - uav_r - S * 12, 36 * S, S * 6,
                                        u['battery'], 900, color)
                if u['status'] == 'swap_locked' and u['swap_countdown'] > 0:
                    ratio = 1.0 - u['swap_countdown'] / 120.0
                    self._draw_arc_ring(C, usx, usy, uav_r + S * 6, ratio, C_SWAP_LOCK, S)

    def _draw_mini_battery(self, C, x, y, w, h, val, max_val, color):
        """在实体上方绘制一条迷你电量条。"""
        S = SS_SCALE
        ratio = max(0.0, val / max_val)
        bar_color = C_BAR_GREEN if ratio > 0.4 else (C_BAR_ORANGE if ratio > 0.15 else C_BAR_RED)
        pygame.draw.rect(C, C_BAR_BG, (x, y, w, h), border_radius=S * 2)
        if ratio > 0:
            pygame.draw.rect(C, bar_color, (x, y, int(w * ratio), h), border_radius=S * 2)
        pygame.draw.rect(C, C_TEXT_DIM, (x, y, w, h), S, border_radius=S * 2)

    def _draw_arc_ring(self, C, cx, cy, radius, ratio, color, S=1):
        """绘制倒计时弧形环（0→1 表示从 0° 到 360°），抗锯齿。"""
        if ratio <= 0:
            return
        steps = max(4, int(ratio * 60))
        start_angle = -math.pi / 2
        end_angle   = start_angle + 2 * math.pi * ratio
        points = []
        for i in range(steps + 1):
            angle = start_angle + (end_angle - start_angle) * i / steps
            px = cx + int(radius * math.cos(angle))
            py = cy + int(radius * math.sin(angle))
            points.append((px, py))
        if len(points) >= 2:
            pygame.draw.lines(C, color, False, points, S * 2)

    # ── HUD 面板 ─────────────────────────────────────────────────────

    def _draw_status_pill(self, C, x, y, text, color, bg):
        """绘制一个圆角状态胶囊标签，返回占用宽度。"""
        surf = self.font_sm_bold.render(text, True, color)
        w, h = surf.get_width() + 14, surf.get_height() + 6
        pygame.draw.rect(C, bg, (x, y, w, h), border_radius=h // 2)
        pygame.draw.rect(C, color, (x, y, w, h), 1, border_radius=h // 2)
        C.blit(surf, (x + 7, y + 3))
        return w

    def _draw_hud(self, C, S):
        """绘制右侧信息面板。"""
        pygame.draw.rect(C, C_HUD_BG, (HUD_X, 0, HUD_W, CANVAS_H))

        pad = 18 * S
        x0  = HUD_X + pad
        cw  = HUD_W - pad * 2
        y   = 16 * S

        # 标题
        title_surf = self.font_xl.render("VRP-D  Simulator", True, C_TEXT_MAIN)
        C.blit(title_surf, (x0, y))
        y += title_surf.get_height() + S * 4
        pygame.draw.rect(C, C_TITLE_ACCENT, (x0, y, cw, S * 2), border_radius=S)
        y += S * 10

        fps_actual = self.clock.get_fps()

        if self.paused:
            pill_text, pill_fg, pill_bg = "II  PAUSED",  C_TEXT_WARN,  (60, 30, 10)
        elif self.done:
            pill_text, pill_fg, pill_bg = ">>  DONE",    C_BAR_GREEN,  (10, 45, 20)
        else:
            pill_text, pill_fg, pill_bg = ">   RUNNING", C_BAR_GREEN,  (10, 45, 20)
        self._draw_status_pill(C, x0, y, pill_text, pill_fg, pill_bg)
        y += S * 30

        info_pairs = [
            ("Time",   f"t = {self.state['t']} s"),
            ("FPS",    f"{fps_actual:.1f}  /  {self.fps} target"),
            ("Reward", f"{self.reward_acc:+.3f}"),
        ]
        for lbl, val in info_pairs:
            lbl_surf = self.font_sm.render(lbl, True, C_TEXT_DIM)
            val_surf = self.font_md_bold.render(val, True, C_TEXT_MAIN)
            C.blit(lbl_surf, (x0, y))
            C.blit(val_surf, (x0 + cw - val_surf.get_width(), y))
            y += max(lbl_surf.get_height(), val_surf.get_height()) + S * 5

        agent_label = "Random Patrol"
        agent_fg, agent_bg = C_BAR_GREEN, (10, 45, 20)
        
        a_lbl = self.font_sm.render("Strategy", True, C_TEXT_DIM)
        C.blit(a_lbl, (x0, y))
        agent_surf = self.font_sm_bold.render(agent_label, True, agent_fg)
        C.blit(agent_surf, (x0 + cw - agent_surf.get_width(), y))
        y += agent_surf.get_height() + S * 8
        
        # 救援队列显示
        q_lbl = self.font_sm.render("Rescue Queue", True, C_TEXT_DIM)
        C.blit(q_lbl, (x0, y))
        queue_text = "[" + ", ".join(self.sim.rescue_queue) + "]" if self.sim.rescue_queue else "None"
        q_fg = C_BAR_ORANGE if self.sim.rescue_queue else C_TEXT_DIM
        q_surf = self.font_sm_bold.render(queue_text, True, q_fg)
        
        C.blit(q_surf, (x0 + cw - q_surf.get_width(), y))
        y += q_surf.get_height() + S * 8

        pygame.draw.line(C, C_HUD_LINE, (HUD_X + S * 8, y), (CANVAS_W - S * 8, y), S)
        y += S * 10

        # 覆盖率
        pct = self.state['coverage_pct']
        cov_lbl = self.font_sm.render("Coverage", True, C_TEXT_DIM)
        cov_pct = self.font_lg.render(f"{pct:.2f}%", True, C_TEXT_MAIN)
        C.blit(cov_lbl, (x0, y))
        C.blit(cov_pct, (x0 + cw - cov_pct.get_width(), y))
        y += cov_pct.get_height() + S * 4

        bar_h = S * 16
        pygame.draw.rect(C, C_BAR_BG, (x0, y, cw, bar_h), border_radius=bar_h // 2)
        fill_w = int(cw * min(1.0, pct / 100.0))
        if fill_w > 0:
            fill_color = C_BAR_RED if pct < 20 else (C_BAR_ORANGE if pct < 60 else C_BAR_GREEN)
            pygame.draw.rect(C, fill_color, (x0, y, fill_w, bar_h), border_radius=bar_h // 2)
        pygame.draw.rect(C, C_HUD_LINE, (x0, y, cw, bar_h), S, border_radius=bar_h // 2)
        y += bar_h + S * 10

        pygame.draw.line(C, C_HUD_LINE, (HUD_X + S * 8, y), (CANVAS_W - S * 8, y), S)
        y += S * 10

        # 实体面板
        c  = self.state['car']
        u1 = self.state['uav_1']
        u2 = self.state['uav_2']

        y = self._draw_entity_panel(C, S, y, x0, cw,
            title="CAR  —  监测车",
            icon_type='car', accent=C_CAR, status=c['status'],
            rows=[
                ("Position", f"({c['x']:.1f}, {c['y']:.1f}) m"),
                ("Node",     str(c['current_node'])),
                ("Mode",     c['mode']),
                ("Battery",  str(c['battery_stock']) + " units"),
                ("Swap CD",  f"{c['swap_countdown']} s" if c['swap_countdown'] > 0 else "none"),
            ],
        )
        y += S * 4
        pygame.draw.line(C, C_HUD_LINE, (HUD_X + S * 8, y), (CANVAS_W - S * 8, y), S)
        y += S * 10

        y = self._draw_entity_panel(C, S, y, x0, cw,
            title="UAV-1  —  无人机 1",
            icon_type='uav', accent=C_UAV1, status=u1['status'],
            rows=[
                ("Position", f"({u1['x']:.1f}, {u1['y']:.1f}) m"),
                ("Mode",     u1['mode']),
                ("Swap CD",  f"{u1['swap_countdown']} s" if u1['swap_countdown'] > 0 else "none"),
            ],
            battery_val=u1['battery'],
        )
        y += S * 4
        pygame.draw.line(C, C_HUD_LINE, (HUD_X + S * 8, y), (CANVAS_W - S * 8, y), S)
        y += S * 10

        y = self._draw_entity_panel(C, S, y, x0, cw,
            title="UAV-2  —  无人机 2",
            icon_type='uav', accent=C_UAV2, status=u2['status'],
            rows=[
                ("Position", f"({u2['x']:.1f}, {u2['y']:.1f}) m"),
                ("Mode",     u2['mode']),
                ("Swap CD",  f"{u2['swap_countdown']} s" if u2['swap_countdown'] > 0 else "none"),
            ],
            battery_val=u2['battery'],
        )
        y += S * 4
        pygame.draw.line(C, C_HUD_LINE, (HUD_X + S * 8, y), (CANVAS_W - S * 8, y), S)
        y += S * 10

        # 控制说明
        y = max(y, CANVAS_H - S * 148)
        ctrl_surfs = [
            ("SPACE", "暂停/继续"),
            ("R",     "重置"),
            ("UP/DN", "调速"),
            ("D",     "诊断"),
            ("Q/ESC", "退出"),
        ]
        key_color = (200, 195, 100)
        col_w = cw // 2 + S * 4
        for i, (key, desc) in enumerate(ctrl_surfs):
            cx_off = x0 + (i % 2) * col_w
            cy_off = y + (i // 2) * S * 24
            k_s = self.font_sm_bold.render(f"[{key}]", True, key_color)
            d_s = self.font_xs.render(desc, True, C_TEXT_DIM)
            C.blit(k_s, (cx_off, cy_off))
            C.blit(d_s, (cx_off + k_s.get_width() + S * 4, cy_off + S * 2))

    def _draw_entity_panel(
        self, C, S, y: int, x0: int, cw: int,
        title: str, rows: list, accent,
        status: str = '', battery_val: int = -1, icon_type: str = '',
    ) -> int:
        """绘制一个实体信息卡片面板，返回结束 Y 坐标。"""
        card_h_est = S * 28 + (S * 12 if battery_val >= 0 else 0) + len(rows) * S * 21 + S * 8
        pygame.draw.rect(C, C_HUD_CARD,
                         (x0 - S * 6, y - S * 2, cw + S * 12, card_h_est), border_radius=S * 6)
        pygame.draw.rect(C, accent,
                         (x0 - S * 6, y - S * 2, S * 4, card_h_est), border_radius=S * 3)

        ix = x0 + S * 4
        if icon_type == 'car':
            pygame.draw.rect(C, accent, (ix, y + S * 2, S * 16, S * 9), border_radius=S * 2)
            pygame.draw.rect(C, accent, (ix + S * 3, y - S * 1, S * 10, S * 6), border_radius=S * 2)
            pygame.gfxdraw.filled_circle(C, ix + S * 3,  y + S * 12, S * 3, accent)
            pygame.gfxdraw.filled_circle(C, ix + S * 13, y + S * 12, S * 3, accent)
            ix += S * 22
        elif icon_type == 'uav':
            cx_i, cy_i = ix + S * 8, y + S * 7
            pygame.draw.circle(C, accent, (cx_i, cy_i), S * 3)
            for angle_deg in (45, 135, 225, 315):
                angle_rad = math.radians(angle_deg)
                ex = cx_i + int(S * 7 * math.cos(angle_rad))
                ey = cy_i + int(S * 7 * math.sin(angle_rad))
                pygame.draw.aaline(C, accent, (cx_i, cy_i), (ex, ey))
                pygame.draw.circle(C, accent, (ex, ey), S * 3)
            ix += S * 22

        title_surf = self.font_md_bold.render(title, True, accent)
        C.blit(title_surf, (ix, y))

        if status:
            is_warn = status in ('swap_locked',)
            s_fg = C_SWAP_LOCK if is_warn else C_TEXT_DIM
            s_bg = (45, 10, 15) if is_warn else (30, 36, 55)
            s_surf = self.font_xs.render(status, True, s_fg)
            sw = s_surf.get_width() + S * 10
            sh = s_surf.get_height() + S * 4
            sx = x0 + cw - sw
            pygame.draw.rect(C, s_bg, (sx, y + S, sw, sh), border_radius=sh // 2)
            pygame.draw.rect(C, s_fg, (sx, y + S, sw, sh), S, border_radius=sh // 2)
            C.blit(s_surf, (sx + S * 5, y + S * 3))

        y += title_surf.get_height() + S * 6

        if battery_val >= 0:
            ratio = battery_val / 900.0
            bar_color = C_BAR_GREEN if ratio > 0.4 else (C_BAR_ORANGE if ratio > 0.15 else C_BAR_RED)
            bh = S * 8
            pygame.draw.rect(C, C_BAR_BG, (x0, y, cw, bh), border_radius=bh // 2)
            if ratio > 0:
                pygame.draw.rect(C, bar_color, (x0, y, int(cw * ratio), bh), border_radius=bh // 2)
            pygame.draw.rect(C, C_HUD_LINE, (x0, y, cw, bh), S, border_radius=bh // 2)
            bv_s = self.font_xs.render(f"{battery_val}/900", True, bar_color)
            C.blit(bv_s, (x0 + cw - bv_s.get_width(), y + bh + S))
            y += bh + S * 12

        for key, val in rows:
            k_s = self.font_xs.render(key, True, C_TEXT_DIM)
            v_s = self.font_sm.render(str(val), True, C_TEXT_MAIN)
            C.blit(k_s, (x0 + S * 2, y + S * 2))
            C.blit(v_s, (x0 + cw - v_s.get_width(), y))
            y += max(k_s.get_height(), v_s.get_height()) + S * 3

        y += S * 4
        return y

    def _draw_pause_overlay(self, C, S):
        """在地图中央绘制半透明暂停提示。"""
        overlay = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 110))
        C.blit(overlay, (0, 0))

        box_w, box_h = S * 420, S * 100
        bx = (MAP_W - box_w) // 2
        by = (MAP_H - box_h) // 2
        pygame.draw.rect(C, (20, 26, 44), (bx, by, box_w, box_h), border_radius=S * 14)
        pygame.draw.rect(C, C_TITLE_ACCENT, (bx, by, box_w, box_h), S * 2, border_radius=S * 14)

        # 暂停双竖线图标
        ic_x, ic_y = bx + S * 30, by + S * 22
        pygame.draw.rect(C, C_TEXT_MAIN, (ic_x,        ic_y, S * 12, S * 56), border_radius=S * 3)
        pygame.draw.rect(C, C_TEXT_MAIN, (ic_x + S*20, ic_y, S * 12, S * 56), border_radius=S * 3)

        t1 = self.font_xl.render("PAUSED", True, C_TEXT_MAIN)
        t2 = self.font_sm.render("按  SPACE  键继续仿真", True, C_TEXT_DIM)
        C.blit(t1, (bx + S * 76, by + S * 20))
        C.blit(t2, (bx + S * 76, by + S * 20 + t1.get_height() + S * 8))


# ──────────────────────────────────────────────
#  入口
# ──────────────────────────────────────────────

# if __name__ == '__main__':
#     print("=" * 65)
#     print("  多智能体路径规划模拟器  — VRP-D Visualizer")
#     print("=" * 65)
#     print("已启动手绘地图编辑器模式 (Map Editor)")
#     print("操作提示：")
#     print("  1. 在深蓝色界面中，使用鼠标左键点击以放置节点。")
#     print("  2. 点击已放置的节点予以选中（高亮），再点击另一节点即可连线。")
#     print("  3. 绘制好所需的路网后，按下 [ENTER] 键开始仿真。")
#     print("=" * 65)
#
#     vis = SimulationVisualizer(map_env=None)
#     vis.run()

if __name__ == '__main__':
    print("=" * 65)
    print("  多智能体路径规划模拟器  — VRP-D Visualizer")
    print("=" * 65)

    # 【核心修改】：导入我们的随机地图生成器
    from env_defs import RandomMapEnv

    print("正在后台生成随机路网...")
    # 实例化一张 200x200 的随机地图
    random_map = RandomMapEnv(grid_resolution=10.0)

    # 直接把随机地图塞给可视化器！(这会直接跳过 Editor 阶段)
    vis = SimulationVisualizer(map_env=random_map)
    vis.run()