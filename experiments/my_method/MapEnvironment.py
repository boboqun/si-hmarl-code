import os
import tkinter as tk
import webbrowser
import urllib3
import numpy as np
import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import tkintermapview

# ================= OSMnx 缓存与网络配置 =================
# 屏蔽 SSL 证书警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# 开启本地缓存（断点续传），极大提高重复测试的速度
ox.settings.use_cache = True
# 开启控制台日志，查看下载进度
ox.settings.log_console = True
# 延长超时时间至 200 秒
ox.settings.timeout = 200
# 绕过代理导致的 SSL 校验错误
ox.settings.requests_kwargs = {'verify': False}


# =========================================================

def interactive_select_bbox(default_lat=0.0, default_lon=0.0):
    """
    弹出一个交互式地图窗口，让用户手动框选测试区域。
    支持在 谷歌混合卫星图 和 OpenStreetMap 之间一键切换。
    返回符合 OSMnx 2.0+ 标准的 bbox: (west, south, east, north)
    """
    print("正在启动交互式地图选择器...")
    root = tk.Tk()
    root.title("请框选模拟区域 (右键点击两次设置对角点)")
    root.geometry("900x750")  # 稍微调高一点窗口以容纳新按钮

    # 创建地图组件
    map_widget = tkintermapview.TkinterMapView(root, width=900, height=550, corner_radius=0)
    map_widget.pack(fill="both", expand=True)

    # 默认坐标为占位值，请替换为目标作业区域；缩放级别 15
    map_widget.set_position(default_lat, default_lon)
    map_widget.set_zoom(15)

    # 瓦片服务器 URL 定义
    GOOGLE_TILE = "https://mt0.google.com/vt/lyrs=y&hl=zh-CN&x={x}&y={y}&z={z}&s=Ga"
    OSM_TILE = "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"

    # 初始默认使用谷歌混合图
    current_layer = "google"
    map_widget.set_tile_server(GOOGLE_TILE, max_zoom=22)

    markers = []
    polygon = None
    selected_bbox = []

    def add_marker(coords):
        nonlocal polygon
        if len(markers) >= 2:
            for m in markers:
                m.delete()
            markers.clear()
            if polygon:
                polygon.delete()
                polygon = None

        marker = map_widget.set_marker(coords[0], coords[1], text=f"顶点 {len(markers) + 1}")
        markers.append(marker)

        if len(markers) == 2:
            lat1, lon1 = markers[0].position
            lat2, lon2 = markers[1].position

            top_left = (max(lat1, lat2), min(lon1, lon2))
            top_right = (max(lat1, lat2), max(lon1, lon2))
            bottom_right = (min(lat1, lat2), max(lon1, lon2))
            bottom_left = (min(lat1, lat2), min(lon1, lon2))

            polygon = map_widget.set_polygon(
                [top_left, top_right, bottom_right, bottom_left],
                fill_color="red", outline_color="red", border_width=2
            )

    map_widget.add_right_click_menu_command(label="在此处设置矩形顶点", command=add_marker, pass_coords=True)

    def confirm_selection():
        if len(markers) == 2:
            lat1, lon1 = markers[0].position
            lat2, lon2 = markers[1].position

            north = max(lat1, lat2)
            south = min(lat1, lat2)
            east = max(lon1, lon2)
            west = min(lon1, lon2)

            selected_bbox.extend([west, south, east, north])
            root.destroy()
        else:
            print("提示: 请先使用鼠标右键在地图上选定两个对角点！")

    def toggle_map_layer():
        """切换底图类型的函数"""
        nonlocal current_layer
        if current_layer == "google":
            # 切换到 OSM
            map_widget.set_tile_server(OSM_TILE, max_zoom=19)
            btn_toggle.config(text="当前底图: OpenStreetMap (点击切换回 谷歌卫星图)", bg="#e0e0e0")
            current_layer = "osm"
        else:
            # 切换回 Google
            map_widget.set_tile_server(GOOGLE_TILE, max_zoom=22)
            btn_toggle.config(text="当前底图: 谷歌混合卫星图 (点击切换至 OpenStreetMap 查路网)", bg="#fffacd")
            current_layer = "google"

    # ================= 底部控制面板 =================
    control_frame = tk.Frame(root)
    control_frame.pack(pady=10)

    lbl_hint = tk.Label(control_frame, text="操作说明：鼠标左键拖动地图，【鼠标右键】点击两次设定测试区的两个对角点。",
                        fg="gray", font=("Arial", 11))
    lbl_hint.pack(pady=2)

    # 新增：切换底图按钮
    btn_toggle = tk.Button(control_frame, text="当前底图: 谷歌混合卫星图 (点击切换至 OpenStreetMap 查路网)",
                           command=toggle_map_layer, font=("Arial", 10), bg="#fffacd")
    btn_toggle.pack(pady=5)

    btn_confirm = tk.Button(control_frame, text="确认框选并开始生成环境", command=confirm_selection, font=("Arial", 14),
                            bg="#90EE90")
    btn_confirm.pack(pady=5)

    root.mainloop()

    if len(selected_bbox) == 4:
        return tuple(selected_bbox)
    else:
        return None

class MapEnvironment:
    """
    地图与环境预处理模块
    用于多智能体协同路径规划 2D 模拟器
    负责真实地图下载、坐标系投影、路网拓扑提取以及覆盖率栅格的初始化
    """

    def __init__(self, bbox, grid_resolution=1.0, padding=50.0):
        self.bbox = bbox
        self.grid_resolution = grid_resolution
        self.padding = padding

        self.G_proj = None
        self.nodes = None
        self.edges = None
        self.task_polygon = None
        self.coverage_grid = None

        self.min_x = 0.0
        self.max_x = 0.0
        self.min_y = 0.0
        self.max_y = 0.0
        self.rows = 0
        self.cols = 0

        print("\n正在构建模拟环境，这可能需要一点时间下载地图数据...")
        self._fetch_and_project_network()
        self._create_simulation_bounds()
        self._initialize_coverage_grid()
        print("环境构建完成！")

    def _fetch_and_project_network(self):
        """
        下载 OSM 路网并投影到局部平面直角坐标系 (UTM)
        """
        # 严格匹配 OSMnx 2.0+ 的坐标顺序：左、下、右、上
        west, south, east, north = self.bbox

        # 1. 下载原始路网 (使用 'all' 提取田间小道和机耕路)
        G_directed = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='all')

        # 2. 转换为无向图 (兼容最新版 API，彻底抹除单行道概念)
        try:
            G_undirected = ox.convert.to_undirected(G_directed)
        except AttributeError:
            G_undirected = G_directed.to_undirected()

        # 3. 投影到局部 UTM 坐标系 (单位变为米)
        self.G_proj = ox.project_graph(G_undirected)

        # 将拓扑图转换为 GeoDataFrame
        self.nodes, self.edges = ox.graph_to_gdfs(self.G_proj)

        print(f"路网提取成功(已转换为双向无向图)：共 {len(self.nodes)} 个节点，{len(self.edges)} 条边。")

    def _create_simulation_bounds(self):
        """
        基于投影后的路网建立仿真空间的物理边界，并生成任务多边形
        """
        bounds = self.nodes.total_bounds
        self.min_x = bounds[0] - self.padding
        self.min_y = bounds[1] - self.padding
        self.max_x = bounds[2] + self.padding
        self.max_y = bounds[3] + self.padding

        self.task_polygon = Polygon([
            (self.min_x, self.min_y),
            (self.min_x, self.max_y),
            (self.max_x, self.max_y),
            (self.max_x, self.min_y)
        ])

        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        print(f"仿真物理空间建立：宽 {width:.2f} 米, 高 {height:.2f} 米 (包含 {self.padding}m 缓冲)。")

    def _initialize_coverage_grid(self):
        """
        初始化用于覆盖率计算的 2D 离散栅格矩阵
        """
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        self.cols = int(np.ceil(width / self.grid_resolution))
        self.rows = int(np.ceil(height / self.grid_resolution))
        self.coverage_grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        print(f"覆盖率栅格初始化：矩阵形状为 ({self.rows}, {self.cols})，分辨率 {self.grid_resolution}m/格。")

    def xy_to_grid(self, x, y):
        col = int((x - self.min_x) // self.grid_resolution)
        row = int((y - self.min_y) // self.grid_resolution)
        col = np.clip(col, 0, self.cols - 1)
        row = np.clip(row, 0, self.rows - 1)
        return row, col

    def get_closest_node(self, x, y):
        node_id = ox.distance.nearest_nodes(self.G_proj, X=x, Y=y)
        return node_id

    def export_web_map(self, filename="simulation_map.html"):
        """
        生成交互式的 HTML Web 地图，用于在浏览器中滑动验证真实地理位置
        """
        try:
            import folium
        except ImportError:
            print("缺少 folium 库，无法导出 Web 地图。")
            return

        print(f"\n正在生成交互式 Web 地图 ({filename})...")

        # 将 UTM 直角坐标系转换为 EPSG:4326 (标准的 WGS84 经纬度)，适配 OSMnx 2.0+
        G_wgs84 = ox.project_graph(self.G_proj, to_crs="EPSG:4326")
        nodes_wgs84, edges_wgs84 = ox.graph_to_gdfs(G_wgs84)

        center_lat = nodes_wgs84['y'].mean()
        center_lon = nodes_wgs84['x'].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=16)

        # 绘制提取出的路网
        folium.GeoJson(
            edges_wgs84,
            name="提取的田间路网 (All Network)",
            style_function=lambda x: {'color': '#3388ff', 'weight': 3, 'opacity': 0.8}
        ).add_to(m)

        # 将任务边界多边形转回经纬度并绘制
        gdf_bounds = gpd.GeoDataFrame(geometry=[self.task_polygon], crs=self.G_proj.graph['crs'])
        gdf_bounds_wgs84 = gdf_bounds.to_crs(epsg=4326)
        task_poly_wgs84 = gdf_bounds_wgs84.geometry.iloc[0]

        lons, lats = task_poly_wgs84.exterior.xy
        boundary_coords = list(zip(lats, lons))

        folium.PolyLine(
            locations=boundary_coords,
            color='red',
            weight=4,
            dash_array='10, 10',
            tooltip="仿真物理边界 (包含 Padding)",
            name="仿真物理边界"
        ).add_to(m)

        folium.LayerControl().add_to(m)

        filepath = os.path.abspath(filename)
        m.save(filepath)
        print(f"Web 地图生成完毕！正在浏览器中打开: {filepath}")
        webbrowser.open('file://' + filepath)


if __name__ == "__main__":
    # 1. 弹出可视化窗口，让用户手动框选田间地块
    test_bbox = interactive_select_bbox(default_lat=0.0, default_lon=0.0)

    if not test_bbox:
        print("用户取消了框选或未完成框选，程序退出。")
        exit()

    print(f"\n接收到用户框选的边界坐标 (西, 南, 东, 北): {test_bbox}")

    # 2. 拿到框选坐标后，丢给环境生成器开始处理
    env = MapEnvironment(bbox=test_bbox, grid_resolution=2.0, padding=50.0)

    # 3. 验证并导出网页地图
    env.export_web_map("region_interactive_box.html")