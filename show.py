import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib_fontja
import networkx as nx
from networkx import Graph
import numpy as np
from PIL import Image
import os  # ファイル・ディレクトリ操作に使用
import shutil  # ファイル・ディレクトリ操作に使用
import time

# 山口くん↓
from typing import Any, List, Tuple
from numpy.typing import NDArray

# パラメータ
population = 2000  # 千葉県の人口密度(km^2)
holding_ratio = 0.886
num_nodes = int(population * holding_ratio)  # ノード数
A = (1 / population) * 1000**2  # 面積(k^2 → m^2　に変換)
min_distance = float(np.sqrt(A / np.pi))  # ノード間の最小距離
radius = 30  # ノードを中心とした円の半径(接続半径) bluetooth想定
multiple = 1  # 円の面積の倍数(√n * pi * r^2)

# 0の時は途中経過をgifで表示、1の時は最終結果だけを画像で表示, 2の時は移動なし表示だけ
plot_pattern = 3
num_div = 10  # セルの分割数(n*n)
dist = 50  # 移動距離
node_limits = 2**2 - 1  # 接続可能ノード数
iterations = 10  # シミュレーション回数
active_node = 5


class setting:
    # 固定変数
    x_range = (0, 1000)  # x軸
    y_range = (0, 1000)  # y軸
    node_x_range = (10, 990)  # ノード用x軸
    node_y_range = (10, 990)  # ノード用y軸
    outputdir_image = "simulation_image"  # imageの保存先
    outputdir_gif = "simulation_gif"  # gifの保存先

    def __init__(
        self,
        plot_pattern,
        num_nodes,
        min_distance,
        radius,
        multiple,
        num_div,
        dist,
        iterations,
        active_node,
        node_limits,
    ):
        print(f"Initializing Setting...")
        # 変数の初期化
        self.num_nodes = num_nodes  # ノード数
        self.positions = {}  # ノードの位置配列
        self.G = nx.Graph()  # グラフの生成
        self.radius = np.sqrt(multiple) * radius  # 各ノード半径を格納する配列
        self.circles = {}  # 各ノードの円の格納配列
        self.fig, self.ax = plt.subplots(1, 3, figsize=(21, 7))
        self.plot_pattern = plot_pattern
        self.num_div = num_div
        self.dist = dist
        self.node_size = 1  # 描画ノードサイズ
        self.iterations = iterations
        self.first_sim = True  # 初回のシミュレーションのフラグ
        self.active_node = active_node  # 動的ノードの保持
        self.node_limits = node_limits
        self.closest_node = None
        self.color = ["blue"]

        # ノードの配置
        center_x = 0
        center_y = 0
        closest_distance = 9999  # 初期値を無限大にする
        for node_id in range(self.num_nodes):
            while True:
                pos = (
                    np.random.default_rng().uniform(
                        self.node_x_range[0], self.node_x_range[1]
                    ),
                    np.random.default_rng().uniform(
                        self.node_y_range[0], self.node_y_range[1]
                    ),
                )
                if all(
                    np.linalg.norm(np.array(pos) - np.array(p)) >= min_distance
                    for p in self.positions.values()
                ):
                    self.positions[node_id] = pos
                    (x, y) = pos
                    distance = (
                        (x - self.x_range[1] / 2) ** 2 + (y - self.y_range[1] / 2) ** 2
                    ) ** 0.5
                    if distance < closest_distance:
                        closest_distance = distance
                        self.closest_node = node_id
                    break

        self.routing = {i: {"neighbors": [], "parent": None} for i in self.positions}
        self.hops = {i: None for i in self.positions}
        self.s = {i: None for i in self.positions}
        for i in self.positions:
            self.s[i] = np.random.default_rng().integers(2, 13)

        if self.plot_pattern != 2 and self.plot_pattern != 3:
            self.plot_node()
            for i in self.positions:
                child_node = self.circle_detection(i)
                for node_id in child_node:
                    self.plot_edge(i, node_id)
                    self.routing_update(node_id, i)
                    self.draw()

        self.generate_circle()

    # 生成済みのノードの円の描画の準備
    def generate_circle(self):
        for node_id, (x, y) in self.positions.items():
            # 接続円をランダムに決める↓
            # self.radius[node_id] = np.random.default_rng().uniform(radius[0], radius[1])
            circle = patches.Circle(
                (x, y),
                radius=self.radius,
                edgecolor="lightblue",
                facecolor="lightblue",
                linestyle="solid",  # デフォルト
                # linestyle="dotted",   #点線
                fill=True,
            )
            self.circles[node_id] = circle
            self.ax[0].add_patch(circle)
            circle.set_visible(False)

    # ノードの描画
    def plot_node(self, node_id=None):
        # 引数が渡された場合、そのノードだけを追加
        if node_id is not None:
            pos = self.positions.get(node_id)  # 引数として渡されたノードの位置を取得
            if pos is not None:
                self.G.add_node(node_id, pos=pos, color=self.color)
        # 引数がない場合はself.positionsの全てのノードを追加
        else:
            for i, pos in self.positions.items():
                self.G.add_node(i, pos=pos, color=self.color)

    # エッジの描画/削除
    def plot_edge(self, node_id_1, node_id_2, remove=False):
        if node_id_1 in self.G and node_id_2 in self.G and remove is False:
            self.G.add_edge(node_id_1, node_id_2)
        elif remove is True:
            if self.G.has_edge(node_id_1, node_id_2) or self.G.has_edge(
                node_id_2, node_id_1
            ):
                self.G.remove_edge(node_id_1, node_id_2)
        else:
            print(f"ERROR:ノード{node_id_1}または、ノード{node_id_2}が存在しません")

    # ノードの色を変える None:デフォルトカラー
    def change_node_color(self, node_ids=[], node_color=""):
        # copy_G = self.G.copy()  # グラフのコピー
        if node_ids is not None:
            self.color = [
                node_color if node in node_ids else "blue" for node in self.G.nodes()
            ]  # 親ノードは赤でそれ以外は薄青にする
            # ノードの色をかえる
            self.clear_plot()
            self.generate_circle()
            self.plot_node()
            seen_routes = set()  # 重複した内容は格納されない
            for node_id_1 in self.routing.keys():
                for node_id_2 in self.routing[node_id_1]["neighbors"]:
                    route = tuple([node_id_1, node_id_2])
                    reverse_route = tuple([node_id_2, node_id_1])
                    if route not in seen_routes and reverse_route not in seen_routes:
                        self.plot_edge(node_id_1, node_id_2)
                        seen_routes.add(route)
                        seen_routes.add(reverse_route)
            self.draw()

    # 円の描画のON/OFF
    def taggle_circle(self, node_id, visible):
        if node_id in self.circles:
            self.circles[node_id].set_visible(visible)
        else:
            print(f"Node {node_id} not found")

    # 円の検知(サークル内にいる子ノードを返す)
    def circle_detection(self, parent_node, reverse=True) -> list:
        parent_pos = np.array(self.positions[parent_node])
        children_in_radius = []

        for node_id, pos in self.positions.items():
            if node_id != parent_node:  # 自分自身は除く
                child_pos = np.array(pos)
                distance = np.linalg.norm(parent_pos - child_pos)
                if distance <= self.radius:
                    children_in_radius.append((node_id, distance))

        # 距離でソートし、ノード ID だけのリストに変換
        # reverse = falseの時 昇順(近い順)/ trueの時 降順(遠い順)
        children_sort = sorted(children_in_radius, key=lambda x: x[1], reverse=reverse)
        child_node_ids = [node_id for node_id, _ in children_sort]  # IDのみ抽出
        return child_node_ids

    # 移動後も接続されているか確認する
    def edge_check(self):
        for node_id in self.positions:
            if len(self.routing[node_id]["neighbors"]) > 0:
                child_node = self.circle_detection(node_id)
                for node in self.routing[node_id]["neighbors"]:
                    # サークル内のノードとルーティングテーブル内のノードが一致しているか確認する
                    # 接続済みのノードがサークル内にいない場合エッジを削除する
                    if not any(node in child_node):
                        # print(f"Removing edge: {sublist}")
                        # ルーティングテーブルからエッジを削除
                        self.routing[node_id]["neighbors"].remove(node)
                        self.plot_edge(node_id, node, True)  # グラフからエッジを削除

    # ルーティングテーブルの更新/追加
    def routing_update(self, child_node=None, parent_node=None):
        # ルーティングテーブルの更新
        if parent_node is not None and child_node is not None:
            if (
                len(self.routing[parent_node]["neighbors"]) < self.node_limits
                and len(self.routing[child_node]["neighbors"]) < self.node_limits
            ):
                # 重複の確認
                if (
                    child_node not in self.routing[parent_node]["neighbors"]
                    and parent_node not in self.routing[child_node]["neighbors"]
                ):
                    self.routing[parent_node]["neighbors"].append(child_node)
                    self.routing[child_node]["neighbors"].append(parent_node)
                    self.routing[child_node]["parent"] = parent_node
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    # 各ノードのホップ数の更新
    def hop_update(self, parent_node, child_node):
        parent_hop = self.hops[parent_node]
        self.hops[child_node] = parent_hop + 1

    # 現在のプロットを全て消去
    def clear_plot(self):
        self.G.clear()
        self.ax[0].cla()
        self.circles.clear()

    # ノードをランダムに動かす (一つのノードのみ移動)
    def move(self, num, movable_node_id=0):
        # 描画されているサークルとエッジを消去する
        self.clear_plot()
        # 指定されたノードのみランダムに動かす
        (x, y) = self.positions[movable_node_id]
        rand_x = np.random.default_rng().uniform(-1, 1)
        rand_y = np.random.default_rng().uniform(-1, 1)
        move_x = self.dist * rand_x + x
        move_y = self.dist * rand_y + y
        move_x = max(self.node_x_range[0], min(self.node_x_range[1], move_x))
        move_y = max(self.node_y_range[0], min(self.node_y_range[1], move_y))
        self.positions[movable_node_id] = (move_x, move_y)
        # 円のサークル再描画
        self.generate_circle()
        # ノードの再描画
        self.plot_node()
        # エッジの再描画
        seen_routes = set()  # 重複した内容は格納されない
        for node_id_1 in self.routing.keys():
            for node_id_2 in self.routing[node_id_1]["neighbors"]:
                route = tuple([node_id_1, node_id_2])
                reverse_route = tuple([node_id_2, node_id_1])
                if route not in seen_routes and reverse_route not in seen_routes:
                    self.plot_edge(node_id_1, node_id_2)
                    seen_routes.add(route)
                    seen_routes.add(reverse_route)
        self.edge_check()
        self.draw(num=num)

    # 現在の状態を保存
    def save_image(self, frame_index=None, density=False):
        if not frame_index is None and density is False:
            image_filename = os.path.join(
                self.outputdir_image, f"simulation_{frame_index}.png"
            )
        elif density is False:
            image_filename = os.path.join(self.outputdir_gif, f"simulation_result.png")
        else:
            image_filename = os.path.join(self.outputdir_gif, f"density_map.png")
        self.fig.savefig(image_filename, bbox_inches="tight")
        return image_filename

    # 保存した画像を結合してgifを生成
    def generate_gif(self, image_files):
        images = [Image.open(img_file) for img_file in image_files]
        gif_filename = os.path.join(self.outputdir_gif, "simulation.gif")
        images[0].save(
            gif_filename, save_all=True, append_images=images[1:], duration=300
        )
        # print(f"GIF saved as {gif_filename}")

    # グラフの描画
    def draw_graph(self, ax=0, num=0):
        if ax == 0:
            div_steps = self.x_range[1] / self.num_div
            self.ax[ax].set_xlim(self.x_range)
            self.ax[ax].set_ylim(self.y_range)
            self.ax[ax].set_xticks(
                np.arange(self.x_range[0], self.x_range[1], step=div_steps)
            )
            self.ax[ax].set_yticks(
                np.arange(self.y_range[0], self.y_range[1], step=div_steps)
            )
            self.ax[ax].grid(
                True, linestyle="--", linewidth=0.5, zorder=0
            )  # 罫線を表示
            self.ax[ax].set_xlabel("X軸")
            self.ax[ax].set_ylabel("Y軸", labelpad=15, rotation="horizontal")
            if num != 0:
                self.ax[ax].set_title(f"残シミュレーション回数: {num} 回")
            else:
                self.ax[ax].set_title(f"シミュレーション実行結果")

        elif ax == 1:
            density_matrix = self.plot_density()
            im = self.ax[ax].imshow(
                np.flipud(density_matrix),
                extent=[
                    self.x_range[0],
                    self.x_range[1],
                    self.y_range[0],
                    self.y_range[1],
                ],
                cmap="Blues",
                alpha=0.8,
            )
            self.ax[ax].set_xlabel("X軸 [m]")
            self.ax[ax].set_ylabel("Y軸 [m]", labelpad=15, rotation="horizontal")
            self.ax[ax].set_title("濃度分布 (Density Distribution)")
            c_bar = plt.colorbar(im, ax=self.ax[ax])
            c_bar.set_label("密度 [node/m^2]")
            cell_div = density_matrix.shape[0]
            x_centers = np.linspace(
                self.cell_size / 2, self.x_range[1] - self.cell_size / 2, cell_div
            )
            y_centers = np.linspace(
                self.cell_size / 2, self.x_range[1] - self.cell_size / 2, cell_div
            )
            # セル上にその濃度の値を表示
            for i, x in enumerate(x_centers):
                for j, y in enumerate(y_centers):
                    value = int(density_matrix[j, i])
                    self.ax[ax].text(
                        x,
                        y,
                        f"{value}",
                        color="black",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

        elif ax == 2:
            density_matrix = self.plot_density()
            density_values = density_matrix.flatten()
            bin_edges = np.arange(density_values.min(), density_values.max() + 1)
            print(f"グラフ幅: {bin_edges}")
            self.ax[ax].hist(
                density_values,
                bins=bin_edges,
                color="blue",
                alpha=0.7,
                edgecolor="black",
            )
            self.ax[ax].set_xlabel("濃度 (Density) [node/m^2]")
            self.ax[ax].set_ylabel("セル数 (Number of Cells) [cell]")
            self.ax[ax].set_title(
                "濃度分布のヒストグラム (Histogram of Density Distribution)"
            )
            self.ax[ax].grid(True, linestyle="--", linewidth=0.5, zorder=0)

    # ノード・エッジ・ラベルの描画
    def draw(self, num=0):
        self.draw_graph(ax=0, num=num)
        nx.draw_networkx_nodes(
            self.G,
            pos=self.positions,
            node_color=self.color,
            node_size=self.node_size,
            ax=self.ax[0],
        )
        nx.draw_networkx_edges(
            self.G, pos=self.positions, edge_color="black", ax=self.ax[0]
        )
        # nx.draw_networkx_labels(
        #     self.G, pos=self.positions, font_color="black", font_size=10, ax=self.ax[0]
        # )

    # ファイルの削除・生成
    def dir(self):
        if os.path.exists(self.outputdir_image):
            if os.path.isfile(self.outputdir_image):
                os.remove(self.outputdir_image)
            elif os.path.isdir(self.outputdir_image):
                shutil.rmtree(self.outputdir_image)
        os.makedirs(self.outputdir_image)

        if os.path.exists(self.outputdir_gif):
            if os.path.isfile(self.outputdir_gif):
                os.remove(self.outputdir_gif)
            elif os.path.isdir(self.outputdir_gif):
                shutil.rmtree(self.outputdir_gif)
        os.makedirs(self.outputdir_gif)

    # 接続様子・最終結果の描画
    def drawing_connections(self):
        frame_index = 0
        image_files = []

        if self.plot_pattern == 0:
            # ノードを動かす
            for i in range(10):  # 10回移動
                rem = 10 - i - 1
                self.move(num=rem, movable_node_id=self.active_node)
                self.taggle_circle(self.active_node, True)  # 円の表示
                self.draw(rem)
                image_files.append(self.save_image(frame_index))
                frame_index += 1

                child_node = self.circle_detection(self.active_node)
                i = 0  # node_limits数になったら終了
                for node_id in child_node:
                    if i < self.node_limits:
                        result = self.routing_update(node_id, self.active_node)
                        if result:
                            self.plot_edge(self.active_node, node_id)
                            i += 1
                    else:
                        break
                    self.draw(rem)
                self.taggle_circle(self.active_node, False)  # 円の非表示
                self.draw(rem)
                image_files.append(self.save_image(frame_index))
                frame_index += 1

                print(f"{rem}")

            self.generate_gif(image_files)
            self.save_image()

        elif self.plot_pattern == 1:
            for i in range(10):
                rem = 10 - i - 1
                self.move(num=rem, movable_node_id=self.active_node)
                child_node = self.circle_detection(self.active_node)
                i = 0
                for node_id in child_node:
                    if i < self.node_limits:
                        result = self.routing_update(node_id, self.active_node)
                        if result:
                            self.plot_edge(self.active_node, node_id)
                            i += 1
                    else:
                        break
                self.draw(rem)
                print(f"{rem}")
                print(
                    f"{self.active_node}: {self.routing[self.active_node]["neighbors"]}"
                )

            self.save_image()

        elif self.plot_pattern == 2:
            for parent_node in self.positions:
                child_node = self.circle_detection(parent_node)
                i = 0
                for node_id in child_node:
                    if i < self.node_limits:
                        result = self.routing_update(node_id, parent_node)
                        if result:
                            self.plot_edge(parent_node, node_id)
                            i += 1
                    else:
                        break
            self.draw()
            for node_id in self.positions:
                if len(self.routing[node_id]["neighbors"]) != 0:
                    self.taggle_circle(node_id=node_id, visible=True)
            _, outside_nodes = self.outside_node_detection()
            self.change_node_color(node_ids=outside_nodes, node_color="red")
            # self.change_node_color(node_ids=[self.center_node], node_color="orange")
            self.draw_graph(ax=1)
            self.draw_graph(ax=2)
            self.save_image()

        elif self.plot_pattern == 3:
            # closest_nodeは通信要求を行う最初の端末
            inside_nodes_list = self.search_path(parent_node=self.closest_node)
            self.inside_nodes, outside_nodes = self.outside_node_detection(
                inside_nodes_list
            )
            self.change_node_color(node_ids=outside_nodes, node_color="red")
            for node_id in inside_nodes_list:
                self.taggle_circle(node_id, visible=True)
            self.draw_graph(ax=1)
            self.draw_graph(ax=2)
            self.save_image()

    # 経路の生成を行う
    def search_path(self, parent_node):
        queue = [parent_node]
        connected_nodes = []
        visited = set()
        visited.add(parent_node)
        self.hops[parent_node] = 0
        self.count = 0

        while queue:
            print(f"{queue}")
            current_node = queue.pop(0)
            connect_node = []
            child_nodes = self.circle_detection(current_node)
            connected_nodes.append(current_node)
            i = 0
            for node_id in child_nodes:
                if i < self.node_limits:
                    if self.adjacent_node_check(current_node, node_id):
                        if self.routing_update(node_id, current_node):
                            self.hop_update(
                                parent_node=current_node, child_node=node_id
                            )
                            if self.edge_deletion_using_mod(child_node=node_id):
                                self.plot_edge(current_node, node_id)
                                if node_id not in connected_nodes:
                                    connect_node.append(node_id)
                                    i += 1
                else:
                    break
            queue.extend(connect_node)
        return connected_nodes

    # h ≡ 0(mod s)の時エッジを削除する関数
    def edge_deletion_using_mod(self, child_node):
        if self.hops[child_node] % self.s[child_node] == 0:
            self.count += 1
            return False
        else:
            return True

    # 送信先ノードの隣接ノードを確認して同じノードに接続しないようにする関数(祖先の子供にもいないか確認)
    def adjacent_node_check(self, parent_node, child_node):
        hops = 10
        visited = set()

        def loop_detection(node, hops):
            if hops > 0:
                visited.add(node)
                child_node_lists = self.circle_detection(node)
                for node_id in child_node_lists:
                    # for node_id in self.routing[node].get("neighbors", []):
                    if node_id == child_node:
                        return True
                    if node_id not in visited:
                        if loop_detection(node_id, hops - 1):
                            return True
            return False

        if parent_node != self.closest_node:
            if self.routing[parent_node]["parent"] is not None:
                grandparent_node = self.routing[parent_node].get("parent")
                if loop_detection(grandparent_node, hops):
                    return False
        return True

    # 全体の描画 ノード0から順に円内にいるノードと接続を開始する
    def show(self):
        print(f"Generating...")
        self.dir()
        self.plot_node()
        self.drawing_connections()
        # 最終的なself.routingの内容を表示
        # print(f"(node_id: routing)")
        # for i in self.positions:
        #     print(f"({i}: {self.routing[i]})")
        print(f"Completed!")
        (x, y) = self.positions[self.closest_node]
        x = round(x, 2)
        y = round(y, 2)
        print(f"center_node: {self.closest_node} (x: {x}, y: {y})")
        print(f"エッジ削除回数: {self.count}")

    def success_rate(self):
        success = 0
        fail = 0
        for node_id in self.positions:
            if len(self.routing[node_id]["neighbors"]) != 0:
                success += 1
            else:
                fail += 1
        connect_rate = success / self.num_nodes
        return connect_rate, success, fail

    # 全てのノードが連結しているかの確認関数
    def connection_check(self):
        visited = set()  # 訪問済みノードの記録変数
        components = []  # 連結されているノードを格納する

        # 深さ優先探索関数
        def dfs(node_id, component):
            visited.add(node_id)
            component.append(node_id)
            for neighbor in self.routing.get(node_id, []):
                if neighbor not in visited:
                    dfs(neighbor, component)

        # 探索開始
        for node_id in self.positions:
            if node_id not in visited:
                component = []
                dfs(node_id, component)
                components.append(component)

        longest_component = max(components, key=len)

        if len(longest_component) == len(self.positions):
            return True, []
        else:
            isolated_nodes = []
            for component in components:
                if component != longest_component:
                    if len(component) == 1:
                        isolated_nodes.append(component[0])  # 最初の要素だけ取り出す
                    else:
                        isolated_nodes.append(
                            component
                        )  # componentに入っている要素を全て取り出す
            return False, isolated_nodes

    # どのノードにも属さないノードを探す関数
    def outside_node_detection(self, inside_nodes_list=None):
        outside_nodes = []
        inside_nodes = []
        for node_id in self.positions:
            outside_nodes.append(node_id)

        if inside_nodes_list is None:
            for node_id in self.positions:
                if node_id not in inside_nodes:
                    around_nodes = self.circle_detection(node_id)
                    if around_nodes:
                        if node_id in outside_nodes:
                            outside_nodes.remove(node_id)
                        inside_nodes.append(node_id)
                        for around_node in around_nodes:
                            if around_node not in inside_nodes:
                                if around_node in outside_nodes:
                                    outside_nodes.remove(around_node)
                                inside_nodes.append(around_node)
        else:
            for node_id in inside_nodes_list:
                around_nodes = self.circle_detection(node_id)
                if around_nodes:
                    if node_id in outside_nodes:
                        outside_nodes.remove(node_id)
                    inside_nodes.append(node_id)
                    for around_node in around_nodes:
                        if around_node not in inside_nodes:
                            if around_node in outside_nodes:
                                outside_nodes.remove(around_node)
                            inside_nodes.append(around_node)

        return inside_nodes, outside_nodes

    def plot_density(self):
        self.cell_size = self.x_range[1] / self.num_div
        density_matrix = np.zeros((self.num_div, self.num_div))
        count = 0
        for node_id in set(self.inside_nodes):
            count += 1
            (x, y) = self.positions[node_id]
            grid_x = int(x // self.cell_size)
            grid_y = int(y // self.cell_size)
            density_matrix[grid_y][grid_x] += 1

        # print(self.inside_nodes)
        # print(np.flipud(density_matrix))
        # density_matrix /= self.cell_size**2
        # print(np.flipud(density_matrix))
        return density_matrix


if __name__ == "__main__":
    # 設定のインスタンス化
    start_time = time.time()
    sim_setting = setting(
        plot_pattern,
        num_nodes,
        min_distance,
        radius,
        multiple,
        num_div,
        dist,
        iterations,
        active_node,
        node_limits,
    )

    sim_setting.show()
    connect_success_rate, succcess, fail = sim_setting.success_rate()
    connect_success_rate = round(connect_success_rate * 100, 2)
    connected, isolated = sim_setting.connection_check()
    min_distance = round(min_distance, 2)
    print(f"ノード数: {num_nodes}")
    print(f"最小距離: {min_distance}")
    print(f"接続成功: {succcess}")
    print(f"接続失敗: {fail}")
    print(f"接続成功割合: {connect_success_rate}%")
    if connected:
        print(f"全てのノードが連結されました。")
    else:
        count = len(isolated) + 1
        print(f"孤立数: {count}")
    end_time = time.time()
    execution_time = end_time - start_time
    rounded_time = round(execution_time, 2)
    print(f"実行にかかった時間: {rounded_time} 秒")
