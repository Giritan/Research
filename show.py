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
population = 1022  # 埼玉県の人口密度(km^2)
holding_ratio = 0.886
num_nodes = int(population * holding_ratio)  # ノード数
A = (1 / population) * 1000 * 1000  # 面積(k^2 → m^2　に変換)
min_distance = int(np.sqrt(A / np.pi))  # ノード間の最小距離
radius = 30  # ノードを中心とした円の半径(接続半径) bluetooth想定
multiple = 1  # 円の面積の倍数(√n * pi * r^2)

# 0の時は途中経過をgifで表示、1の時は最終結果だけを画像で表示, 2の時はノードの移動を表示
plot_pattern = 1
num_div = 10  # セルの分割数
dist = 50  # 移動距離
hops = 8  # 接続可能ノード数
iterations = 10  # シミュレーション回数
active_node = 0


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
        hops,
    ):
        print(f"Initializing Setting...")
        # 変数の初期化
        self.num_nodes = num_nodes  # ノード数
        self.positions = {}  # ノードの位置配列
        self.G = nx.Graph()  # グラフの生成
        self.radius = np.sqrt(multiple) * radius  # 各ノード半径を格納する配列
        self.circles = {}  # 各ノードの円の格納配列
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.plot_pattern = plot_pattern
        self.num_div = num_div
        self.dist = dist
        self.node_size = 20  # 描画ノードサイズ
        self.iterations = iterations
        self.first_sim = True  # 初回のシミュレーションのフラグ
        self.active_node = active_node  # 動的ノードの保持
        self.hops = hops

        # ノードの配置
        for i in range(self.num_nodes):
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
                    self.positions[i] = pos
                    break
            print(f"{i} is created!")

        self.routing = {i: [] for i in self.positions}

        self.plot_node()
        for i in self.positions:
            child_node = self.circle_detection(i)
            print(f"{i}: search child node")
            for node_id in child_node:
                self.plot_edge(i, node_id)
                self.routing_update(node_id, i)
                self.draw()
            print(f"{i}: child node connected!")

        self.generate_circle()

    # 生成済みのノードの円の描画の準備
    def generate_circle(self):
        for node_id, (x, y) in self.positions.items():
            # 接続円をランダムに決める↓
            # self.radius[node_id] = np.random.default_rng().uniform(radius[0], radius[1])
            circle = patches.Circle(
                (x, y),
                radius=self.radius,
                edgecolor="blue",
                linestyle="dotted",
                fill=False,
            )
            self.circles[node_id] = circle
            self.ax.add_patch(circle)
            circle.set_visible(False)

    # ノードの描画
    def plot_node(self, node_id=None):
        # 引数が渡された場合、そのノードだけを追加
        if node_id is not None:
            pos = self.positions.get(node_id)  # 引数として渡されたノードの位置を取得
            if pos is not None:
                self.G.add_node(node_id, pos=pos, color="lightblue")
        # 引数がない場合はself.positionsの全てのノードを追加
        else:
            for i, pos in self.positions.items():
                self.G.add_node(i, pos=pos, color="lightblue")

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
    def change_node_color(self, node_id=None, node_color=""):
        copy_G = self.G.copy()  # グラフのコピー
        if node_id is not None:
            self.draw_graph()
            color = [
                node_color if node == node_id else "lightblue"
                for node in self.G.nodes()
            ]  # 親ノードは赤でそれ以外は薄青にする
            nx.draw_networkx_nodes(
                copy_G,
                pos=self.positions,
                node_color=color,
                node_size=self.node_size,
                ax=self.ax,
            )
            nx.draw_networkx_edges(
                copy_G, pos=self.positions, edge_color="gray", ax=self.ax
            )
            # nx.draw_networkx_labels(
            #     copy_G, pos=self.positions, font_color="black", font_size=0, ax=self.ax
            # )
        else:
            if copy_G in self.G:
                copy_G.clear()
            self.draw()

    # 円の描画のON/OFF
    def taggle_circle(self, node_id, visible):
        if node_id in self.circles:
            self.circles[node_id].set_visible(visible)
        else:
            print(f"Node {node_id} not found")

    # 円の検知(サークル内にいる子ノードを返す)
    def circle_detection(self, parent_node) -> list:
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
        children_sort = sorted(children_in_radius, key=lambda x: x[1], reverse=False)
        child_node_ids = [node_id for node_id, _ in children_sort]  # IDのみ抽出
        return child_node_ids

    # 移動後も接続されているか確認する
    def edge_check(self):
        for node_id in self.positions:
            if len(self.routing[node_id]) > 0:
                child_node = self.circle_detection(node_id)
                to_remove = []
                for sublist in self.routing[node_id]:
                    # print(f"Checking sublist: {sublist}")
                    # サークル内のノードとルーティングテーブル内のノードが一致しているか確認する
                    # 接続済みのノードがサークル内にいない場合エッジを削除する
                    if not any(node in child_node for node in sublist):
                        to_remove.append(sublist)

                for sublist in to_remove:
                    node_id_1, node_id_2 = sublist
                    # print(f"Removing edge: {sublist}")
                    # ルーティングテーブルからエッジを削除
                    self.routing[node_id].remove(sublist)
                    self.plot_edge(node_id_1, node_id_2, True)  # グラフからエッジを削除

    # ルーティングテーブルの更新/追加
    def routing_update(self, node_id, parent_node=None):
        # ノードにルーティングを含んでないときに新たにルーティングテーブルを作成する
        if node_id not in self.routing:
            self.routing = {node_id: []}
            # [from_node, to_node] == [A → B]

        # ルーティングテーブルの更新
        if parent_node is not None:
            if (
                len(self.routing[parent_node]) < hops
                and len(self.routing[node_id]) < hops
            ):
                # 重複の確認
                pair = (parent_node, node_id)
                pair_reverse = (node_id, parent_node)
                if (
                    pair not in self.routing[parent_node]
                    and pair_reverse not in self.routing[parent_node]
                ):
                    self.routing[parent_node].append((parent_node, node_id))
                if (
                    pair not in self.routing[node_id]
                    and pair_reverse not in self.routing[node_id]
                ):
                    self.routing[node_id].append((parent_node, node_id))

    # 現在のプロットを全て消去
    def clear_plot(self):
        self.G.clear()
        self.ax.cla()
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
        for _, node_pare in self.routing.items():
            for node_id in node_pare:
                node_id_1, node_id_2 = node_id
                route = tuple([node_id_1, node_id_2])
                if route not in seen_routes:
                    self.plot_edge(node_id_1, node_id_2)
                    seen_routes.add(route)
        self.edge_check()
        self.draw(num=num)

    # 現在の状態を保存
    def save_image(self, frame_index=None):
        if not frame_index is None:
            image_filename = os.path.join(
                self.outputdir_image, f"simulation_{frame_index}.png"
            )
        else:
            image_filename = os.path.join(self.outputdir_gif, f"simulation_result.png")
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
    def draw_graph(self, num):
        div_steps = self.x_range[1] / self.num_div
        self.ax.set_xlim(self.x_range)
        self.ax.set_ylim(self.y_range)
        self.ax.set_xticks(np.arange(self.x_range[0], self.x_range[1], step=div_steps))
        self.ax.set_yticks(np.arange(self.y_range[0], self.y_range[1], step=div_steps))
        self.ax.grid(True, linestyle="--", linewidth=0.5, zorder=0)  # 罫線を表示
        self.ax.set_xlabel("X軸")
        self.ax.set_ylabel("Y軸", labelpad=15, rotation="horizontal")
        if self.iterations != 0:
            self.ax.set_title(f"残シミュレーション回数: {num} 回")
        else:
            self.ax.set_title(f"シミュレーション実行結果")

    # ノード・エッジ・ラベルの描画
    def draw(self, num=0):
        self.draw_graph(num=num)
        nx.draw_networkx_nodes(
            self.G,
            pos=self.positions,
            node_color="lightblue",
            node_size=self.node_size,
            ax=self.ax,
        )
        nx.draw_networkx_edges(
            self.G, pos=self.positions, edge_color="gray", ax=self.ax
        )
        # nx.draw_networkx_labels(
        #     self.G, pos=self.positions, font_color="black", font_size=0, ax=self.ax
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
                for node_id in child_node:
                    self.plot_edge(self.active_node, node_id)
                    self.routing_update(node_id, self.active_node)
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
                for node_id in child_node:
                    self.plot_edge(self.active_node, node_id)
                    self.routing_update(node_id, self.active_node)
                self.draw(rem)
                print(f"{rem}")

            self.save_image()

    # 全体の描画 ノード0から順に円内にいるノードと接続を開始する
    def show(self):
        print(f"Generating...")
        self.dir()
        self.plot_node()
        self.drawing_connections()

        # 最終的なself.routingの内容を表示
        print(f"(node_id: routing)")
        for i in self.positions:
            print(f"({i}: {self.routing[i]})")
        print(f"Completed!")

    def success_rate(self):
        success = 0
        fail = 0
        for node_id in self.positions:
            if len(self.routing[node_id]) != 0:
                success += 1
            else:
                fail += 1
        connect_rate = success / self.num_nodes
        return connect_rate, success, fail

    # 全てのノードが連結しているかの確認関数
    def connection_check(self):
        visited = set()  # 訪問済みノードの記録変数

        # 深さ優先探索関数
        def dfs(node_id):
            visited.add(node_id)
            for neighbor in self.routing.get(node_id, []):
                next_node = neighbor[0] if isinstance(neighbor, tuple) else neighbor
                if next_node not in visited:
                    dfs(next_node)

        # ノード0から探索開始
        dfs(0)
        if len(visited) == self.num_nodes:
            return True, []
        else:
            isolated_nodes = [
                node for node in self.positions.keys() if node not in visited
            ]
            return False, isolated_nodes


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
        hops,
    )

    sim_setting.show()
    connect_success_rate, succcess, fail = sim_setting.success_rate()
    connect_success_rate = round(connect_success_rate, 2) * 100
    connected, isolated = sim_setting.connection_check()

    print(f"ノード数: {num_nodes}")
    print(f"最小距離: {min_distance}")
    print(f"接続成功: {succcess}")
    print(f"接続失敗: {fail}")
    print(f"接続成功割合: {connect_success_rate}%")
    if connected:
        print(f"全てのノードが連結されました。")
    else:
        print(f"孤立ノード: {isolated}")
    end_time = time.time()
    execution_time = end_time - start_time
    rounded_time = round(execution_time, 2)
    print(f"実行にかかった時間: {rounded_time} 秒")
