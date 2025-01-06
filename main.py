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
num_nodes = 30  # ノード数
min_distance = 5  # ノード間の最小距離
radius = 10  # ノードを中心とした円の半径(接続半径)
multiple = 2  # 円の面積の倍数(√n * pi * r^2)

# 0の時は途中経過をgifで表示、1の時は最終結果だけを画像で表示, 2の時はノードの移動を表示
plot_pattern = 1
num_div = 5  # セルの分割数
dist = 10  # 移動距離(8以上に設定するとedge_checkの表示がバグってしまう　最終的な結果には問題ない)
hops = 5  # 接続可能ノード数
iterations = 10  # シミュレーション回数


class setting:
    # 固定変数
    x_range = (0, 100)  # x軸
    y_range = (0, 100)  # y軸
    node_x_range = (10, 90)  # ノード用x軸
    node_y_range = (10, 90)  # ノード用y軸
    outputdir_image = "simulation_image"  # imageの保存先
    outputdir_gif = "simulation_gif"  # gifの保存先

    def __init__(
        self, plot_pattern, num_nodes, min_distance, radius, multiple, num_div, dist
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
        self.node_size = 200  # 描画ノードサイズ

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

        self.routing = {i: [] for i in self.positions}
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
            nx.draw_networkx_labels(
                copy_G, pos=self.positions, font_color="black", ax=self.ax
            )
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
                    print(f"Checking sublist: {sublist}")
                    # サークル内のノードとルーティングテーブル内のノードが一致しているか確認する
                    # 接続済みのノードがサークル内にいない場合エッジを削除する
                    if not any(node in child_node for node in sublist):
                        to_remove.append(sublist)

                for sublist in to_remove:
                    node_id_1, node_id_2 = sublist
                    print(f"Removing edge: {sublist}")
                    # ルーティングテーブルからエッジを削除
                    self.routing[node_id].remove(sublist)
                    self.plot_edge(node_id_1, node_id_2, True)  # グラフからエッジを削除

    # 現在のプロットを全て消去
    def clear_plot(self):
        self.G.clear()
        self.ax.cla()
        self.circles.clear()

    # ノードをランダムに動かす
    def move(self):
        # 描画されているサークルとエッジを消去する
        self.clear_plot()
        # ランダムな方向にノードを動かす
        for node_id, (x, y) in self.positions.items():
            rand_x = np.random.default_rng().uniform(-1, 1)
            rand_y = np.random.default_rng().uniform(-1, 1)
            move_x = self.dist * rand_x + x
            move_y = self.dist * rand_y + y
            move_x = max(self.node_x_range[0], min(self.node_x_range[1], move_x))
            move_y = max(self.node_y_range[0], min(self.node_y_range[1], move_y))
            self.positions[node_id] = (move_x, move_y)
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
        self.draw()

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
    def draw_graph(self):
        div_steps = self.x_range[1] / self.num_div
        self.ax.set_xlim(self.x_range)
        self.ax.set_ylim(self.y_range)
        self.ax.set_xticks(np.arange(self.x_range[0], self.x_range[1], step=div_steps))
        self.ax.set_yticks(np.arange(self.y_range[0], self.y_range[1], step=div_steps))
        self.ax.grid(True, linestyle="--", linewidth=0.5, zorder=0)  # 罫線を表示
        self.ax.set_xlabel("X軸")
        self.ax.set_ylabel("Y軸", labelpad=15, rotation="horizontal")
        if self.iterations != 0:
            self.ax.set_title(f"残シミュレーション回数: {self.iterations} 回")
        else:
            self.ax.set_title(f"シミュレーション実行結果")

    # ノード・エッジ・ラベルの描画
    def draw(self):
        self.draw_graph()
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
        nx.draw_networkx_labels(
            self.G, pos=self.positions, font_color="black", ax=self.ax
        )

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
            for parent_node in range(self.num_nodes):
                self.taggle_circle(parent_node, True)  # 円の表示
                self.draw()
                image_files.append(self.save_image(frame_index))
                frame_index += 1

                child_node = self.circle_detection(parent_node)
                for node_id in child_node:
                    self.plot_edge(parent_node, node_id)
                    self.routing_update(node_id, parent_node)
                    self.draw()
                    image_files.append(self.save_image(frame_index))
                    frame_index += 1
                self.taggle_circle(parent_node, False)  # 円の非表示
                self.draw()
                image_files.append(self.save_image(frame_index))
                frame_index += 1
                self.move()

            self.generate_gif(image_files)

        elif self.plot_pattern == 1:
            for parent_node in range(self.num_nodes):
                child_node = self.circle_detection(parent_node)
                for node_id in child_node:
                    self.plot_edge(parent_node, node_id)
                    self.routing_update(node_id, parent_node)
                self.draw()
                self.move()
            self.save_image()

        elif self.plot_pattern == 2:
            for parent_node in range(self.num_nodes):
                self.draw()
                image_files.append(self.save_image(frame_index))
                frame_index += 1
                self.move()

            self.generate_gif(image_files)

    # 全体の描画 ノード0から順に円内にいるノードと接続を開始する
    def show(self):
        print(f"Generating...")
        self.dir()
        self.plot_node()
        self.drawing_connections()

        # 最終的なself.routingの内容を表示
        print(f"(node_id: route hops : routing)")
        for i in self.positions:
            print(f"({i}: {self.routing[i]})")
        print(f"Completed!")


# ルーティングテーブルを制御するクラス
class routing_control(setting):
    def __init__(
        self,
        hops,
        plot_pattern,
        num_nodes,
        min_distance,
        radius,
        multiple,
        num_div,
        dist,
    ):
        super().__init__(
            plot_pattern, num_nodes, min_distance, radius, multiple, num_div, dist
        )
        self.hops = hops

    # ルーティングテーブルの更新/追加
    def routing_update(self, node_id, parent_node=None):
        # ノードにルーティングを含んでないときに新たにルーティングテーブルを作成する
        if node_id not in self.routing:
            self.routing = {node_id: []}
            # [from_node, to_node] == [A → B]

        # ルーティングテーブルの更新
        if parent_node is not None:
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

    # 接続ノード数の制限を行う
    def routing_hops(self, parent_node, child_node):
        parent_hops = len(self.routing[parent_node])
        child_hops = len(self.routing[child_node])
        if parent_hops < self.hops:
            if child_hops < self.hops:
                return None
            else:
                return child_node
        else:
            return parent_node


class EXPOSED(routing_control, setting):
    def __init__(
        self,
        hops,
        plot_pattern,
        num_nodes,
        min_distance,
        radius,
        multiple,
        num_div,
        dist,
    ):
        super().__init__(
            hops, plot_pattern, num_nodes, min_distance, radius, multiple, num_div, dist
        )
        print(f"Initializing EXPOSED...")
        self.communication_freq = {  # 各ノードの通信頻度の格納配列
            i: np.random.default_rng().uniform(0, 0.5) for i in self.positions
        }
        self.iterations = iterations
        # self.target_node = np.random.choice(list(self.positions.keys()))  # 捜索対象

    # 通信を行うかの確率計算
    def should_transmit(self) -> list:
        self.transmit_node = []  # 送信ノードを格納
        for node_id in self.positions:
            random_values = np.random.default_rng().uniform(0, 1)
            if random_values < self.communication_freq[node_id]:
                self.transmit_node.append(node_id)

    # さらし端末検出
    def exposed_detection(self, child_node):
        for parent_node in self.transmit_node:
            for node_id in child_node:
                if parent_node == node_id:
                    # 除くノードは通信頻度が低い方のノードを除く
                    if (
                        self.communication_freq[parent_node]
                        > self.communication_freq[node_id]
                    ):
                        super().change_node_color(node_id, "yellow")
                        super().taggle_circle(node_id, False)
                        self.transmit_node.remove(node_id)
                    else:
                        super().change_node_color(parent_node, "yellow")
                        super().taggle_circle(parent_node, False)
                        self.transmit_node.remove(parent_node)
                    self.exposed_count += 1

    # さらしを考慮した場合
    def exposed_connect(self):
        frame_index = 0  # 保存画像インデックス
        image_files = []  # 保存画像の格納リスト
        self.num_connection_attempts = 0  # 接続試行回数
        self.error_counter = 0  # 接続失敗回数(ルーティングテーブルが満杯だった時加算)
        self.exposed_count = 0  # さらしノードカウンタ
        # 各親ノードの周りにある子ノードを格納する配列
        child_node = {i: [] for i in self.positions.keys()}

        if self.plot_pattern == 0:  # gifと最終結果の画像出力
            while self.iterations > 0:
                self.iterations -= 1
                self.should_transmit()  # 送信要求を行うノードを出力
                super().move()  # ノードをランダムに移動させる

                # 通信要求を出しているノードの色とサークルを描画
                for parent_node in self.transmit_node:
                    # 親ノードの色を赤色に変える
                    super().change_node_color(parent_node, "red")
                    # 親ノードの接続半径を可視化する
                    super().taggle_circle(parent_node, True)
                    # 親ノードの周りの子ノードを検出する
                    child_node[parent_node] = super().circle_detection(parent_node)
                    # 現在描画状態を保存
                    image_files.append(super().save_image(frame_index))
                    frame_index += 1

                    # # ノードの接続
                    # for parent_node in self.transmit_node:
                    # さらし端末が起きている端末は親ノードから除く
                    self.exposed_detection(child_node[parent_node])
                    for i in range(len(child_node[parent_node])):
                        self.num_connection_attempts += 1
                        node_id = child_node[parent_node][i]
                        # ルーティングテーブルに空きがあるか確認
                        result = super().routing_hops(parent_node, node_id)
                        if result is None:
                            # 接続済みのノードは無視する
                            if not any(
                                node_id in sublist
                                for sublist in self.routing[parent_node]
                            ):
                                super().plot_edge(parent_node, node_id)
                                super().routing_update(node_id, parent_node)
                                super().change_node_color(parent_node, "red")
                        else:
                            # 空きがない方のノードをorangeで表示
                            super().change_node_color(result, "orange")
                            # print(f"{result} error node")
                            self.error_counter += 1
                        image_files.append(super().save_image(frame_index))
                        frame_index += 1
                    # 全ノードの色をデフォルトの"lightblue"にする
                    super().change_node_color()
                # 親ノードのサークルを消す
                for parent_node in self.transmit_node:
                    super().taggle_circle(parent_node, False)
                image_files.append(super().save_image(frame_index))
                frame_index += 1
                # 残りのシミュレーション回数を表示
                print(f"{self.iterations + 1}")
            for node_id in self.positions:
                if len(self.routing[node_id]) > 0:
                    super().taggle_circle(node_id, True)
            # 最終結果とgifの生成を行う
            super().save_image()
            super().generate_gif(image_files)

        elif self.plot_pattern == 1:
            while self.iterations != 0:
                self.iterations -= 1
                super().move()
                self.should_transmit()
                for parent_node in self.transmit_node:
                    child_node[parent_node] = super().circle_detection(parent_node)

                for parent_node in self.transmit_node:
                    self.exposed_detection(child_node[parent_node])
                    for i in range(len(child_node[parent_node])):
                        self.num_connection_attempts += 1
                        node_id = child_node[parent_node][i]
                        # ルーティングテーブルに空きがあるか確認
                        result = super().routing_hops(parent_node, node_id)
                        if result is None:
                            if not any(
                                node_id in sublist
                                for sublist in self.routing[parent_node]
                            ):  # 接続済みのノードは無視する
                                super().plot_edge(parent_node, node_id)
                                super().routing_update(node_id, parent_node)
                        else:
                            # 空きがなければエラーとしてカウントする
                            self.error_counter += 1
                    super().change_node_color()
                print(f"{self.iterations + 1}")
            for node_id in self.positions:
                if len(self.routing[node_id]) > 0:
                    super().taggle_circle(node_id, True)
            super().save_image()

    def show_exposed(self):
        super().dir()
        super().plot_node()

        print(f"Generating...")
        self.exposed_connect()
        rounded_communication_freq = {}
        for node_id, value in self.communication_freq.items():
            rounded_communication_freq[node_id] = round(value * 100, 1)
        print(f"ノード番号: 通信確率 : ルーティングテーブル")
        for node_id in self.positions:
            print(
                f"{node_id}: {rounded_communication_freq[node_id]}% : {self.routing[node_id]}"
            )
        print(f"接続試行回数:      {self.num_connection_attempts}回")
        print(f"接続失敗回数:      {self.error_counter}回")
        print(f"さらしノード回数:  {self.exposed_count}回")
        print(f"Completed!")


if __name__ == "__main__":
    start_time = time.time()
    basic = EXPOSED(
        hops, plot_pattern, num_nodes, min_distance, radius, multiple, num_div, dist
    )
    basic.show_exposed()
    end_time = time.time()
    execution_time = end_time - start_time
    rounded_time = round(execution_time, 2)
    print(f"実行にかかった時間: {rounded_time} 秒")

    # basic = setting(
    #     plot_pattern, num_nodes, min_distance, radius, multiple, num_div, dist
    # )
    # basic.show()
