import matplotlib.pyplot as plt
import matplotlib.patches as patches
import japanize_matplotlib
import networkx as nx
import numpy as np
from PIL import Image
import os       #ファイル・ディレクトリ操作に使用
import shutil   #ファイル・ディレクトリ操作に使用
# 山口くん↓
from typing import Any, List, Tuple
from numpy.typing import NDArray

#パラメータ
num_nodes = 20          # ノード数
x_range = (0, 100)      # x軸
y_range = (0, 100)      # y軸
node_x_range = (10, 90) # ノード用x軸
node_y_range = (10, 90) # ノード用y軸
min_distance = 10       # ノード間の最小距離
radius = 10             # ノードを中心とした円の半径(接続半径)
multiple = 2            # 円の面積の倍数(√n * pi * r^2)
outputdir_image = "simulation_image" #imageの保存先
outputdir_gif = "simulation_gif"     #gifの保存先
plot_pattern = 1    #0の時は途中経過をgifで表示、1の時は最終結果だけを画像で表示
num_div = 2        #セルの分割数
dist = 2        #移動距離
rand_dist = (0, 1)  #移動距離用の乱数


class setting:
    def __init__(self, plot_pattern, num_nodes, x_range, y_range, node_x_range, node_y_range, min_distance, radius, multiple, outputdir_image, outputdir_gif, num_div, dist, rand_dist):
        print(f"Initializing Setting...")
        #変数の初期化
        self.num_nodes = num_nodes               # ノード数
        self.x_range = x_range                   # x軸
        self.y_range = y_range                   # y軸
        self.node_x_range = node_x_range         #ノードのx軸
        self.node_y_range = node_y_range         #ノードのy軸
        self.positions = {}                      # ノードの位置配列
        self.G = nx.Graph()                      # グラフの生成
        # self.radius = {}                        #各ノードの半径をランダムに決め格納する配列
        self.radius = np.sqrt(multiple) * radius                        #各ノード半径を格納する配列
        self.circles = {}                        # 各ノードの円の格納配列
        self.outputdir_image = outputdir_image               # 出力画像の保存先
        self.outputdir_gif = outputdir_gif               # 出力画像の保存先
        self.fig, self.ax = plt.subplots(figsize = (7, 7))
        self.plot_pattern = plot_pattern
        self.num_div = num_div
        self.dist = dist
        self.rand_dist = rand_dist
        
        #ノードの配置
        for i in range(self.num_nodes):
            while True:
                pos = (np.random.default_rng().uniform(node_x_range[0], node_x_range[1]), 
                        np.random.default_rng().uniform(node_y_range[0], node_y_range[1]))
                if all(np.linalg.norm(np.array(pos) - np.array(p)) >= min_distance for p in self.positions.values()):
                    self.positions[i] = pos
                    break

        
        self.routing = {i: [] for i in self.positions}

        #生成済みのノードの円の描画の準備
        for node_id, (x, y) in self.positions.items():
            # self.radius[node_id] = np.random.default_rng().uniform(radius[0], radius[1])
            circle = patches.Circle((x, y), 
                                    radius=self.radius, 
                                    edgecolor='blue', 
                                    linestyle='dotted', 
                                    fill=False)
            self.circles[node_id] = circle
            self.ax.add_patch(circle)
            circle.set_visible(False)

    # ノードの描画
    def plot_node(self, node_id=None):
        # 引数が渡された場合、そのノードだけを追加
        if node_id is not None:
            pos = self.positions.get(node_id)  # 引数として渡されたノードの位置を取得
            if pos is not None:
                self.G.add_node(node_id, pos=pos, color='lightblue')
        # 引数がない場合はself.positionsの全てのノードを追加
        else:
            for i, pos in self.positions.items():
                self.G.add_node(i, pos=pos, color='lightblue')

    # エッジの描画
    def plot_edge(self, node_id_1, node_id_2):
        if node_id_1 in self.G and node_id_2 in self.G:
            self.G.add_edge(node_id_1, node_id_2)
        else:
            print(f"ERROR:ノード{node_id_1}または、ノード{node_id_2}が存在しません")
    
    # ノードの追加(手動)
    def add_node(self, node_id, x, y, parent):
        pos = (x, y)
        self.G.add_node(node_id)
        self.positions[node_id] = pos

    #ルーティングテーブルの更新/追加
    def update_routing(self, node_id, parent_node = None):
        #ノードにルーティングを含んでないときに新たにルーティングテーブルを作成する
        if node_id not in self.routing:
            self.routing = {node_id: []}
            #[from_node, to_node] == [A → B]

        #ルーティングテーブルの更新
        if parent_node is not None:
            #重複の確認
            pair = (parent_node, node_id)
            pair_reverse = (node_id, parent_node)
            if pair not in self.routing[parent_node] and pair_reverse not in self.routing[parent_node]:
                self.routing[parent_node].append((parent_node, node_id))
            if pair not in self.routing[node_id] and pair_reverse not in self.routing[node_id]:
                self.routing[node_id].append((parent_node, node_id))

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
    
    # ノードをランダムに動かす
    def move(self):
        rand = np.random.default_rng().uniform(self.rand_dist[0], self.rand_dist[1])
        for node_id, (x, y) in self.positions.items():
            move_x = self.dist * rand + x
            move_y = self.dist * rand + y
            if not (self.node_x_range[0] <= move_x and self.node_x_range >= move_x):
                move_x = x
            if not (self.node_y_range[0] <= move_y and self.node_y_range >= move_y):
                move_y = y
            self.positions[node_id] = (move_x, move_y)

    # 現在の状態を保存
    def save_image(self, frame_index = None):
        if self.plot_pattern == 0:
            image_filename = os.path.join(self.outputdir_image, f"simulation_{frame_index}.png")
        elif self.plot_pattern == 1 and frame_index is None:
            image_filename = os.path.join(self.outputdir_image, f"simulation_result.png")
        self.fig.savefig(image_filename, bbox_inches='tight')
        return image_filename

    # 保存した画像を結合してgifを生成
    def generate_gif(self, image_files):
        images = [Image.open(img_file) for img_file in image_files]
        gif_filename = os.path.join(self.outputdir_gif, "simulation.gif")
        images[0].save(
            gif_filename, save_all=True, append_images=images[1:], duration=200, loop=1
        )
        # print(f"GIF saved as {gif_filename}")

    #グラフの描画
    def draw_graph(self):
        div_steps = self.x_range[1]/self.num_div
        self.ax.set_xlim(self.x_range)
        self.ax.set_ylim(self.y_range)
        self.ax.set_xticks(np.arange(self.x_range[0], self.x_range[1], step=div_steps))
        self.ax.set_yticks(np.arange(self.y_range[0], self.y_range[1], step=div_steps))
        self.ax.grid(True, linestyle='--', linewidth=0.5, zorder=0)  # 罫線を表示
        self.ax.set_xlabel("X軸")
        self.ax.set_ylabel("Y軸", labelpad=15, rotation="horizontal")
        self.ax.set_title("ノードの描画")

    #ノード・エッジ・ラベルの描画
    def draw(self):
        node_size = 200
        self.draw_graph()
        nx.draw_networkx_nodes(self.G, pos=self.positions, node_color='lightblue', node_size=node_size, ax=self.ax)
        nx.draw_networkx_edges(self.G, pos=self.positions, edge_color='gray', ax=self.ax)
        nx.draw_networkx_labels(self.G, pos=self.positions, font_color='black', ax=self.ax)

    #ファイルの削除・生成
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

    #接続様子・最終結果の描画
    def drawing_connections(self):
        frame_index = 0
        image_files = []

        if self.plot_pattern == 0:
            # self.draw_graph() #必要なのか？
            for parent_node in range(self.num_nodes):
                self.taggle_circle(parent_node, True)  # 円の表示
                self.draw()
                image_files.append(self.save_image(frame_index))
                frame_index += 1

                child_node = self.circle_detection(parent_node)
                for node_id in child_node:
                    self.plot_edge(parent_node, node_id)
                    self.update_routing(node_id, parent_node)
                    self.draw()
                    image_files.append(self.save_image(frame_index))
                    frame_index += 1
                self.taggle_circle(parent_node, False) # 円の非表示
                self.draw()
                image_files.append(self.save_image(frame_index))
                frame_index += 1

            self.generate_gif(image_files)

        elif self.plot_pattern == 1:
            # self.draw_graph() #必要なのか？
            for parent_node in range(self.num_nodes):
                child_node = self.circle_detection(parent_node)
                for node_id in child_node:
                    self.plot_edge(parent_node, node_id)
                    self.update_routing(node_id, parent_node)
                    self.draw()
            self.save_image()
            


    # 全体の描画 ノード0から順に円内にいるノードと接続を開始する
    def show(self):
        self.dir()
        self.plot_node()
        self.drawing_connections()

        # 最終的なself.routingの内容を表示
        for i in self.positions:
            print(f"({i}: {self.routing[i]})")
        # plt.show()


class AODV(setting):
    def __init__(self, plot_pattern, num_nodes, x_range, y_range, node_x_range, node_y_range, min_distance, radius, multiple, outputdir_image, outputdir_gif, num_div, dist, rand_dist):
        super().__init__(plot_pattern, num_nodes, x_range, y_range, node_x_range, node_y_range, min_distance, radius, multiple, outputdir_image, outputdir_gif, num_div, dist, rand_dist)
        print(f"Initializing AODV...")
        self.current = 0  #通信要求出すノード
        self.target_node = np.random.choice(list(self.positions.keys()))  # 捜索対象

    def route_discovery(self):
        route_info_id = 0 #経路情報
        #捜索対象がcurrentじゃないようにする
        while self.target_node == current:
            self.target_node = np.random.choice(list(self.positions.keys()))  # 捜索対象

        while current != self.target_node:
            self.route_info = {route_info_id: []}
            next = self.circle_detection(current)
            for i in next:
                if self.target_node != i:
                    self.add_link(route_info_id, current, next)

                else:
                    self.add_link(route_info_id, current, i)
                    current = i
                    return self.route_info[route_info_id]
            
    def search(self, current, route_info_id):
        next = self.circle_detection(current)
        for i in next:
            if self.target_node != i:
                self.add_link(route_info_id, current, next)
                self.search(next, route_info_id)
                    
    #接続先を追加する
    def add_link(self, route_info_id, from_node, to_node):
        self.route_info[route_info_id].append([from_node, to_node])

    #ルーティング方法ごとに描画方法を変える
    def show_aodv(self):
        print(f"Generating...")
        super().show()
        print(f"Completed!")

if __name__ == "__main__":
    basic = AODV(plot_pattern,
                 num_nodes,
                 x_range,
                 y_range, 
                 node_x_range, 
                 node_y_range, 
                 min_distance, 
                 radius, 
                 multiple, 
                 outputdir_image, 
                 outputdir_gif,
                 num_div,
                 dist,
                 rand_dist)
    basic.show_aodv()