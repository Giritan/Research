import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# グラフの設定
fig, ax = plt.subplots()

# サークルを作成
circle = plt.Circle((5, 5), 1, color='r', animated=True)  # 中心(5,5)、半径1の赤いサークル
ax.add_patch(circle)

# 軸の範囲を設定
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# 無限に続くジェネレータ関数
def frame_function():
    while True:
        yield np.random.choice([True, False])  # True: 表示, False: 非表示

# 初期化関数
def init():
    circle.set_visible(False)  # 初期状態でサークルを非表示にする
    return circle,

# 更新関数
def update(frame):
    circle.set_visible(frame)  # frameがTrueなら表示、Falseなら非表示
    return circle,

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=frame_function, init_func=init, interval=500, blit=True)

# Pillowライターを使用してGIFとして保存
ani.save("circle_blinking_pillow.gif", writer="pillow", fps=10)

# アニメーションを表示
plt.show()
