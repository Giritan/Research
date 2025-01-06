import tkinter as tk
from main import EXPOSED


def main():
    root = tk.Tk()
    root.title("アドホックネットワークシミュレーション")

    canvas = tk.Canvas(root, width=400, height=400)
    canvas.pack()
