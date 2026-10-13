"""
GUI.py - ヘルムホルツコイル磁場シミュレーター GUI（起動: python GUI.py）

役割分担
  Calculation.py : 磁場強度 |B| の計算（helmholtz_circle）
  Graph.py       : 直線上の |B| グラフ      create_line / compute_line / plot_graph
  FieldResult.py : 断面の |B| ヒートマップ   create_grid / compute_field / plot_heatmap
  GUI.py         : 入力・計算フロー・表示・保存と、コイルの3D表示だけを担当

画面
  上部 : 電流[A]・リング半径[m]・グラフプロット軸・グラフの固定座標（プロット軸以外の2座標）
         ヒートマップ投影面・断面位置（投影面に垂直な軸の座標）＋計算実行ボタン
  左下 : コイル形状と、ヒートマップ投影面・グラフの直線の3D表示
  右下 : タブ「グラフ」「ヒートマップ」（それぞれ保存可能）

位置の指定（いずれも既定値 0。座標ごとに入力値を記憶する）
  グラフ     : プロット軸が z なら (x, y)、x なら (y, z)、y なら (x, z) を入力
  ヒートマップ: 投影面が xy なら z、xz なら y、yz なら x を入力（例: xz 面で y = 0.03）

計算フロー
  「磁場の計算を実行中」→「磁場計算が終了しました。グラフ描画を行いますか?」
  →(はい)「グラフ描画を実行しました」→「ヒートマップを作成しますか?」→(はい)「ヒートマップを作成しました」
"""

import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import Graph
import FieldResult as FR

# ---------------------------------------------------------------------------
# 設定値
# ---------------------------------------------------------------------------
RANGE_FACTOR = 2.0    # 計算・表示範囲 = ±RANGE_FACTOR × 半径（グラフの直線上・断面内）
N_LINE = 401          # グラフの測定点数
N_GRID = 101          # ヒートマップの格子数（1辺）。点数の2乗だけ計算時間が増える

AXIS = Graph.AXIS
AXES = tuple(AXIS)
PLANES = FR.PLANES
other_axes = Graph.other_axes
normal_axis = FR.normal_axis


class HelmholtzGUI:
    def __init__(self, master):
        self.master = master
        self.graph_result = None
        self.heat_result = None
        self._worker = None
        self._worker_out = {}
        self._progress = (0, 1)

        master.title("ヘルムホルツコイル 磁場シミュレーター")
        master.geometry("1400x820")

        self.current_var = tk.StringVar(value="1.0")
        self.radius_var = tk.StringVar(value="0.1")
        self.axis_var = tk.StringVar(value="z")
        self.plane_var = tk.StringVar(value="xz")
        self.status_var = tk.StringVar(value="パラメータを入力して「計算実行」を押してください")

        # 位置の入力値は座標単位で保持（軸・投影面を切り替えても他の座標の入力が消えない）
        self.line_vars = {n: tk.StringVar(value="0") for n in AXES}     # グラフの固定座標
        self.offset_vars = {n: tk.StringVar(value="0") for n in AXES}   # 断面位置
        self.fix_label_vars = [tk.StringVar(), tk.StringVar()]
        self.offset_label_var = tk.StringVar()

        self._build_param_panel()
        self._build_main_area()
        self._refresh_position_inputs()

        # 入力が変わるたびに3D図を更新。軸・投影面を変えたら入力欄の対応する座標も切り替える
        for var in (self.radius_var, *self.line_vars.values(), *self.offset_vars.values()):
            var.trace_add("write", lambda *_: self.draw_coil())
        for var in (self.axis_var, self.plane_var):
            var.trace_add("write", lambda *_: self._on_selection_change())
        self.draw_coil()

    # ---------------- レイアウト ----------------
    def _build_param_panel(self):
        top = ttk.Frame(self.master, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        row1 = ttk.Frame(top)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="電流 [A]:").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.current_var, width=10).pack(side=tk.LEFT, padx=(2, 16))
        ttk.Label(row1, text="リング半径 [m]:").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.radius_var, width=10).pack(side=tk.LEFT, padx=(2, 16))
        ttk.Label(row1, text="グラフプロット軸 (x, y, z から選択):").pack(side=tk.LEFT)
        ttk.Combobox(row1, textvariable=self.axis_var, values=AXES,
                     state="readonly", width=4).pack(side=tk.LEFT, padx=(2, 16))
        ttk.Label(row1, text="固定座標").pack(side=tk.LEFT)
        self.fix_entries = []
        for label_var in self.fix_label_vars:
            ttk.Label(row1, textvariable=label_var).pack(side=tk.LEFT, padx=(8, 0))
            entry = ttk.Entry(row1, width=10)
            entry.pack(side=tk.LEFT, padx=2)
            self.fix_entries.append(entry)

        row2 = ttk.Frame(top)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="ヒートマップ投影面 (xy, xz, yz から選択):").pack(side=tk.LEFT)
        ttk.Combobox(row2, textvariable=self.plane_var, values=PLANES,
                     state="readonly", width=4).pack(side=tk.LEFT, padx=(2, 16))
        ttk.Label(row2, textvariable=self.offset_label_var).pack(side=tk.LEFT)
        self.offset_entry = ttk.Entry(row2, width=10)
        self.offset_entry.pack(side=tk.LEFT, padx=(2, 16))
        self.run_button = ttk.Button(row2, text="計算実行", command=self.on_run)
        self.run_button.pack(side=tk.LEFT, padx=8)
        ttk.Label(row2, textvariable=self.status_var, foreground="#2d5f7f").pack(side=tk.LEFT, padx=8)

    def _build_main_area(self):
        paned = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # 左: コイル3D表示
        left = ttk.LabelFrame(paned, text="コイル形状とヒートマップ投影面", padding=4)
        self.coil_fig = Figure(figsize=(5.5, 5.5), dpi=100)
        self.coil_ax = self.coil_fig.add_subplot(111, projection="3d")
        self.coil_canvas = FigureCanvasTkAgg(self.coil_fig, master=left)
        self.coil_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        paned.add(left, weight=1)

        # 右: グラフ / ヒートマップのタブ。self.tabs[kind] = (Figure, canvas, 保存ボタン)
        self.notebook = ttk.Notebook(paned)
        self.tabs = {}
        self._make_plot_tab("グラフ", "graph")
        self._make_plot_tab("ヒートマップ", "heatmap")
        paned.add(self.notebook, weight=2)

    def _make_plot_tab(self, title, kind):
        frame = ttk.Frame(self.notebook, padding=4)
        fig = Figure(figsize=(7, 5.5), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        bar = ttk.Frame(frame)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        save = ttk.Button(bar, text=f"{title}を保存", state=tk.DISABLED,
                          command=lambda: self.save_figure(kind))
        save.pack(side=tk.RIGHT, padx=4, pady=2)
        toolbar = NavigationToolbar2Tk(canvas, bar, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.LEFT)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.notebook.add(frame, text=title)
        self.tabs[kind] = (fig, canvas, save)

    # ---------------- 位置の入力欄 ----------------
    def _refresh_position_inputs(self):
        """選択中のプロット軸・投影面に合わせて、位置入力欄のラベルと対応する座標を切り替える。"""
        for label_var, entry, name in zip(self.fix_label_vars, self.fix_entries,
                                          other_axes(self.axis_var.get())):
            label_var.set(f"{name} [m]:")
            entry.config(textvariable=self.line_vars[name])
        n = normal_axis(self.plane_var.get())
        self.offset_label_var.set(f"断面位置 {n} [m]:")
        self.offset_entry.config(textvariable=self.offset_vars[n])

    def _on_selection_change(self):
        self._refresh_position_inputs()
        self.draw_coil()

    # ---------------- 入力チェック ----------------
    @staticmethod
    def _read_float(var):
        """数値として読めない・有限でない値は None"""
        try:
            value = float(var.get())
        except ValueError:
            return None
        return value if np.isfinite(value) else None

    def _read_params(self):
        """入力欄を読んで計算パラメータにまとめる。不正なら警告を出して None。"""
        I = self._read_float(self.current_var)
        a = self._read_float(self.radius_var)
        if I is None:
            messagebox.showerror("入力エラー", "電流には数値を入力してください。")
            return None
        if a is None or a <= 0:
            messagebox.showerror("入力エラー", "リング半径には正の数値を入力してください。")
            return None

        axis, plane = self.axis_var.get(), self.plane_var.get()
        fixed = {n: self._read_float(self.line_vars[n]) for n in other_axes(axis)}
        offset = self._read_float(self.offset_vars[normal_axis(plane)])
        for name, value in (*fixed.items(), (normal_axis(plane), offset)):
            if value is None:
                kind = "グラフの固定座標" if name in fixed else "断面位置"
                messagebox.showerror("入力エラー", f"{kind} {name} には数値を入力してください。")
                return None
        return dict(a=a, I=I, axis=axis, fixed=fixed, plane=plane, offset=offset)

    # ---------------- コイル3D表示 ----------------
    def draw_coil(self):
        a = self._read_float(self.radius_var)
        axis, plane = self.axis_var.get(), self.plane_var.get()
        normal = normal_axis(plane)
        fixed = {n: self._read_float(self.line_vars[n]) for n in other_axes(axis)}
        offset = self._read_float(self.offset_vars[normal])
        if a is None or a <= 0 or offset is None or any(v is None for v in fixed.values()):
            return  # 入力途中（空欄・"0." など）は更新しない
        L = RANGE_FACTOR * a
        M = max(L, abs(offset), *(abs(v) for v in fixed.values()))   # 位置が範囲外でも見えるように
        ax = self.coil_ax
        ax.clear()

        # コイル（z = ±a/2）
        th = np.linspace(0, 2 * np.pi, 200)
        for z0 in (+a / 2, -a / 2):
            ax.plot(a * np.cos(th), a * np.sin(th), np.full_like(th, z0), color="#1b3a4f", lw=2.5)

        # ヒートマップ投影面（面内は ±L、垂直方向は offset の位置）
        corner = [None] * 3
        corner[AXIS[plane[0]]] = np.array([-L, L, L, -L])
        corner[AXIS[plane[1]]] = np.array([-L, -L, L, L])
        corner[AXIS[normal]] = np.full(4, offset)
        ax.add_collection3d(Poly3DCollection([list(zip(*corner))], alpha=0.25,
                                             facecolor="#2d5f7f", edgecolor="#2d5f7f"))

        # グラフの直線（固定座標を通る、axis 方向）
        line = [np.full(2, fixed.get(n, 0.0)) for n in AXES]
        line[AXIS[axis]] = np.array([-L, L])
        ax.plot(*line, color="#d9534f", lw=2, label=f"plot axis: {axis}")

        ax.set_xlim(-M, M); ax.set_ylim(-M, M); ax.set_zlim(-M, M)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        fixed_text = ", ".join(f"{n}={v:g}" for n, v in fixed.items())
        ax.set_title(f"R = {a:g} m\nheatmap: {plane}-plane ({normal} = {offset:g})\n"
                     f"graph: {axis}-axis ({fixed_text})", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        self.coil_canvas.draw_idle()

    # ---------------- 計算フロー ----------------
    def on_run(self):
        params = self._read_params()
        if params is None:
            return
        self.run_button.config(state=tk.DISABLED)
        self.status_var.set("磁場の計算を実行中…")
        self._worker_out = {}
        self._progress = (0, 1)
        self._worker = threading.Thread(target=self._work, args=(params,), daemon=True)
        self._worker.start()
        self.master.after(100, self._poll_worker)

    def _work(self, p):
        """別スレッドで計算（画面が固まらないように）。Graph / FieldResult の関数を呼ぶだけ。"""
        try:
            line = Graph.create_line(p["axis"], p["a"], p["fixed"], N_LINE, RANGE_FACTOR)
            self._worker_out["graph"] = Graph.compute_line(line, p["a"], p["I"])
            grid = FR.create_grid(p["plane"], p["a"], N_GRID, RANGE_FACTOR, p["offset"])
            self._worker_out["heat"] = FR.compute_field(
                grid, p["a"], p["I"], progress=lambda done, total: setattr(self, "_progress", (done, total)))
        except Exception as e:  # 計算側のエラーをGUIに伝える
            self._worker_out["error"] = e

    def _poll_worker(self):
        if self._worker.is_alive():
            done, total = self._progress
            self.status_var.set(f"磁場の計算を実行中… {done / total:.0%}")
            self.master.after(100, self._poll_worker)
            return
        self.run_button.config(state=tk.NORMAL)
        if "error" in self._worker_out:
            self.status_var.set("計算中にエラーが発生しました")
            messagebox.showerror("計算エラー", str(self._worker_out["error"]))
            return

        self.graph_result = self._worker_out["graph"]
        self.heat_result = self._worker_out["heat"]
        self.status_var.set("磁場計算が終了しました")

        if messagebox.askyesno("確認", "磁場計算が終了しました。\nグラフ描画を行いますか?"):
            self._render("graph", lambda ax: Graph.plot_graph(self.graph_result, ax))
            self.status_var.set("グラフ描画を実行しました")
            messagebox.showinfo("完了", "グラフ描画を実行しました。")

        if messagebox.askyesno("確認", "ヒートマップを作成しますか?"):
            self._render("heatmap", lambda ax: FR.plot_heatmap(self.heat_result, ax))
            self.status_var.set("ヒートマップを作成しました")
            messagebox.showinfo("完了", "ヒートマップを作成しました。")

    def _render(self, kind, draw):
        """kind のタブに draw(ax) で作図して表示する。"""
        fig, canvas, save = self.tabs[kind]
        fig.clear()
        draw(fig.add_subplot(111))
        fig.tight_layout()
        canvas.draw()
        save.config(state=tk.NORMAL)
        self.notebook.select(list(self.tabs).index(kind))
        self.master.update_idletasks()

    # ---------------- 保存 ----------------
    def _default_filename(self, kind):
        if kind == "graph":
            r = self.graph_result
            tag = r.line.axis + "_" + "_".join(f"{n}{v:g}" for n, v in r.line.fixed.items())
        else:
            r = self.heat_result
            tag = f"{r.grid.plane}_{r.grid.normal_axis}{r.grid.offset:g}"
        return f"helmholtz_{kind}_{tag}_I{r.I:g}A_R{r.a:g}m.png"

    def save_figure(self, kind):
        path = filedialog.asksaveasfilename(
            title="保存先を選択", initialfile=self._default_filename(kind), defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if not path:
            return
        self.tabs[kind][0].savefig(path, dpi=200, bbox_inches="tight")
        self.status_var.set(f"保存しました: {path}")


def launch():
    root = tk.Tk()
    HelmholtzGUI(root)

    def on_close():
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    launch()