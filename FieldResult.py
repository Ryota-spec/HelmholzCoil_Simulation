"""
FieldResult.py - 磁場の二次元断面（ヒートマップ）を作成するモジュール

関数は3つ（＋データ構造2つ）
  create_grid()   : 二次元格子を作る                         -> Grid2D
  compute_field() : 各格子点で Calculation.helmholtz_circle を呼ぶ -> FieldResult
  plot_heatmap()  : 中心磁場 |B(0)| を最大・0 を最小としてヒートマップを描く -> (fig, ax)

データ構造
  Grid2D      : 断面の定義と格子点座標
                  plane  "xy" / "xz" / "yz"（1文字目 = 横軸、2文字目 = 縦軸）
                  offset 断面に垂直な軸の座標 [m]（例: plane="xz", offset=0.03 → y = 3 cm の断面）
                  u, v   横軸・縦軸の座標 1次元配列 (Nu,), (Nv,)
                  points 各格子点の3次元座標 (Nv, Nu, 3)   最後の軸 = (x, y, z)
  FieldResult : 計算結果
                  grid, a, I
                  B        各格子点の磁場強度 |B| (Nv, Nu) [T]    ← Calculation が返すスカラー値
                  B_center 原点の磁場強度 |B(0,0,0)| [T]

  points[j, i] は横軸 u[i]、縦軸 v[j] の点（matplotlib の pcolormesh と同じ並び）。

座標系は Calculation.py に合わせ、コイルは z = ±a/2、半径 a、軸 = z。
"""

import math
import time
import warnings
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import IntegrationWarning

import Calculation as Cal

AXIS = {"x": 0, "y": 1, "z": 2}
PLANES = ("xy", "xz", "yz")
WIRE_TOL = 1e-3     # 導線からの距離がこの値（×a）以内の点は、数値積分が収束しないので計算しない


def normal_axis(plane: str) -> str:
    """断面に垂直な軸の名前（xz 面なら "y"）"""
    return ({"x", "y", "z"} - set(plane)).pop()


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------
@dataclass
class Grid2D:
    plane: str
    offset: float
    u: np.ndarray
    v: np.ndarray
    points: np.ndarray

    @property
    def normal_axis(self) -> str:
        """断面に垂直な軸の名前（xz 面なら "y"）"""
        return normal_axis(self.plane)

    @property
    def shape(self) -> tuple:
        """(Nv, Nu)"""
        return self.points.shape[:2]


@dataclass
class FieldResult:
    grid: Grid2D
    a: float
    I: float
    B: np.ndarray
    B_center: float


# ---------------------------------------------------------------------------
# 1. 二次元格子の作成
# ---------------------------------------------------------------------------
def create_grid(plane: str, a: float, n: int = 101, span: float = 2.0,
                offset: float = 0.0) -> Grid2D:
    """断面 plane 上に n×n の格子を作る。範囲は各軸 ±span×a。
    n は奇数にすると原点（中心）が格子点に含まれる。"""
    if plane not in PLANES:
        raise ValueError(f"plane は {PLANES} のいずれかを指定してください: {plane!r}")
    if a <= 0 or n < 2:
        raise ValueError("a は正、n は 2 以上を指定してください")

    L = span * a
    u = np.linspace(-L, L, n)
    v = np.linspace(-L, L, n)
    U, V = np.meshgrid(u, v)                      # (Nv, Nu)

    points = np.empty((n, n, 3))
    points[..., AXIS[plane[0]]] = U
    points[..., AXIS[plane[1]]] = V
    points[..., AXIS[normal_axis(plane)]] = offset
    return Grid2D(plane=plane, offset=float(offset), u=u, v=v, points=points)


# ---------------------------------------------------------------------------
# 2. 各格子点の磁場計算
# ---------------------------------------------------------------------------
def compute_field(grid: Grid2D, a: float, I: float, progress=None) -> FieldResult:
    """各格子点で Calculation.helmholtz_circle（磁場強度 [T]）を呼び、FieldResult にまとめる。
    progress: 進捗通知用のコールバック progress(done, total)（GUI の進捗表示用、省略可）。
    導線から WIRE_TOL×a 以内の点は、磁場が発散し数値積分が誤った値を返すので NaN にする
    （ヒートマップでは灰色）。それ以外の導線近傍の点は積分の警告だけ抑える。
    """
    B = np.full(grid.shape, np.nan)
    total = grid.shape[0] * grid.shape[1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        for k, idx in enumerate(np.ndindex(grid.shape)):
            x, y, z = grid.points[idx]
            if not _near_wire(x, y, z, a):
                B[idx] = Cal.helmholtz_circle(x, y, z, a, I)
            if progress is not None:
                progress(k + 1, total)

        B_center = float(Cal.helmholtz_circle(0.0, 0.0, 0.0, a, I))

    return FieldResult(grid=grid, a=a, I=I, B=B, B_center=B_center)


def _near_wire(x: float, y: float, z: float, a: float) -> bool:
    """点 (x, y, z) が導線（z = ±a/2、半径 a の円）から WIRE_TOL×a 以内か"""
    rho = math.hypot(x, y)
    return min(math.hypot(rho - a, z - a / 2), math.hypot(rho - a, z + a / 2)) < WIRE_TOL * a


# ---------------------------------------------------------------------------
# 3. ヒートマップ作成
# ---------------------------------------------------------------------------
def plot_heatmap(result: FieldResult, ax=None, cmap: str = "viridis", show_coil: bool = True):
    """|B| を 0 〜 |B(0)|（中心磁場）の範囲で色付けする。中心磁場を超える領域は最大色で飽和。
    ax を渡すとそこに描く（GUI 埋め込み用）。省略時は新しい Figure を作る。
    戻り値: (fig, ax)  保存は fig.savefig(...)"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    g = result.grid
    vmax = result.B_center if result.B_center > 0 else np.nanmax(result.B)

    colormap = plt.get_cmap(cmap).with_extremes(bad="lightgray")   # 計算できなかった点（NaN）を灰色に
    mesh = ax.pcolormesh(g.u, g.v, result.B, shading="nearest",
                         cmap=colormap, vmin=0.0, vmax=vmax)
    fig.colorbar(mesh, ax=ax, extend="max", label="|B| [T]")

    if show_coil:
        _draw_coil_section(ax, result)

    ax.set_aspect("equal")
    ax.set_xlabel(f"{g.plane[0]} [m]")
    ax.set_ylabel(f"{g.plane[1]} [m]")
    ax.set_title(f"|B| on {g.plane}-plane ({g.normal_axis} = {g.offset:g} m)  "
                 f"a = {result.a:g} m, I = {result.I:g} A")
    return fig, ax


def _draw_coil_section(ax, result: FieldResult):
    """断面とコイル（z = ±a/2、半径 a）の位置関係を描く"""
    g, a = result.grid, result.a
    if g.plane == "xy":
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(a * np.cos(th), a * np.sin(th), "r--", lw=1, label="coil (projection)")
    else:
        # xz / yz 断面: 導線が断面を貫く点 (±sqrt(a^2 - d^2), ±a/2)
        d = g.offset
        if abs(d) <= a:
            h = np.sqrt(a**2 - d**2)
            ax.plot([-h, h, -h, h], [-a / 2, -a / 2, a / 2, a / 2], "o",
                    mfc="none", mec="red", ms=8, label="coil wire")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=8)


# ---------------------------------------------------------------------------
# 単体テスト用
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    a, I = 0.1, 1.0
    grid = create_grid("xz", a, n=61)
    t0 = time.time()
    res = compute_field(grid, a, I)
    print(f"計算時間 {time.time() - t0:.1f} s, 格子 {grid.shape}, |B(0)| = {res.B_center:.4e} T")
    plot_heatmap(res)
    plt.show()