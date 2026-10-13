"""
Graph.py - 磁場強度 |B| の直線分布（1次元グラフ）を作成するモジュール

使う自作モジュールは Calculation.py のみ。FieldResult.py と同じ3関数の構成。
  create_line()  : 直線上の測定点を作る                         -> Line1D
  compute_line() : 各点で Calculation.helmholtz_circle を呼ぶ    -> LineResult
  plot_graph()   : |B| を直線上の位置に対して描く                -> (fig, ax)

データ構造
  Line1D     : 直線の定義と測定点
                 axis    直線の向き "x" / "y" / "z"
                 fixed   axis 以外の2座標の固定値 [m]  例) axis="z" -> {"x": 0.03, "y": 0.02}
                 s       axis 方向の座標 (N,)
                 points  各点の3次元座標 (N, 3)   最後の軸 = (x, y, z)
  LineResult : 計算結果
                 line, a, I
                 B        各点の磁場強度 |B| (N,) [T]    ← Calculation が返すスカラー値
                 B_center 原点の磁場強度 |B(0,0,0)| [T]

座標系は Calculation.py に合わせ、コイルは z = ±a/2、半径 a、軸 = z。
（固定座標を入力で受け取る形にしたので、input() やグラフの plt.show() は持たない。
  どこから呼んでも使えるよう、表示・保存は呼び出し側に任せる。）
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import IntegrationWarning

import Calculation as Cal

AXIS = {"x": 0, "y": 1, "z": 2}
WIRE_TOL = 1e-3     # 導線からの距離がこの値（×a）以内の点は、数値積分が収束しないので計算しない


def other_axes(axis: str) -> list:
    """axis 以外の2軸（固定座標になる軸）。 例) "z" -> ["x", "y"]"""
    return [n for n in AXIS if n != axis]


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------
@dataclass
class Line1D:
    axis: str
    fixed: dict
    s: np.ndarray
    points: np.ndarray


@dataclass
class LineResult:
    line: Line1D
    a: float
    I: float
    B: np.ndarray
    B_center: float


# ---------------------------------------------------------------------------
# 1. 直線上の測定点の作成
# ---------------------------------------------------------------------------
def create_line(axis: str, a: float, fixed: dict, n: int = 401, span: float = 2.0) -> Line1D:
    """axis 方向の直線上に n 点を作る。範囲は ±span×a。他の2座標は fixed で固定する。
    n は奇数にすると原点を通る直線で中心が測定点に含まれる。"""
    if axis not in AXIS:
        raise ValueError(f"axis は x, y, z のいずれかを指定してください: {axis!r}")
    names = other_axes(axis)
    missing = [k for k in names if k not in fixed]
    if missing:
        raise ValueError(f"fixed に {missing} の値が必要です（axis={axis!r}）")
    if a <= 0 or n < 2:
        raise ValueError("a は正、n は 2 以上を指定してください")

    s = np.linspace(-span * a, span * a, n)
    points = np.empty((n, 3))
    for name in names:
        points[:, AXIS[name]] = fixed[name]
    points[:, AXIS[axis]] = s
    return Line1D(axis=axis, fixed={k: float(fixed[k]) for k in names}, s=s, points=points)


# ---------------------------------------------------------------------------
# 2. 各点の磁場計算
# ---------------------------------------------------------------------------
def compute_line(line: Line1D, a: float, I: float) -> LineResult:
    """各点で Calculation.helmholtz_circle（磁場強度 [T]）を呼び、LineResult にまとめる。
    導線から WIRE_TOL×a 以内の点は、磁場が発散し数値積分が誤った値を返すので NaN にする
    （グラフでは線が途切れる）。それ以外の導線近傍の点は積分の警告だけ抑える。"""
    B = np.full(len(line.s), np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        for k, (x, y, z) in enumerate(line.points):
            if not _near_wire(x, y, z, a):
                B[k] = Cal.helmholtz_circle(x, y, z, a, I)

        B_center = float(Cal.helmholtz_circle(0.0, 0.0, 0.0, a, I))

    return LineResult(line=line, a=a, I=I, B=B, B_center=B_center)


# ---------------------------------------------------------------------------
# 3. グラフ作成
# ---------------------------------------------------------------------------
def plot_graph(result: LineResult, ax=None, show_coil: bool = True):
    """|B| を直線上の位置に対して描く。点線は中心磁場 |B(0)|、赤の縦線はコイルの位置。
    ax を渡すとそこに描く（GUI 埋め込み用）。省略時は新しい Figure を作る。
    戻り値: (fig, ax)  保存は fig.savefig(...)"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5.5))
    else:
        fig = ax.figure

    line = result.line
    ax.plot(line.s, result.B, color="k", lw=2, label="|B|")
    ax.axhline(result.B_center, color="gray", lw=0.8, ls="--", label="|B(0)| (center)")

    if show_coil:
        for k, pos in enumerate(_coil_marks(line, result.a)):
            ax.axvline(pos, color="red", lw=0.8, ls=":", label="coil" if k == 0 else "_nolegend_")

    names = other_axes(line.axis)
    fixed_text = f"({', '.join(names)}) = ({', '.join(f'{line.fixed[n]:g}' for n in names)}) m"
    ax.set_xlabel(f"{line.axis} [m]   (fixed: {fixed_text})")
    ax.set_ylabel("|B| [T]")
    ax.set_title(f"|B| along {line.axis}-axis at {fixed_text}\n"
                 f"a = {result.a:g} m, I = {result.I:g} A", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig, ax


def _near_wire(x: float, y: float, z: float, a: float) -> bool:
    """点 (x, y, z) が導線（z = ±a/2、半径 a の円）から WIRE_TOL×a 以内か"""
    rho = math.hypot(x, y)
    return min(math.hypot(rho - a, z - a / 2), math.hypot(rho - a, z + a / 2)) < WIRE_TOL * a


def _coil_marks(line: Line1D, a: float) -> list:
    """グラフに目印を入れるコイル関連の位置 [m]。
    z 方向: コイル面の位置 ±a/2。
    x / y 方向: 直線が導線と交わるときだけ、その交点（z = ±a/2 かつ 水平距離が a 以内）。"""
    if line.axis == "z":
        return [-a / 2, a / 2]
    perp = line.fixed["y"] if line.axis == "x" else line.fixed["x"]   # 直線に垂直な水平方向の座標
    if np.isclose(abs(line.fixed["z"]), a / 2, rtol=0, atol=1e-9 * a) and abs(perp) <= a:
        h = np.sqrt(a**2 - perp**2)
        return [-h, h]
    return []


# ---------------------------------------------------------------------------
# 単体テスト用
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    a, I = 0.1, 1.0
    line = create_line("z", a, {"x": 0.03, "y": 0.02})
    res = compute_line(line, a, I)
    print(f"点数 {len(line.s)}, |B(0)| = {res.B_center:.4e} T")
    plot_graph(res)
    plt.show()