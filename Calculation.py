import math
from scipy import integrate

MU0 = 4 * math.pi * 1e-7

def _biot_savart_vector(x, y, z, a, I, z0=0.0):
    """
    z=z0 の平面上にあり、z軸を中心とする半径 a の円電流が点 (x, y, z) に作る磁場ベクトル [T]
    ※内部計算用の関数
    """
    dz = z - z0

    def denom(theta):
        return ((x - a*math.cos(theta))**2 + (y - a*math.sin(theta))**2 + dz**2) ** 1.5

    def fx(theta):
        return a * dz * math.cos(theta) / denom(theta)

    def fy(theta):
        return a * dz * math.sin(theta) / denom(theta)

    def fz(theta):
        return a * (a - x*math.cos(theta) - y*math.sin(theta)) / denom(theta)

    k = MU0 * I / (4 * math.pi)
    Bx = integrate.quad(fx, 0, 2*math.pi)[0]
    By = integrate.quad(fy, 0, 2*math.pi)[0]
    Bz = integrate.quad(fz, 0, 2*math.pi)[0]
    return (k*Bx, k*By, k*Bz)


def biot_savart_circle(x, y, z, a, I, z0=0.0):
    """単一の円電流が作る磁場強度の絶対値 [T] を返す"""
    Bx, By, Bz = _biot_savart_vector(x, y, z, a, I, z0)
    return math.sqrt(Bx**2 + By**2 + Bz**2)


def helmholtz_circle(x, y, z, a, I):
    """間隔 a で z = ±a/2 に置いた2つのコイルの磁場の和の絶対値 [T] を返す"""
    # 1. 必ずベクトルとして取得する
    B1 = _biot_savart_vector(x, y, z, a, I, z0=+a/2)
    B2 = _biot_savart_vector(x, y, z, a, I, z0=-a/2)
    
    # 2. ベクトルの成分ごとに足し合わせる
    Bx = B1[0] + B2[0]
    By = B1[1] + B2[1]
    Bz = B1[2] + B2[2]
    
    # 3. 最後に合成ベクトルの絶対値を計算する
    return math.sqrt(Bx**2 + By**2 + Bz**2)


if __name__ == "__main__":
    a, I = 1.0, 1.0
    
    # 検算1：単一ループの中心 → μ0 I / (2a)
    # （中心ではBx=0, By=0のため、Bzの絶対値と一致する）
    print("単一ループの比較:")
    print("  計算値:", biot_savart_circle(0, 0, 0, a, I))
    print("  理論値:", MU0*I/(2*a))
    
    # 検算2：ヘルムホルツの中心 → (4/5)^(3/2) μ0 I / a
    print("\nヘルムホルツの比較:")
    print("  計算値:", helmholtz_circle(0, 0, 0, a, I))
    print("  理論値:", (4/5)**1.5 * MU0*I/a)