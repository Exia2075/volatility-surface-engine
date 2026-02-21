from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from collections import defaultdict
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from models.volatility_surface import VolatilitySurface

def plot_surface(surface: VolatilitySurface,
                 save_path: str | None = None,
                 show: bool = True,
                 colormap: str = "RdYlGn_r") -> plt.Figure:
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    y_label = "Moneyness (K/S)" if surface.axis_mode == "moneyness" else "Strike Price ($)"

    surf = ax.plot_surface(
        surface.T_grid * 365,
        surface.y_grid, 
        surface.iv_grid * 100,
        cmap = colormap,
        alpha = 0.85,
        linewidth = 0,
        antialiased = True,
    )

    ax.scatter(
        surface.T_points * 365,
        surface.y_points,
        surface.iv_points * 100,
        color = "black",
        s = 12,
        alpha = 0.6,
        zorder = 5,
        label = "Market quotes",
    )

    if surface.axis_mode == "moneyness":
        atm_val = 1.0
    else:
        atm_val = surface.S

    T_line  = np.linspace(surface.T_grid.min() * 365, surface.T_grid.max() * 365, 100)
    atm_arr = np.full_like(T_line, atm_val)

    mid_row = surface.iv_grid[surface.iv_grid.shape[0] // 2, :]
    iv_line = np.interp(T_line,
                        np.linspace(surface.T_grid.min() * 365,
                                    surface.T_grid.max() * 365,
                                    len(mid_row)),
                        mid_row * 100)

    ax.plot(T_line, atm_arr, iv_line,
            color="navy", linewidth=2.5, alpha=0.9, label="ATM term structure")
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.05)
    cbar.set_label("Implied Volatility (%)", fontsize=11)
    cbar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))

    ax.set_xlabel("Days to Expiry", fontsize=11, labelpad=10)
    ax.set_ylabel(y_label, fontsize=11, labelpad=10)
    ax.set_zlabel("Implied Vol (%)", fontsize=11, labelpad=10)

    ax.zaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f%%"))

    title_lines = [
        f"{surface.ticker} — Implied Volatility Surface  ({surface.axis_mode.capitalize()} mode)",
        f"Underlying: ${surface.S:.2f}   |   "
        f"{surface.n_solved}/{surface.n_total} contracts solved   |   "
        f"Grid: {surface.T_grid.shape[0]}×{surface.T_grid.shape[1]}",
    ]
    ax.set_title("\n".join(title_lines), fontsize=12, pad=15)

    ax.view_init(elev=25, azim=-50)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"[Plot] Surface saved to: {save_path}")

    if show:
        plt.show()

    return fig

def plot_term(surface: VolatilitySurface, 
              save_path: str | None = None,
              show: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9,5))

    by_T: dict[float, list] = defaultdict(list)
    for T, y, iv in zip(surface.T_points, surface.y_points, surface.iv_points):
        by_T[round(T, 4)].append((y, iv))

    atm_target = 1.0 if surface.axis_mode == "moneyness" else surface.S
    Ts, atm_ivs = [], []
    for T, points in sorted(by_T.items()):
        closest = min(points, key=lambda p: abs(p[0] - atm_target))
        Ts.append(T * 365)
        atm_ivs.append(closest[1] * 100)

    ax.plot(Ts, atm_ivs, "o-", color="steelblue", linewidth=2.5,
            markersize=8, markerfacecolor="white", markeredgewidth=2)
    ax.fill_between(Ts, atm_ivs, alpha=0.15, color="steelblue")

    ax.set_xlabel("Days to Expiry", fontsize=12)
    ax.set_ylabel("ATM Implied Volatility (%)", fontsize=12)
    ax.set_title(f"{surface.ticker} - ATM Volatility Term Structure", fontsize=13)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"[Plot] Term structure saved to: {save_path}")
    if show:
        plt.show()

    return fig

if __name__ == "__main__":
    from data.data_fetcher import MockDataFetcher
    from models.volatility_surface import VolatilitySurfaceBuilder

    fetcher   = MockDataFetcher(ticker="MOCK", S=100.0, base_vol=0.20,
                                skew=-0.10, curvature=0.15)
    contracts = fetcher.fetch(option_type="call", remove_illiquid=True)

    builder = VolatilitySurfaceBuilder(axis_mode="moneyness", grid_size=50)
    surface = builder.build(contracts)

    print(f"\nSurface stats:")
    print(f"IV range: {surface.iv_points.min():.1%} – {surface.iv_points.max():.1%}")
    print(f"T range: {surface.T_points.min()*365:.0f}d – {surface.T_points.max()*365:.0f}d")
    print(f"Y range: {surface.y_points.min():.2f} – {surface.y_points.max():.2f}")

    plot_surface(surface,
                 save_path="/mnt/user-data/outputs/vol_surface_moneyness.png",
                 show=False)

    plot_term(surface,
              save_path="/mnt/user-data/outputs/vol_term_structure.png",
              show=False)
    
    builder2 = VolatilitySurfaceBuilder(axis_mode="strike", grid_size=50)
    surface2 = builder2.build(contracts)
    plot_surface(surface2,
                 save_path="/mnt/user-data/outputs/vol_surface_strike.png",
                 show=False)
    
    print("\nAll plots saved.")