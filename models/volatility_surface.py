# 1. take a list of OptionContracts
# 2. runs IV solver on each
# 3. interpolates smooth grid for visualization

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import RBFInterpolator

from models.implied_vol import compute_implied_vol, IVResult
from data.data_fetcher import OptionContract

@dataclass(slots=True, frozen=True)
class SolvedContract:
    contract: OptionContract
    iv_res: IVResult

    @property
    def implied_vol(self) -> float | None:
        return self.iv_res.implied_vol
    
    @property
    def T(self):
        return self.contract.T
    
    @property
    def strike(self): 
        return self.contract.strike
    
    @property
    def moneyness(self): 
        return self.contract.moneyness
    
@dataclass(slots=True, frozen=True)
class VolatilitySurface:
    ticker: str
    axis_mode: str
    
    T_points:     np.ndarray 
    y_points:     np.ndarray  
    iv_points:    np.ndarray

    T_grid:       np.ndarray     
    y_grid:       np.ndarray     
    iv_grid:      np.ndarray

    n_total: int
    n_solved: int
    n_failed: int
    S: float

class VolatilitySurfaceBuilder:
    def __init__(self, axis_mode: str="moneyness", grid_size: int=50):
        if axis_mode not in ("strike", "moneyness"):
            raise ValueError("axis_mode must be 'strike' or 'moneyness'")
        self.axis_mode = axis_mode
        self.grid_size = grid_size

    def _solve_all(self, contracts: list[OptionContract]) -> list[SolvedContract]:
        solved = []
        failed = 0
        for c in contracts: 
            res = compute_implied_vol(
                market_price = c.market_price,
                S = c.S,
                K = c.strike,
                T = c.T,
                r = c.r,
                q = c.q,
            )
            solved.append(SolvedContract(contract=c, iv_res = res))
            if not res.converged:
                failed += 1
        
        converged = len(solved) - failed
        print(f"[Surface] IV solved: {converged}/{len(contracts)} contracts converged "
              f"({failed} failed)")
        return solved

    def _interpolate_grid(self, 
                          T_pts: np.ndarray,
                          y_pts: np.ndarray,
                          iv_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        T_axis = np.linspace(T_pts.min(), T_pts.max(), self.grid_size)
        y_axis = np.linspace(y_pts.min(), y_pts.max(), self.grid_size)
        T_grid, y_grid = np.meshgrid(T_axis, y_axis)

        T_scale = T_pts.max() - T_pts.min() or 1.0
        y_scale = y_pts.max() - y_pts.min() or 1.0

        points_norm = np.column_stack([
            (T_pts - T_pts.min()) / T_scale,
            (y_pts - y_pts.min()) / y_scale,
        ])
        query_norm = np.column_stack([
            (T_grid.ravel() - T_pts.min()) / T_scale,
            (y_grid.ravel() - y_pts.min()) / y_scale,
        ])

        rbf = RBFInterpolator(points_norm, iv_pts, kernel='thin_plate_spline', smoothing=0.0001)
        iv_grid = rbf(query_norm).reshape(T_grid.shape)

        iv_grid = np.clip(iv_grid, 0.01, 2.0)
        return T_grid, y_grid, iv_grid
    
    def build(self, contracts: list[OptionContract]) -> VolatilitySurface:
        if not contracts:
            raise ValueError("No contracts provided - cannot build surface.")

        S = contracts[0].S
        ticker = contracts[0].ticker

        print(f"[Surface] Building {self.axis_mode} surface for {ticker} "
              f"({len(contracts)} contracts)...")
        
        solved = self._solve_all(contracts)

        good = [s for s in solved if s.iv_res.converged]
        n_failed = len(solved) - len(good)

        if len(good) < 4:
            raise ValueError(
                f"Only {len(good)} contracts converged — not enough to build a surface. "
                "Try relaxing filters or using a ticker with more liquid options."
            )
        
        T_pts  = np.array([s.T for s in good])
        y_pts  = np.array([s.moneyness if self.axis_mode == "moneyness"
                           else s.strike for s in good])
        iv_pts = np.array([s.implied_vol for s in good])

        T_grid, y_grid, iv_grid = self._interpolate_grid(T_pts, y_pts, iv_pts)

        return VolatilitySurface(
            ticker = ticker,
            axis_mode = self.axis_mode,
            T_points = T_pts,
            y_points = y_pts,
            iv_points = iv_pts,
            T_grid = T_grid,
            y_grid = y_grid,
            iv_grid = iv_grid,
            n_total = len(solved),
            n_solved = len(good),
            n_failed = n_failed,
            S = S,
        )