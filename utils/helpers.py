from __future__ import annotations
import sys

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
RESET = "\033[0m"

def print_header(ticker: str, mode: str, filter_mode: str):
    print(f"{BOLD}Volatility Surface Engine{RESET}")
    print(f"Ticker: {BOLD}{ticker}{RESET}")
    print(f"Axis: {mode}")
    print(f"Filter: {filter_mode}")

def print_success(msg: str):
    print(f"{GREEN}OK{RESET} {msg}")

def print_warning(msg: str):
    print(f"{YELLOW}WARN{RESET} {msg}")

def print_error(msg: str):
    print(f"{RED}ERROR{RESET} {msg}", file=sys.stderr)

def print_summary(surface):
    print("Surface Summary")
    print(f"Contracts solved: {surface.n_solved} / {surface.n_total}")
    print(f"Failed: {surface.n_failed}")
    print(f"IV range: {surface.iv_points.min():.1%} - {surface.iv_points.max():.1%}")
    print(f"Maturity range: {surface.T_points.min()*365:.0f}d - {surface.T_points.max()*365:.0f}d")
    if surface.axis_mode == "moneyness":
        print(f"Moneyness range: {surface.y_points.min():.2f} - {surface.y_points.max():.2f}")
    else:
        print(f"Strike range: ${surface.y_points.min():.1f} - ${surface.y_points.max():.1f}")

def validate_ticker(ticker: str) -> str:
    if not ticker or not ticker.strip():
        raise ValueError("Ticker symbol cannot be empty")
    cleaned = ticker.strip().upper()
    if not all(c.isalnum() or c in ('.', '-') for c in cleaned):
        raise ValueError(f"Invalid ticker symbol: '{ticker}'")
    if len(cleaned) > 10:
        raise ValueError(f"Ticker '{cleaned}' is too long")
    return cleaned

def validate_axis(axis: str) -> str:
    valid = ("moneyness", "strike")
    if axis not in valid:
        raise ValueError(f"--axis must be one of {valid}, got '{axis}'")
    return axis

def validate_filter(filter_mode: str) -> str:
    valid = ("add", "OTM", "ITM")
    if filter_mode not in valid:
        raise ValueError(f"--filter must be one of {valid}, got '{filter_mode}'")
    return filter_mode