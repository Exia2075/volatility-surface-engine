from __future__ import annotations
import sys

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