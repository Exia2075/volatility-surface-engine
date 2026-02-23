# cli entry point
# run `python main.py --help` for usage details

import argparse
import sys
import traceback
import config

from utils.helpers import (
    print_header,
    print_success,
    print_warning, 
    print_error,
    print_summary,
    validate_ticker,
    validate_axis,
    validate_filter,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vol_surface",
        description="volatility surface engine: visualize implied vol surfaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python main.py --ticker AAPL
  python main.py --ticker SPY --axis moneyness --filter OTM
  python main.py --ticker TSLA --axis strike --filter ITM --save
  python main.py --mock --axis moneyness --save --output demo.png
        """,
    )

    # required / mock
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--ticker", type=str,
        help="ticker symbol to fetch live option data for (e.g. AAPL, SPY, TSLA)",
    )
    source.add_argument(
        "--mock", action="store_true",
        help="run in offline demo mode using synthetic option data (no API needed)",
    )

    # surface options
    parser.add_argument(
        "--axis", type=str, default=config.DEFAULT_AXIS_MODE,
        choices=["moneyness", "strike"],
        help="y-axis of the surface: 'moneyness' (K/S) or 'strike' (default: %(default)s)",
    )
    parser.add_argument(
        "--filter", type=str, default="all",
        choices=["all", "OTM", "ITM"],
        dest="filter_mode",
        help="filter contracts: all, OTM (out-of-the-money), or ITM (default: %(default)s)",
    )
    parser.add_argument(
        "--option-type", type=str, default=config.DEFAULT_OPTION_TYPE,
        choices=["call", "put"],
        help="option type to use (default: %(default)s)",
    )

    # maturity filtering
    parser.add_argument(
        "--min-maturity", type=int, default=config.MIN_T_DAYS,
        metavar="DAYS",
        help="minimum days to expiry (default: %(default)s)",
    )
    parser.add_argument(
        "--max-maturity", type=int, default=config.MAX_T_DAYS,
        metavar="DAYS",
        help="Maximum days to expiry (default: %(default)s)",
    )

    # output
    parser.add_argument(
        "--save", action="store_true",
        help="save the surface plot to a file instead of (or in addition to) displaying it",
    )
    parser.add_argument(
        "--output", type=str, default=config.DEFAULT_SAVE_PATH,
        metavar="FILE",
        help="output file path when --save is used (default: %(default)s)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="don't display the interactive plot window (useful for headless environments)",
    )
    parser.add_argument(
        "--term-structure", action="store_true",
        help="also plot the ATM volatility term structure (2D)",
    )
    parser.add_argument(
        "--grid-size", type=int, default=config.DEFAULT_GRID_SIZE,
        metavar="N",
        help="interpolation grid resolution NxN (default: %(default)s)",
    )

    # rate override
    parser.add_argument(
        "--rate", type=float, default=None,
        metavar="RATE",
        help="override the risk-free rate (e.g. 0.05 for 5%%). fetched from ^IRX if omitted.",
    )

    return parser
