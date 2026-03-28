# cli entry point
# run `python main.py --help` for usage details

import argparse
import sys
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

from models.volatility_surface import VolatilitySurfaceBuilder
from data.data_fetcher import MockDataFetcher, DataFetcher
from visualization.plot_surface import plot_surface, plot_term, plot_smile

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

    # Required / Mock (mutally exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--ticker", type=str,
        help="ticker symbol to fetch live option data for (e.g. AAPL, SPY, TSLA)",
    )
    source.add_argument(
        "--mock", action="store_true",
        help="run in offline demo mode using synthetic option data (no API needed)",
    )

    # Option type
    parser.add_argument(
        "--option-type", type=str, default=config.DEFAULT_OPTION_TYPE,
        choices=["call", "put"],
        help="option type to use (default: %(default)s)",
    )

    # Surface options
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

    # Maturity filtering
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

    # Output
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

def run(args: argparse.Namespace) -> int:
    try:
        if args.mock:
            ticker = "MOCK"
        else:
            ticker = validate_ticker(args.ticker)

        axis = validate_axis(args.axis)
        filter_mode = validate_filter(args.filter_mode)

        min_T = args.min_maturity / 365.25
        max_T = args.max_maturity / 365.25

        if min_T >= max_T:
            raise ValueError("--min-maturity must be less than --max-maturity")
    except ValueError as e:
        print_error(str(e))
        return 1
    
    print_header(ticker, axis, filter_mode)

    # fetch data
    try:
        if args.mock:
            fetcher = MockDataFetcher(ticker="MOCK", S=100.0, base_vol=0.20, skew=-0.10, curvature=0.15)
        else:
            r = args.rate if args.rate is not None else config.DEFAULT_RISK_FREE_RATE
            fetcher = DataFetcher(ticker=ticker, r=r)

        contracts = fetcher.fetch(
            option_type = args.option_type,
            remove_illiquid = config.REMOVE_ILLIQUID,
            min_T = min_T,
            max_T = max_T,
            otm_only = (filter_mode == "OTM"),
            itm_only = (filter_mode == "ITM"),
        )

    except ImportError as e:
        print_error(str(e))
        return 1
    except ValueError as e:
        print_error(f"data fetch failed: {e}")
        return 1
    except Exception as e:
        print_error(f"unknown error while fetching data: {e}")
        return 1
    
    if not contracts:
        print_error(
            "no contracts returned after filtering. "
            "try relaxing --filter, --min-maturity, or --max-maturity"
        )
        return 1
    
    print_success(f"fetched {len(contracts)} contract(s) for {ticker}")

    # build volatility surface
    try:
        builder = VolatilitySurfaceBuilder(axis_mode = axis, grid_size = args.grid_size)
        surface = builder.build(contracts)

    except ValueError as e:
        print_error(f"surface construction failed: {e}")
        return 1
    except Exception as e:
        print_error(f"unknown error building surface: {e}")
        return 1
    
    print_success("volatility surface constructed")
    print_summary(surface)

    if surface.n_failed > 0:
        print_warning(
            f"{surface.n_failed} contract(s) failed to converge: "
            "these are excluded from the surface (Likely illiquid or stale quotes)"
        )

    # plot
    try:
        save_path = args.output if args.save else None
        show = not args.no_show

        plot_surface(surface, save_path=save_path, show=show)

        if args.term_structure:
            ts_path = save_path.replace(".png", "_term_structure.png") if save_path else None
            plot_term(surface, save_path=save_path, show=show)

    except Exception as e:
        print_error(f"plotting failed: {e}")
        return 1

    print_success("OK")
    return 0

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run(args))