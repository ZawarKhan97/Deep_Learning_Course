#!/usr/bin/env python3
"""CLI: plot activation functions and their derivatives.

Math lives in ``common.activations``; shared plotting in ``common.plotting``.
Figure style and x-range come from ``configs/plot_config.yaml`` (always saved).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script from Assignment_1/ without installing the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.activations import get_activation, list_activations  # noqa: E402
from common.plotting import (  # noqa: E402
    load_plot_config,
    plot_function_and_derivative,
    resolve_save_dir,
    sample_x,
)


def build_parser() -> argparse.ArgumentParser:
    names = list_activations()
    choices = names + ["all"]
    parser = argparse.ArgumentParser(
        prog="plot_derivatives",
        description=(
            "Find and plot activation functions together with their derivatives. "
            "Each function gets its own figure (activation on top, derivative below). "
            "Figures are always saved; ranges and style come from the plot config YAML."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python plot_derivatives.py relu\n"
            "  python plot_derivatives.py sigmoid\n"
            "  python plot_derivatives.py tanh\n"
            "  python plot_derivatives.py all\n"
            "\n"
            "Edit configs/plot_config.yaml to change x-range, DPI, colors, save_dir, etc.\n"
        ),
    )
    parser.add_argument(
        "function",
        type=str.lower,
        choices=choices,
        help=(
            "Activation to plot: "
            + ", ".join(f"'{n}'" for n in names)
            + ", or 'all' (separate figure for each)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to plot YAML config (default: configs/plot_config.yaml).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_plot_config(args.config)
    x = sample_x(config)
    save_dir = resolve_save_dir(config, Path(__file__).resolve().parent)

    names = list_activations() if args.function == "all" else [args.function]
    for name in names:
        act = get_activation(name)
        plot_function_and_derivative(
            x,
            act.f(x),
            act.df(x),
            title=act.title,
            ylabel_f=act.ylabel_f,
            ylabel_df=act.ylabel_df,
            filename=act.filename,
            config=config,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    main()
