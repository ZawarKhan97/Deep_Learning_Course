#!/usr/bin/env python3
"""CLI: plot binary entropy and Set A / Set B empirical entropy histograms."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from common.entropy import (  # noqa: E402
    binary_entropy,
    empirical_probabilities,
    make_set_a,
    make_set_b,
    shannon_entropy,
)
from common.plotting import (  # noqa: E402
    load_plot_config,
    plot_binary_entropy,
    plot_maxent_gaussian_comparison,
    plot_set_ab_histograms,
    resolve_save_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot_entropy",
        description=(
            "Plot (1) binary entropy H(p) and (2) Set A vs Set B outcome "
            "frequencies with Shannon entropy. Figures are always saved; "
            "settings come from configs/plot_config.yaml (entropy section)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python plot_entropy.py\n"
            "  python plot_entropy.py --config ../configs/plot_config.yaml\n"
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
    args = build_parser().parse_args()
    config = load_plot_config(args.config)
    ent = config.get("entropy", {})

    save_dir = resolve_save_dir(config, Path(__file__).resolve().parent)

    # --- Special case: binary entropy curve ---
    p_min = float(ent.get("p_min", 0.001))
    p_max = float(ent.get("p_max", 0.999))
    points = int(ent.get("points", 1000))
    binary_name = str(ent.get("filename_binary", ent.get("filename", "binary_entropy.png")))
    p = np.linspace(p_min, p_max, points)
    plot_binary_entropy(
        p,
        binary_entropy(p, base=2.0),
        filename=binary_name,
        config=config,
        save_dir=save_dir,
    )

    # --- Set A (peaked / low H) vs Set B (uniform / high H) ---
    values = np.arange(1, 11)
    n = int(ent.get("set_size", 100))
    seed = int(ent.get("seed", 42))
    rng_a = np.random.default_rng(seed)
    rng_b = np.random.default_rng(seed + 1)
    set_a = make_set_a(n=n, values=values, rng=rng_a)
    set_b = make_set_b(n=n, values=values, rng=rng_b)

    probs_a = empirical_probabilities(set_a, values)
    probs_b = empirical_probabilities(set_b, values)
    counts_a = probs_a * n
    counts_b = probs_b * n
    h_a = shannon_entropy(probs_a, base=2.0)
    h_b = shannon_entropy(probs_b, base=2.0)

    sets_name = str(ent.get("filename_sets", "set_ab_entropy.png"))
    plot_set_ab_histograms(
        values,
        counts_a,
        counts_b,
        h_a,
        h_b,
        filename=sets_name,
        config=config,
        save_dir=save_dir,
    )
    print(f"Set A Shannon entropy: {h_a:.4f} bits")
    print(f"Set B Shannon entropy: {h_b:.4f} bits")

    # --- Q4: same μ,σ² — different shapes; Gaussian has max H ---
    q4 = ent.get("q4_plot", {})
    mu = float(q4.get("mu", 0.0))
    sigma = float(q4.get("sigma", 1.0))
    q4_name = str(q4.get("filename", "maxent_gaussian.png"))
    plot_maxent_gaussian_comparison(
        mu,
        sigma,
        filename=q4_name,
        config=config,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
