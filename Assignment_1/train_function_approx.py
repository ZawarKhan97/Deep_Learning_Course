#!/usr/bin/env python3
"""Train the Q1 two-layer network on a target function for several dataset sizes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from common.mlp import (  # noqa: E402
    TARGET_FUNCTIONS,
    TwoLayerWeights,
    degrees_to_radians,
    forward,
    make_dataset,
    mse,
    train_test_split,
    train_two_layer,
)
from common.plotting import (  # noqa: E402
    load_plot_config,
    plot_nn_mse_vs_n,
    plot_nn_test_points,
    plot_nn_time_vs_n,
    plot_nn_train_fits,
    resolve_save_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_function_approx",
        description=(
            "Approximate f(x) with a 1-3-1 tanh/linear network for dataset "
            "sizes 10, 25, 50, 100 (80/20 train/test). Uses a fixed epoch "
            "budget (not MSE→0). Always saves figures."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python train_function_approx.py square\n"
            "  python train_function_approx.py sin\n"
            "  python train_function_approx.py abs_square\n"
            "  python train_function_approx.py all\n"
        ),
    )
    parser.add_argument(
        "function",
        type=str.lower,
        choices=sorted(TARGET_FUNCTIONS.keys()) + ["all"],
        help="Target function, or 'all'.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Plot/YAML config path (default: configs/plot_config.yaml).",
    )
    return parser


def run_one(name: str, config: dict, save_dir: Path) -> None:
    q1 = config.get("q1", {})
    func = TARGET_FUNCTIONS[name]
    sizes = [int(n) for n in q1.get("dataset_sizes", [10, 25, 50, 100])]
    # Per-target domain override (sin uses degrees; square uses plain x).
    domain = q1.get(name, {})
    x_min = float(domain.get("x_min", q1.get("x_min", -2.0)))
    x_max = float(domain.get("x_max", q1.get("x_max", 2.0)))
    train_frac = float(q1.get("train_frac", 0.8))
    epochs = int(q1.get("epochs", 5000))
    lr = float(q1.get("lr", 0.05))
    seed = int(q1.get("seed", 42))

    x_grid = np.linspace(x_min, x_max, int(q1.get("grid_points", 400)))
    y_true = func(x_grid)

    unit = str(domain.get("x_unit", q1.get("x_unit", "x")))
    # For sin: sample/plot in degrees, but feed radians into the network
    # so tanh units do not saturate on |x|~180.
    use_deg2rad = name == "sin" and unit.lower().startswith("deg")
    print(f"Target: {name}  domain=[{x_min}, {x_max}] ({unit}), epochs={epochs}")
    if use_deg2rad:
        print("  sin: converting degree inputs to radians for the network")

    fit_curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    test_points: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    train_mses: list[float] = []
    test_mses: list[float] = []
    times_s: list[float] = []
    weights_by_n: dict[str, dict[str, float]] = {}

    def net_x(x_plot: np.ndarray) -> np.ndarray:
        return degrees_to_radians(x_plot) if use_deg2rad else x_plot

    for n in sizes:
        x, y = make_dataset(func, n, x_min=x_min, x_max=x_max, seed=seed + n)
        x_tr, y_tr, x_te, y_te = train_test_split(
            x, y, train_frac=train_frac, seed=seed + 1000 + n
        )
        init = TwoLayerWeights.random(rng=np.random.default_rng(seed + 7 + n))
        t0 = time.perf_counter()
        trained, _ = train_two_layer(
            net_x(x_tr),
            y_tr,
            weights=init,
            lr=lr,
            epochs=epochs,
            rng=np.random.default_rng(seed),
        )
        elapsed = time.perf_counter() - t0
        times_s.append(elapsed)
        weights_by_n[str(n)] = {
            "w1": trained.w1,
            "b1": trained.b1,
            "w2": trained.w2,
            "b2": trained.b2,
            "w3": trained.w3,
            "b3": trained.b3,
            "W1": trained.W1,
            "W2": trained.W2,
            "W3": trained.W3,
            "b_out": trained.b_out,
        }

        y_hat_grid = forward(net_x(x_grid), trained)
        y_hat_te = forward(net_x(x_te), trained)
        fit_curves[n] = (x_grid, y_hat_grid)
        test_points[n] = (x_te, y_hat_te, y_te)

        tr_mse = mse(forward(net_x(x_tr), trained), y_tr)
        te_mse = mse(y_hat_te, y_te)
        train_mses.append(tr_mse)
        test_mses.append(te_mse)
        print(
            f"  N={n:3d}  train_MSE={tr_mse:.6f}  test_MSE={te_mse:.6f}  "
            f"time={elapsed:.3f}s"
        )

    titles = {
        "square": r"$f(x)=x^{2}$",
        "sin": r"$f(x)=\sin(x)$",
        "abs_square": r"$f(x)=|x|^{2}$",
    }
    label = titles[name]

    plot_nn_train_fits(
        x_grid,
        y_true,
        fit_curves,
        title=f"Training fits — {label}",
        filename=f"q1_train_{name}.png",
        config=config,
        save_dir=save_dir,
    )
    plot_nn_test_points(
        x_grid,
        y_true,
        test_points,
        title=f"Test predictions — {label}",
        filename=f"q1_test_{name}.png",
        config=config,
        save_dir=save_dir,
    )
    # Keep legacy fit_*_errors name the user asked for, plus q1_* aliases.
    plot_nn_mse_vs_n(
        sizes,
        train_mses,
        test_mses,
        title=rf"Train/test MSE vs dataset size — {label}",
        filename=f"fit_{name}_errors.png",
        config=config,
        save_dir=save_dir,
    )
    plot_nn_time_vs_n(
        sizes,
        times_s,
        title=rf"Training time vs dataset size — {label}",
        filename=f"fit_{name}_time.png",
        config=config,
        save_dir=save_dir,
    )

    weights_path = save_dir / f"q1_weights_{name}.json"
    with weights_path.open("w", encoding="utf-8") as fh:
        json.dump({"sizes": sizes, "weights": weights_by_n}, fh, indent=2)
    print(f"Saved: {weights_path}")

    # LaTeX table fragment for inclusion in the write-up
    keys = ["w1", "b1", "w2", "b2", "w3", "b3", "W1", "W2", "W3", "b_out"]
    tex_path = save_dir / f"q1_weights_{name}.tex"
    lines = [
        r"\begin{tabular}{l" + "c" * len(sizes) + "}",
        r"\hline",
        r"Parameter & " + " & ".join(rf"$N={n}$" for n in sizes) + r" \\",
        r"\hline",
    ]
    for key in keys:
        cells = [f"{weights_by_n[str(n)][key]:.4f}" for n in sizes]
        lines.append(rf"${key}$ & " + " & ".join(cells) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {tex_path}")


def main() -> None:
    args = build_parser().parse_args()
    config = load_plot_config(args.config)
    save_dir = resolve_save_dir(config, Path(__file__).resolve().parent)
    names = list(TARGET_FUNCTIONS.keys()) if args.function == "all" else [args.function]
    for name in names:
        run_one(name, config, save_dir)


if __name__ == "__main__":
    main()
