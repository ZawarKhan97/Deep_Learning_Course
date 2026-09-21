"""Reusable plotting helpers for activations and other 1-D functions.

Figure style (x-range, DPI, colors, ``save_dir``) is loaded from
``configs/plot_config.yaml`` via :func:`load_plot_config`. The main entry
point is :func:`plot_function_and_derivative`, which draws the function and
its derivative in two panels and **always** saves a PNG.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "plot_config.yaml"


def load_plot_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load plot settings from a YAML file.

    Parameters
    ----------
    path :
        Optional path to a YAML config. Defaults to
        ``configs/plot_config.yaml`` under the repository root.

    Returns
    -------
    dict
        Parsed configuration. Must contain at least ``x_min``, ``x_max``,
        ``points``, ``figsize``, ``dpi``, and ``save_dir``.

    Raises
    ------
    ValueError
        If required keys are missing or ``x_min >= x_max``.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    required = ("x_min", "x_max", "points", "figsize", "dpi", "save_dir")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Plot config missing keys: {', '.join(missing)}")
    if cfg["x_min"] >= cfg["x_max"]:
        raise ValueError("plot config: x_min must be less than x_max")
    return cfg


def sample_x(config: dict[str, Any]) -> np.ndarray:
    """Build the evaluation grid from plot config.

    Parameters
    ----------
    config :
        Dict with ``x_min``, ``x_max``, and ``points``.

    Returns
    -------
    numpy.ndarray
        1-D linearly spaced sample points.
    """
    return np.linspace(config["x_min"], config["x_max"], int(config["points"]))


def resolve_save_dir(config: dict[str, Any], base_dir: Path) -> Path:
    """Resolve ``save_dir`` from config and ensure it exists.

    Parameters
    ----------
    config :
        Plot config containing ``save_dir``.
    base_dir :
        Directory used when ``save_dir`` is relative (usually the calling
        script's folder).

    Returns
    -------
    pathlib.Path
        Absolute directory path (created if needed).
    """
    save_dir = Path(config["save_dir"])
    if not save_dir.is_absolute():
        save_dir = base_dir / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def plot_function_and_derivative(
    x: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    *,
    title: str,
    ylabel_f: str,
    ylabel_df: str,
    filename: str,
    config: dict[str, Any],
    save_dir: Path,
) -> Path:
    """Plot a function and its derivative in two panels and always save.

    Parameters
    ----------
    x :
        Sample locations (1-D).
    y :
        Function values ``f(x)``.
    dy :
        Derivative values ``f'(x)``.
    title :
        Base name used in panel titles (e.g. ``\"ReLU\"``).
    ylabel_f :
        Y-axis label for the top (function) panel.
    ylabel_df :
        Y-axis label for the bottom (derivative) panel.
    filename :
        Output file name inside ``save_dir`` (e.g. ``\"relu.png\"``).
    config :
        Style dict from :func:`load_plot_config` (figsize, colors, dpi, …).
    save_dir :
        Directory where the PNG is written (always created/used).

    Returns
    -------
    pathlib.Path
        Full path of the saved figure.

    Notes
    -----
    Only **subplot** titles are used (function on top, derivative below).
    A figure-level ``suptitle`` is intentionally omitted so it cannot overlap
    the top panel title after ``tight_layout``.
    """
    figsize = tuple(config.get("figsize", [8, 6]))
    linewidth = float(config.get("linewidth", 2))
    grid_alpha = float(config.get("grid_alpha", 0.3))
    axis_color = config.get("axis_color", "gray")
    function_color = config.get("function_color", "C0")
    derivative_color = config.get("derivative_color", "C1")
    dpi = int(config.get("dpi", 150))
    show = bool(config.get("show", True))

    fig, (ax_f, ax_df) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax_f.plot(x, y, color=function_color, linewidth=linewidth)
    ax_f.axhline(0, color=axis_color, linewidth=0.8, linestyle="--")
    ax_f.axvline(0, color=axis_color, linewidth=0.8, linestyle="--")
    ax_f.set_ylabel(ylabel_f)
    ax_f.set_title(f"{title} Function")
    ax_f.grid(True, alpha=grid_alpha)

    ax_df.plot(x, dy, color=derivative_color, linewidth=linewidth)
    ax_df.axhline(0, color=axis_color, linewidth=0.8, linestyle="--")
    ax_df.axvline(0, color=axis_color, linewidth=0.8, linestyle="--")
    ax_df.set_xlabel(r"$x$")
    ax_df.set_ylabel(ylabel_df)
    ax_df.set_title(f"Derivative of {title}")
    ax_df.grid(True, alpha=grid_alpha)

    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if show and os.environ.get("DISPLAY"):
        plt.show()
    else:
        plt.close(fig)

    return out_path
