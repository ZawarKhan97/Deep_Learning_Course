# Deep Learning Course

This repository contains assignments and projects for the Deep Learning course
taught by **Dr. Ali Irtaza**, Department of Electrical Engineering,
**Institute of Space Technology**.

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib`, `PyYAML`.

## Shared libraries

| Path | Purpose |
|------|---------|
| [`common/activations.py`](common/activations.py) | Activation math + registry (ReLU, Sigmoid, Tanh, …) |
| [`common/plotting.py`](common/plotting.py) | Shared plot helper (`plot_function_and_derivative`) |
| [`configs/plot_config.yaml`](configs/plot_config.yaml) | x-range, DPI, colors, `save_dir`, `show` |

To add a new activation, register it once in `common/activations.py`. Plot styling
is controlled only via the YAML config — no CLI flags for save/range.

## Assignment 1 — Q5: Activation Functions and Derivatives

| File | Description |
|------|-------------|
| [`Assignment_1/plot_derivatives.py`](Assignment_1/plot_derivatives.py) | CLI that uses the shared libs and always saves figures |
| [`Assignment_1/assignment_1.tex`](Assignment_1/assignment_1.tex) | LaTeX write-up |
| [`Assignment_1/figures/`](Assignment_1/figures/) | Generated PNGs |

### Plot script usage

```bash
cd Assignment_1

python plot_derivatives.py --help
python plot_derivatives.py relu
python plot_derivatives.py sigmoid
python plot_derivatives.py tanh
python plot_derivatives.py all
```

Figures are **always saved** under `Assignment_1/figures/` (see `save_dir` in
[`configs/plot_config.yaml`](configs/plot_config.yaml)). Change `x_min`, `x_max`,
`points`, colors, or DPI there and re-run.

Optional: `python plot_derivatives.py all --config /path/to/custom.yaml`

### Compile the LaTeX document

```bash
cd Assignment_1
pdflatex assignment_1.tex
```
