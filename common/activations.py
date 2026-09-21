"""Activation functions and their derivatives.

This module is the shared **math library** for the course. Define each
activation and its derivative once, register it, then reuse it from any
assignment script or notebook.

Generate an API manual from these docstrings with tools such as::

    pdoc -o docs common
    # or
    sphinx-apidoc / pydoctor / doxygen (with EXTRACT_ALL)

Example
-------
>>> import numpy as np
>>> from common.activations import relu, relu_derivative, get_activation
>>> x = np.array([-1.0, 0.0, 2.0])
>>> relu(x)
array([0., 0., 2.])
>>> relu_derivative(x)
array([0., 0., 1.])
>>> get_activation("sigmoid").f(x)
array([0.26894142, 0.5       , 0.88079708])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ArrayFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Activation:
    """Container for one activation and the metadata used when plotting it.

    Attributes
    ----------
    name :
        Short registry key, e.g. ``\"relu\"`` (used by the CLI).
    f :
        Callable that evaluates the activation on an array ``x``.
    df :
        Callable that evaluates the derivative on an array ``x``.
    title :
        Human-readable name shown on figures (e.g. ``\"ReLU\"``).
    ylabel_f :
        Y-axis label for the function panel (may include Matplotlib mathtext).
    ylabel_df :
        Y-axis label for the derivative panel.
    filename :
        Output PNG name when saving a figure (e.g. ``\"relu.png\"``).
    """

    name: str
    f: ArrayFn
    df: ArrayFn
    title: str
    ylabel_f: str
    ylabel_df: str
    filename: str


ACTIVATIONS: dict[str, Activation] = {}
"""Global registry mapping ``name -> Activation``. Prefer :func:`register`."""


def register(activation: Activation) -> Activation:
    """Add an activation to the shared registry.

    Parameters
    ----------
    activation :
        Fully populated :class:`Activation` instance.

    Returns
    -------
    Activation
        The same object, for convenient chaining.

    Notes
    -----
    To introduce a new activation in the course codebase:

    1. Implement ``my_act`` and ``my_act_derivative``.
    2. Call :func:`register` with an :class:`Activation` that points at them.
    3. The CLI will pick it up automatically via :func:`list_activations`.
    """
    ACTIVATIONS[activation.name] = activation
    return activation


def get_activation(name: str) -> Activation:
    """Look up a registered activation by name (case-insensitive).

    Parameters
    ----------
    name :
        Registry key such as ``\"relu\"``, ``\"sigmoid\"``, or ``\"tanh\"``.

    Returns
    -------
    Activation
        Matching entry from :data:`ACTIVATIONS`.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    key = name.lower()
    if key not in ACTIVATIONS:
        known = ", ".join(sorted(ACTIVATIONS))
        raise KeyError(f"Unknown activation '{name}'. Known: {known}")
    return ACTIVATIONS[key]


def list_activations() -> list[str]:
    """Return the names of all registered activations, in registration order.

    Returns
    -------
    list of str
        Keys present in :data:`ACTIVATIONS`.
    """
    return list(ACTIVATIONS.keys())


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit (ReLU).

    Definition
    ----------
    .. math::

        \\mathrm{ReLU}(x) = \\max(0, x) =
        \\begin{cases}
            0, & x < 0 \\\\
            x, & x \\ge 0
        \\end{cases}

    Parameters
    ----------
    x :
        Input array (any shape).

    Returns
    -------
    numpy.ndarray
        Element-wise ReLU of ``x``.
    """
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU (unit step; subgradient at 0 taken as 0).

    Derivation
    ----------
    Away from the kink at the origin the derivative is elementary:

    * for :math:`x < 0`, ReLU is the constant 0, so the slope is 0;
    * for :math:`x > 0`, ReLU is the identity, so the slope is 1.

    At :math:`x = 0` ReLU is not differentiable in the classical sense. Deep
    learning frameworks conventionally take the **subgradient** value 0
    (equivalently: treat the left derivative as the chosen value). In code
    that is ``(x > 0)``, which is ``False`` (→ 0) at ``x == 0``.

    .. math::

        \\mathrm{ReLU}'(x) =
        \\begin{cases}
            0, & x \\le 0 \\\\
            1, & x > 0
        \\end{cases}

    Parameters
    ----------
    x :
        Input array (any shape).

    Returns
    -------
    numpy.ndarray
        Element-wise derivative with dtype ``float``.
    """
    return (x > 0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid.

    Definition
    ----------
    .. math::

        \\sigma(x) = \\frac{1}{1 + e^{-x}}

    Parameters
    ----------
    x :
        Input array (any shape).

    Returns
    -------
    numpy.ndarray
        Values in ``(0, 1)``.
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the logistic sigmoid.

    Derivation
    ----------
    Write :math:`\\sigma(x) = (1 + e^{-x})^{-1}`. Differentiating with the
    chain rule:

    .. math::

        \\sigma'(x)
          = (1 + e^{-x})^{-2} \\cdot e^{-x}
          = \\frac{e^{-x}}{(1 + e^{-x})^2}
          = \\frac{1}{1 + e^{-x}} \\cdot \\frac{e^{-x}}{1 + e^{-x}}
          = \\sigma(x)\\,(1 - \\sigma(x)).

    The last form is preferred numerically: evaluate :math:`\\sigma(x)` once,
    then multiply by :math:`1 - \\sigma(x)`.

    Parameters
    ----------
    x :
        Input array (any shape).

    Returns
    -------
    numpy.ndarray
        Element-wise :math:`\\sigma(x)(1 - \\sigma(x))`.
    """
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_fn(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent.

    Definition
    ----------
    .. math::

        \\tanh(x) = \\frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
                 = \\frac{\\sinh(x)}{\\cosh(x)}

    Parameters
    ----------
    x :
        Input array (any shape).

    Returns
    -------
    numpy.ndarray
        Values in ``(-1, 1)``.
    """
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of hyperbolic tangent.

    Derivation
    ----------
    Using the quotient rule on :math:`\\tanh x = \\sinh x / \\cosh x`, or the
    standard identity :math:`\\frac{d}{dx}\\tanh x = \\mathrm{sech}^2 x`, one
    obtains

    .. math::

        \\tanh'(x) = 1 - \\tanh^2(x) = \\mathrm{sech}^2(x).

    Implementation evaluates :math:`t = \\tanh(x)` once and returns
    ``1 - t**2``.

    Parameters
    ----------
    x :
        Input array (any shape).

    Returns
    -------
    numpy.ndarray
        Element-wise :math:`1 - \\tanh^2(x)`.
    """
    t = np.tanh(x)
    return 1.0 - t**2


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

register(
    Activation(
        name="relu",
        f=relu,
        df=relu_derivative,
        title="ReLU",
        ylabel_f=r"$\mathrm{ReLU}(x)$",
        ylabel_df=r"$\mathrm{ReLU}'(x)$",
        filename="relu.png",
    )
)
register(
    Activation(
        name="sigmoid",
        f=sigmoid,
        df=sigmoid_derivative,
        title="Sigmoid",
        ylabel_f=r"$\sigma(x)$",
        ylabel_df=r"$\sigma'(x)$",
        filename="sigmoid.png",
    )
)
register(
    Activation(
        name="tanh",
        f=tanh_fn,
        df=tanh_derivative,
        title="Hyperbolic Tangent",
        ylabel_f=r"$\tanh(x)$",
        ylabel_df=r"$\tanh'(x)$",
        filename="tanh.png",
    )
)
