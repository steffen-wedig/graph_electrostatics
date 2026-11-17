"""The Taylor-series branch of the smeared-Coulomb kernels by downward recursion.

_smeared_coulomb_kernels_series sums the Boys-function series F_N(x) of the highest
order only (Horner) and obtains the lower orders from F_n = (exp(-x) + 2x F_{n+1}) /
(2n + 1). These tests pin it against a direct term-by-term summation and against
exact Boys-function values (values, first and second derivatives, both width
conventions), and check that the dispatched kernels have finite derivatives at zero
distance and around the crossover.
"""

import math

import pytest
import scipy.special
import torch

from graph_longrange.realspace_electrostatics import (
    SERIES_CROSSOVER,
    SERIES_NUM_TERMS,
    _smeared_coulomb_kernels_series,
    smeared_coulomb_kernels,
)

SCALAR_WIDTH = 1.5
WIDTH_VALUES = [0.7, 1.0, 1.5, 2.3]

# Float64 rounding level. The partial sums are O(1) and every term is at most
# crossover^k / k! <= 0.5, so the two summation orders differ by a few ulp.
VALUE_TOLERANCE = {"rtol": 1e-13, "atol": 1e-15}
# Derivatives of the truncated series: d^m/dx^m F_n is a series of F_{n+m} cut after
# SERIES_NUM_TERMS - m terms, so the truncation error grows with m. The direct
# summation and the recursion truncate different series and differ by that error.
DERIVATIVE_TOLERANCE = {
    1: {"rtol": 1e-12, "atol": 1e-15},
    2: {"rtol": 2e-11, "atol": 1e-15},
}
SQRT_TWO_OVER_PI = math.sqrt(2.0 / math.pi)


def exact_boys_function(order, scaled_distance_squared):
    """F_n(x) = gamma(n + 1/2) P(n + 1/2, x) / (2 x^(n + 1/2)), for x > 0."""
    x = scaled_distance_squared.numpy()
    values = (
        scipy.special.gammainc(order + 0.5, x)
        * scipy.special.gamma(order + 0.5)
        / (2.0 * x ** (order + 0.5))
    )
    return torch.from_numpy(values)


def reference_series(scaled_distance_squared, combined_smearing_width, highest_order):
    """Direct term-by-term summation of the series of every order."""
    accumulators = [
        torch.zeros_like(scaled_distance_squared) for _ in range(highest_order + 1)
    ]
    term = torch.ones_like(scaled_distance_squared)
    for k in range(SERIES_NUM_TERMS):
        for order in range(highest_order + 1):
            accumulators[order] = accumulators[order] + term / (2 * order + 2 * k + 1)
        term = term * (-scaled_distance_squared) / (k + 1)

    kernels = []
    width_power = 1.0 / combined_smearing_width
    sqrt_two_over_pi = (2.0 / torch.pi) ** 0.5
    for order in range(highest_order + 1):
        kernels.append(sqrt_two_over_pi * width_power * accumulators[order])
        width_power = width_power / combined_smearing_width**2
    return kernels


def scaled_distance_grid():
    """0 .. SERIES_CROSSOVER including both ends and points just next to them."""
    grid = torch.linspace(0.0, SERIES_CROSSOVER, 101, dtype=torch.float64)
    extra = torch.tensor(
        [1e-300, 1e-12, 1e-6, SERIES_CROSSOVER - 1e-9], dtype=torch.float64
    )
    return torch.sort(torch.cat([grid, extra])).values


def width_cases():
    """(x shape transform, width): scalar width on [n], tensor width on [n, 1]."""
    return [
        pytest.param(lambda x: x, SCALAR_WIDTH, id="scalar_width"),
        pytest.param(
            lambda x: x.unsqueeze(-1),
            torch.tensor(WIDTH_VALUES, dtype=torch.float64),
            id="broadcast_width",
        ),
    ]


@pytest.mark.parametrize("highest_order", [0, 1, 2])
@pytest.mark.parametrize("reshape,width", width_cases())
def test_series_matches_reference_loop(highest_order, reshape, width):
    scaled = reshape(scaled_distance_grid())
    kernels = _smeared_coulomb_kernels_series(scaled, width, highest_order)
    expected = reference_series(scaled, width, highest_order)
    assert len(kernels) == highest_order + 1
    for kernel, expected_kernel in zip(kernels, expected, strict=True):
        assert kernel.shape == expected_kernel.shape
        assert kernel.dtype == torch.float64
        torch.testing.assert_close(kernel, expected_kernel, **VALUE_TOLERANCE)


@pytest.mark.parametrize("highest_order", [0, 1, 2])
@pytest.mark.parametrize("reshape,width", width_cases())
def test_series_derivatives_match_reference_loop(highest_order, reshape, width):
    """First and second derivatives in x, through create_graph=True as in force training."""
    grid = scaled_distance_grid()
    scaled = reshape(grid.clone()).requires_grad_(True)
    scaled_reference = reshape(grid.clone()).requires_grad_(True)
    kernels = _smeared_coulomb_kernels_series(scaled, width, highest_order)
    expected = reference_series(scaled_reference, width, highest_order)

    for kernel, expected_kernel in zip(kernels, expected, strict=True):
        derivatives = []
        for inputs, output in ((scaled, kernel), (scaled_reference, expected_kernel)):
            first = torch.autograd.grad(
                output.sum(), inputs, create_graph=True, retain_graph=True
            )[0]
            second = torch.autograd.grad(first.sum(), inputs, retain_graph=True)[0]
            derivatives.append((first, second))
        (first, second), (expected_first, expected_second) = derivatives
        assert torch.isfinite(first).all() and torch.isfinite(second).all()
        torch.testing.assert_close(first, expected_first, **DERIVATIVE_TOLERANCE[1])
        torch.testing.assert_close(second, expected_second, **DERIVATIVE_TOLERANCE[2])


@pytest.mark.parametrize("highest_order", [0, 1, 2])
def test_series_matches_exact_boys_function(highest_order):
    """Values and derivatives against F_n, using dF_n/dx = -F_{n+1}, d2F_n/dx2 = F_{n+2}."""
    grid = scaled_distance_grid()
    grid = grid[grid > 1e-6]  # the closed form of the reference divides by x^(n + 1/2)
    scaled = grid.clone().requires_grad_(True)
    kernels = _smeared_coulomb_kernels_series(scaled, 1.0, highest_order)
    for order, kernel in enumerate(kernels):
        boys_value = kernel / SQRT_TWO_OVER_PI
        first = torch.autograd.grad(
            boys_value.sum(), scaled, create_graph=True, retain_graph=True
        )[0]
        second = torch.autograd.grad(first.sum(), scaled, retain_graph=True)[0]
        torch.testing.assert_close(
            boys_value.detach(), exact_boys_function(order, grid), **VALUE_TOLERANCE
        )
        torch.testing.assert_close(
            first, -exact_boys_function(order + 1, grid), **DERIVATIVE_TOLERANCE[1]
        )
        torch.testing.assert_close(
            second, exact_boys_function(order + 2, grid), **DERIVATIVE_TOLERANCE[2]
        )


def test_series_derivatives_at_zero_are_the_leading_coefficients():
    """d/dx and d^2/dx^2 of the accumulators at x = 0 are c[1, n] and 2 c[2, n]."""
    scaled = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    kernels = _smeared_coulomb_kernels_series(scaled, 1.0, 2)
    sqrt_two_over_pi = SQRT_TWO_OVER_PI
    for order, kernel in enumerate(kernels):
        first = torch.autograd.grad(
            kernel.sum(), scaled, create_graph=True, retain_graph=True
        )[0]
        second = torch.autograd.grad(first.sum(), scaled, retain_graph=True)[0]
        expected_first = -sqrt_two_over_pi / (2 * order + 3)
        expected_second = 2.0 * sqrt_two_over_pi / (2.0 * (2 * order + 5))
        torch.testing.assert_close(
            first, torch.full_like(first, expected_first), rtol=1e-15, atol=0.0
        )
        torch.testing.assert_close(
            second, torch.full_like(second, expected_second), rtol=1e-15, atol=0.0
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_series_keeps_dtype_and_scalar_input_shape(dtype):
    scaled = torch.tensor(0.3, dtype=dtype)
    kernels = _smeared_coulomb_kernels_series(scaled, SCALAR_WIDTH, 2)
    expected = reference_series(scaled.double(), SCALAR_WIDTH, 2)
    for kernel, expected_kernel in zip(kernels, expected, strict=True):
        assert kernel.shape == torch.Size([])
        assert kernel.dtype == dtype
        torch.testing.assert_close(
            kernel.double(),
            expected_kernel,
            rtol=1e-6 if dtype == torch.float32 else 1e-13,
            atol=0.0,
        )


@pytest.mark.parametrize("reshape,width", width_cases())
def test_dispatched_kernel_derivatives_finite_at_zero_and_across_crossover(
    reshape, width
):
    scaled_values = torch.tensor(
        [
            0.0,
            1e-12,
            0.25,
            SERIES_CROSSOVER - 1e-9,
            SERIES_CROSSOVER,
            SERIES_CROSSOVER + 1e-9,
            0.75,
            5.0,
        ],
        dtype=torch.float64,
    )
    width_tensor = torch.as_tensor(width, dtype=torch.float64)
    distance_squared = (
        reshape(scaled_values) * 2.0 * width_tensor.min() ** 2
    ).requires_grad_(True)
    kernels = smeared_coulomb_kernels(distance_squared, width, highest_order=2)
    for kernel in kernels:
        assert torch.isfinite(kernel).all()
        first = torch.autograd.grad(
            kernel.sum(), distance_squared, create_graph=True, retain_graph=True
        )[0]
        second = torch.autograd.grad(first.sum(), distance_squared, retain_graph=True)[
            0
        ]
        assert torch.isfinite(first).all()
        assert torch.isfinite(second).all()
