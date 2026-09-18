"""CPU lock tests for the TaylorSeer baseline (no GPU, no TRELLIS model).

fast-trellis2 is the family's frozen control: the Taylor-series forecast every
HiCache variant is compared against.  These tests pin the cache schedule and
the forecast math on fixed series so any accidental change fails loudly.

Run with::

    python tests/test_taylor_baseline.py
"""
import importlib.util
from pathlib import Path

import torch


def _load_taylor():
    path = Path(__file__).resolve().parents[1] / "taylor_utils_ss" / "__init__.py"
    spec = importlib.util.spec_from_file_location("taylor_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_taylor = _load_taylor()


def _drive(num_steps, velocities, **kwargs):
    """Run the cache schedule over a fixed velocity series.

    Returns (types, forecasts) where forecasts maps taylor-step -> tensor.
    """
    taylor_dic, current = _taylor.taylor_init(num_steps, **kwargs)
    current["layer"] = "final"
    current["module"] = "final"
    types, forecasts = [], {}
    for step in range(num_steps):
        current["step"] = step
        _taylor.taylor_cal_type(taylor_dic, current)
        types.append(current["type"])
        if current["type"] == "full":
            _taylor.derivative_approximation(taylor_dic, current, velocities[step])
        else:
            forecasts[step] = _taylor.taylor_formula(taylor_dic, current)
    return types, forecasts


def test_init_defaults_are_pinned():
    dic, current = _taylor.taylor_init(12)
    assert dic["taylor_interval"] == 3
    assert dic["max_order"] == 1
    assert dic["first_enhance"] == 2
    assert dic["end_enhance"] == 24
    assert dic["taylor_enabled"] is True
    assert current["step"] == 0
    assert current["num_steps"] == 12


def test_schedule_cadence_is_pinned():
    flat = [torch.zeros(2) for _ in range(12)]
    types, _ = _drive(12, flat, taylor_interval=3, max_order=1,
                      first_enhance=2, end_enhance=12)
    assert types == ["full", "full",
                     "taylor", "taylor", "full",
                     "taylor", "taylor", "full",
                     "taylor", "taylor", "full", "taylor"]


def test_order1_taylor_is_exact_on_linear_series():
    a = torch.tensor([1.0, -2.0])
    b = torch.tensor([0.5, 0.25])
    velocities = [a + b * s for s in range(12)]
    _, forecasts = _drive(12, velocities, taylor_interval=3, max_order=1,
                          first_enhance=2, end_enhance=12)
    assert len(forecasts) == 7
    for step, forecast in forecasts.items():
        assert torch.allclose(forecast, a + b * step, atol=1e-6), step


def test_forecast_is_deterministic():
    torch.manual_seed(0)
    velocities = [torch.randn(4) for _ in range(12)]
    _, first = _drive(12, velocities, taylor_interval=3, max_order=1,
                      first_enhance=2, end_enhance=12)
    _, second = _drive(12, velocities, taylor_interval=3, max_order=1,
                       first_enhance=2, end_enhance=12)
    assert first.keys() == second.keys()
    for step in first:
        assert torch.equal(first[step], second[step])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"[PASS] {name}")
