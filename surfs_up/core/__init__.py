"""Framework-neutral application services used by the web interface."""

from .models import SimulationRequest
from .plotting import (
    format_datetime_axis_like_surf,
    plot_custom_timeseries,
    plot_radial,
    sample_custom_timeseries,
    timeseries_figsize,
)
from .codegen import build_generated_code, build_uniform_boundary_code
from .runner import RunResult, run_generated_code

__all__ = [
    "RunResult",
    "SimulationRequest",
    "build_generated_code",
    "build_uniform_boundary_code",
    "format_datetime_axis_like_surf",
    "plot_custom_timeseries",
    "plot_radial",
    "run_generated_code",
    "sample_custom_timeseries",
    "timeseries_figsize",
]
