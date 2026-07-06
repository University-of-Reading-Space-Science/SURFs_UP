"""Framework-neutral application services shared by desktop and web UIs."""

from .models import SimulationRequest
from .codegen import build_generated_code, build_uniform_boundary_code
from .runner import RunResult, run_generated_code

__all__ = [
    "RunResult",
    "SimulationRequest",
    "build_generated_code",
    "build_uniform_boundary_code",
    "run_generated_code",
]
