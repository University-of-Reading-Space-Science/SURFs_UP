"""Serializable configuration objects for a SURF simulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class SimulationRequest:
    """A UI-independent description of one simulation."""

    model: dict[str, Any]
    ambient: dict[str, Any]
    cmes: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_mappings(
        cls,
        model: Mapping[str, Any],
        ambient: Mapping[str, Any],
        cmes: list[Mapping[str, Any]] | None = None,
    ) -> "SimulationRequest":
        request = cls(
            model=dict(model),
            ambient=dict(ambient),
            cmes=[dict(cme) for cme in (cmes or [])],
        )
        request.validate()
        return request

    def validate(self) -> None:
        """Reject malformed values before a potentially expensive run starts."""
        required = {
            "solver",
            "rmin",
            "rmax",
            "latitude",
            "simtime_days",
            "start_datetime",
            "cr_num",
            "cr_lon_init_deg",
        }
        missing = sorted(required.difference(self.model))
        if missing:
            raise ValueError(f"Missing model settings: {', '.join(missing)}")
        if float(self.model["rmin"]) >= float(self.model["rmax"]):
            raise ValueError("The inner radial boundary must be smaller than the outer boundary.")
        if float(self.model["simtime_days"]) <= 0:
            raise ValueError("Simulation duration must be positive.")
        if float(self.model.get("dr_rs", 1.5)) <= 0:
            raise ValueError("Radial grid spacing must be positive.")
        if int(self.model.get("nlon", 128)) <= 0:
            raise ValueError("Longitude grid size must be positive.")
        if float(self.model.get("vmax_kms", 3000.0)) <= 0:
            raise ValueError("Maximum grid speed must be positive.")
        if not -90 <= float(self.model["latitude"]) <= 90:
            raise ValueError("Latitude must be between -90 and 90 degrees.")
        if not self.ambient.get("source"):
            raise ValueError("An ambient solar-wind source is required.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly top-level mapping."""
        return asdict(self)
