"""Flask application factory using the same core services as the desktop GUI."""

from __future__ import annotations

import datetime
import csv
import inspect
import io
import json
import os
import pickle
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from flask import Flask, abort, jsonify, render_template, request, send_file

from surfs_up.core import (
    SimulationRequest,
    build_generated_code,
    format_datetime_axis_like_surf,
    plot_custom_timeseries,
    plot_radial as plot_radial_profile,
    run_generated_code,
    sample_custom_timeseries,
    timeseries_figsize,
)

_RUNS: OrderedDict[str, object] = OrderedDict()
_RUNS_LOCK = threading.Lock()
_RUN_PROGRESS: OrderedDict[str, str] = OrderedDict()
_RUN_PROGRESS_LOCK = threading.Lock()
_PLOT_LOCK = threading.Lock()
_MAX_RETAINED_RUNS = 8
_RUN_CACHE_DIR = Path(
    os.environ.get("SURFS_UP_RUN_CACHE_DIR", Path.home() / ".cache" / "surfs_up" / "runs")
)
_DONKI_URL = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CMEAnalysis"


def _retain_model(model: object, simulation: SimulationRequest) -> str:
    run_id = uuid.uuid4().hex
    retained = {"model": model, "simulation": simulation}
    with _RUNS_LOCK:
        _RUNS[run_id] = retained
        while len(_RUNS) > _MAX_RETAINED_RUNS:
            _RUNS.popitem(last=False)
    _write_run_cache(run_id, retained)
    return run_id


def _run_for(run_id: str) -> dict[str, object]:
    with _RUNS_LOCK:
        retained = _RUNS.get(run_id)
    if retained is None:
        retained = _read_run_cache(run_id)
    if retained is None:
        abort(404, "Run not found or no longer retained.")
    with _RUNS_LOCK:
        _RUNS[run_id] = retained
        while len(_RUNS) > _MAX_RETAINED_RUNS:
            _RUNS.popitem(last=False)
    if isinstance(retained, dict):
        return retained
    return {"model": retained, "simulation": None}


def _model_for(run_id: str) -> object:
    return _run_for(run_id)["model"]


def _run_cache_path(run_id: str) -> Path:
    if not run_id.isalnum():
        abort(404)
    return _RUN_CACHE_DIR / f"{run_id}.pickle"


def _write_run_cache(run_id: str, retained: dict[str, object]) -> None:
    try:
        _RUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _run_cache_path(run_id)
        with tempfile.NamedTemporaryFile(
            "wb", dir=_RUN_CACHE_DIR, delete=False, prefix=f"{run_id}.", suffix=".tmp"
        ) as handle:
            pickle.dump(retained, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path = Path(handle.name)
        temp_path.replace(path)
        _prune_run_cache()
    except Exception:
        # In-memory retention is still enough for local/single-worker use; a disk
        # cache is only needed when a deployment serves follow-up plot requests
        # from a different Python process.
        pass


def _read_run_cache(run_id: str) -> dict[str, object] | None:
    try:
        path = _run_cache_path(run_id)
        with path.open("rb") as handle:
            retained = pickle.load(handle)
        os.utime(path, None)
        return retained if isinstance(retained, dict) else {"model": retained, "simulation": None}
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _prune_run_cache() -> None:
    cached_runs = sorted(
        _RUN_CACHE_DIR.glob("*.pickle"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    cutoff = time.time() - 24 * 60 * 60
    for path in cached_runs[_MAX_RETAINED_RUNS:]:
        path.unlink(missing_ok=True)
    for path in cached_runs[:_MAX_RETAINED_RUNS]:
        if path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)


def _set_run_progress(progress_id: str, message: str) -> None:
    """Store a short-lived status message for a run being processed."""
    if not progress_id:
        return
    with _RUN_PROGRESS_LOCK:
        _RUN_PROGRESS[progress_id] = message
        while len(_RUN_PROGRESS) > 16:
            _RUN_PROGRESS.popitem(last=False)


def _model_defaults() -> dict[str, object]:
    """Return the same time-dependent defaults initialized by the Qt model tab."""
    import astropy.units as u
    import surf.surf_inputs as sin

    today = datetime.datetime.now(datetime.UTC).replace(tzinfo=None, microsecond=0)
    now = (
        today - datetime.timedelta(days=5)
    ).replace(tzinfo=None, microsecond=0)
    cr_num, cr_lon = sin.datetime2surfinputs(now)
    earth_latitude = sin.get_earth_lat(now)
    return {
        "default_start": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "default_iswa_map_datetime": today.strftime("%Y-%m-%dT%H:%M"),
        "default_cr_num": int(cr_num),
        "default_cr_lon": cr_lon.to_value(u.deg),
        "default_latitude": (
            earth_latitude.to_value(u.deg)
            if hasattr(earth_latitude, "to_value")
            else float(earth_latitude)
        ),
    }


def _float(name: str, default: float) -> float:
    value = request.form.get(name, "").strip()
    return float(value) if value else default


def _save_uploaded_file(uploaded) -> Path:
    upload_dir = Path(tempfile.gettempdir()) / "surfs_up_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / Path(uploaded.filename).name
    uploaded.save(path)
    return path


def _fetch_donki_cmes(
    start: datetime.datetime, duration_days: float
) -> list[dict[str, object]]:
    """Download and normalize DONKI cone CMEs for a model run interval."""
    end = start + datetime.timedelta(days=duration_days)
    query = urlencode(
        {
            "startDate": start.date().isoformat(),
            "endDate": end.date().isoformat(),
            "completeEntryOnly": "true",
            "speed": "0",
            "halfAngle": "0",
            "catalog": "ALL",
        }
    )
    with urlopen(f"{_DONKI_URL}?{query}", timeout=30) as response:
        analyses = json.load(response)
    results = []
    for analysis in analyses:
        launch_text = analysis.get("time21_5")
        if not launch_text:
            continue
        launch = datetime.datetime.fromisoformat(
            str(launch_text).replace("Z", "+00:00")
        ).replace(tzinfo=None)
        if not start <= launch <= end:
            continue
        if any(
            analysis.get(key) is None
            for key in ("longitude", "latitude", "speed", "halfAngle")
        ):
            continue
        results.append(
            {
                "longitude": float(analysis["longitude"]),
                "latitude": float(analysis["latitude"]),
                "speed": float(analysis["speed"]),
                "width": 2 * float(analysis["halfAngle"]),
                "t_launch_day": (launch - start).total_seconds() / 86400,
                "t_launch_datetime": launch.strftime("%Y-%m-%d %H:%M:%S"),
                "thickness_rs": 0,
                "initial_height_rs": 21.5,
                "cme_expansion": False,
                "cme_fixed_duration": True,
                "fixed_duration_hr": 12,
                "profile_type": "square",
                "plasma_mode": "Fraction of ambient",
                "density_fraction": 1,
                "temperature_fraction": 1,
                "source": "donki",
            }
        )
    return results


def _example_input_path(pattern: str, missing_message: str) -> Path:
    import surf

    examples = Path(surf.__file__).resolve().parent / "data" / "example_inputs"
    matches = sorted(examples.glob(pattern))
    if not matches:
        raise ValueError(missing_message)
    return matches[0]


def _parse_wsa_start_time(filepath: Path):
    """Extract WSA map time from FITS metadata when available, else filename."""
    import re

    from astropy.io import fits

    filepath = Path(filepath)

    try:
        header = fits.getheader(filepath)
        for key in ("DATE-OBS", "DATE_OBS", "DATE", "MAPDATE"):
            if key in header:
                value = str(header[key]).strip()
                for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
                    try:
                        return datetime.datetime.strptime(value[:19], fmt)
                    except ValueError:
                        continue
    except Exception:
        pass

    name = filepath.name
    match = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})Z", name)
    if match:
        return datetime.datetime.strptime(
            f"{match.group(1)}T{match.group(2)}", "%Y-%m-%dT%H"
        )

    match = re.search(r"(\d{8})(\d{2})", name)
    if match:
        return datetime.datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H")

    return None


def _parse_cortom_start_time(filepath: Path):
    """Extract CorTom map time from filename."""
    import re

    filepath = Path(filepath)
    match = re.search(r"(\d{14})", filepath.name)
    if match:
        return datetime.datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    return None


def _ambient_file_start_time(source: str, filepath: Path):
    if source == "wsa":
        return _parse_wsa_start_time(filepath)
    if source == "cortom":
        return _parse_cortom_start_time(filepath)
    return None


def _iswa_map_datetime(value: str, fallback: str) -> datetime.datetime:
    """Parse the ISWA WSA map date/time control.

    Accept both date-only values from older forms/configurations and
    ``datetime-local`` values from the current web interface.
    """
    text = (value or fallback or "").strip().replace(" ", "T")
    if not text:
        return datetime.datetime.now(datetime.UTC).replace(tzinfo=None, microsecond=0)
    if "T" not in text:
        text = f"{text}T23:59:59"
    return datetime.datetime.fromisoformat(text)


def _draw_speed_map(ax, speed_map, longitudes, latitudes, extraction_latitude, title):
    import astropy.units as u
    import numpy as np

    speed_values = (
        speed_map.to_value(u.km / u.s)
        if hasattr(speed_map, "to_value")
        else np.asarray(speed_map)
    )
    lon_values = (
        longitudes.to_value(u.deg)
        if hasattr(longitudes, "to_value")
        else np.rad2deg(np.asarray(longitudes))
    )
    lat_values = (
        latitudes.to_value(u.deg)
        if hasattr(latitudes, "to_value")
        else np.rad2deg(np.asarray(latitudes))
    )
    speed_values = np.asarray(speed_values)
    if speed_values.shape == (len(lon_values), len(lat_values)):
        speed_values = speed_values.T
    if speed_values.shape != (len(lat_values), len(lon_values)):
        raise ValueError(
            "Speed map dimensions do not match its longitude and latitude coordinates."
        )

    image = ax.pcolormesh(
        lon_values,
        lat_values,
        speed_values,
        shading="auto",
        cmap="viridis",
    )
    ax.axhline(
        extraction_latitude,
        color="red",
        linewidth=1.8,
        linestyle="--",
        label=f"Extracted latitude: {extraction_latitude:.1f}°",
    )
    ax.set_xlim(float(np.nanmin(lon_values)), float(np.nanmax(lon_values)))
    ax.set_ylim(float(np.nanmin(lat_values)), float(np.nanmax(lat_values)))
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.figure.colorbar(image, ax=ax, label="Speed [km/s]")


def _ambient_preview_figure():
    import astropy.units as u
    import matplotlib
    import numpy as np
    import surf.surf_inputs as sin

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = request.form.get("ambient_source", "user_specified")
    latitude = _float("latitude", 0.0) * u.deg
    include_bpol = "include_bpol" in request.form

    def plot_mas():
        cr_num = int(_float("mas_cr_num", 2000))
        map_to_inner = "mas_decelerate" in request.form
        speed_map, map_longitudes, map_latitudes = sin.get_MAS_vr_map(cr_num)
        v_orig = sin.get_MAS_long_profile(cr_num, latitude)
        if include_bpol:
            b_orig = sin.get_MAS_br_long_profile(cr_num, latitude)
            if len(b_orig) != len(v_orig):
                b_lon = np.linspace(0.0, 360.0, len(b_orig), endpoint=False)
                v_lon = np.linspace(0.0, 360.0, len(v_orig), endpoint=False)
                b_orig = np.interp(v_lon, b_lon, np.asarray(b_orig), period=360.0)
            if map_to_inner:
                mapped = sin.map_v_boundary_inwards(
                    v_orig,
                    30.0 * u.solRad,
                    21.5 * u.solRad,
                    b_orig=b_orig,
                )
                if isinstance(mapped, tuple):
                    v_mapped, b_mapped = mapped
                else:
                    v_mapped = mapped
                    b_mapped = np.ones(len(v_orig)) * np.nan
            else:
                v_mapped = v_orig
                b_mapped = b_orig
        else:
            b_orig = None
            if map_to_inner:
                v_mapped = sin.map_v_boundary_inwards(
                    v_orig,
                    30.0 * u.solRad,
                    21.5 * u.solRad,
                )
            else:
                v_mapped = v_orig

        carr_lon = np.linspace(0.0, 360.0, len(v_orig), endpoint=False)
        if include_bpol:
            fig, (ax_map, ax_v, ax_b) = plt.subplots(3, 1, figsize=(10, 12))
        else:
            fig, (ax_map, ax_v) = plt.subplots(2, 1, figsize=(10, 9))
        _draw_speed_map(
            ax_map,
            speed_map,
            map_longitudes,
            map_latitudes,
            latitude.value,
            f"MAS speed map | CR {cr_num}",
        )
        ax_map.set_xlabel("Carrington longitude [deg]")
        ax_v.plot(carr_lon, v_orig.to_value(u.km / u.s), linewidth=1.5, label="Original at 30 Rs")
        ax_v.plot(
            carr_lon,
            v_mapped.to_value(u.km / u.s),
            linewidth=1.5,
            linestyle="--",
            label=(
                "Mapped to 21.5 Rs" if map_to_inner else "Original (no deceleration mapping)"
            ),
        )
        ax_v.set_xlim(0.0, 360.0)
        ax_v.set_ylabel("Vin [km/s]")
        ax_v.set_title(f"MAS boundary profiles | CR {cr_num} | lat {latitude.value:.1f} deg")
        ax_v.grid(True, alpha=0.3)
        ax_v.legend()
        if include_bpol:
            ax_b.plot(carr_lon, np.asarray(b_orig), linewidth=1.5, label="Original bpol at 30 Rs")
            ax_b.plot(
                carr_lon,
                np.asarray(b_mapped),
                linewidth=1.5,
                linestyle="--",
                label=(
                    "Mapped bpol to 21.5 Rs" if map_to_inner else "Original bpol (no deceleration mapping)"
                ),
            )
            ax_b.set_xlim(0.0, 360.0)
            ax_b.set_xlabel("Carrington longitude [deg]")
            ax_b.set_ylabel("bpol")
            ax_b.grid(True, alpha=0.3)
            ax_b.legend()
        else:
            ax_v.set_xlabel("Carrington longitude [deg]")
        fig.tight_layout()
        return fig

    def plot_file_source(
        title: str,
        speed_map_title: str,
        path: Path,
        source_radius_rs: float,
        profile_loader,
        br_profile_loader,
        speed_map_loader,
        decelerate_key: str,
        reduction_key: str | None = None,
    ):
        map_to_inner = decelerate_key in request.form
        apply_speed_reduction = reduction_key is not None and reduction_key in request.form
        speed_map, map_longitudes, map_latitudes = speed_map_loader(path)
        v_orig = profile_loader(path, latitude)
        if apply_speed_reduction:
            longitude = np.linspace(0.0, 2.0 * np.pi, len(v_orig), endpoint=False) * u.rad
            v_reduced, _ = sin.map_v_inwards(
                v_orig,
                215.0 * u.solRad,
                longitude,
                21.5 * u.solRad,
            )
        else:
            v_reduced = v_orig

        include_bpol_plot = include_bpol and (br_profile_loader is not None)
        if include_bpol_plot:
            b_orig = br_profile_loader(path, latitude)
            if map_to_inner:
                mapped = sin.map_v_boundary_inwards(
                    v_reduced,
                    source_radius_rs * u.solRad,
                    21.5 * u.solRad,
                    b_orig=b_orig,
                )
                if isinstance(mapped, tuple):
                    v_mapped, b_mapped = mapped
                else:
                    v_mapped = mapped
                    b_mapped = np.ones(len(v_orig)) * np.nan
            else:
                v_mapped = v_reduced
                b_mapped = b_orig
        else:
            if map_to_inner:
                v_mapped = sin.map_v_boundary_inwards(
                    v_reduced,
                    source_radius_rs * u.solRad,
                    21.5 * u.solRad,
                )
            else:
                v_mapped = v_reduced

        carr_lon = np.linspace(0.0, 360.0, len(v_orig), endpoint=False)
        include_speed_map = speed_map_loader is not None
        if include_speed_map and include_bpol_plot:
            fig, (ax_map, ax_v, ax_b) = plt.subplots(3, 1, figsize=(10, 12))
        elif include_speed_map:
            fig, (ax_map, ax_v) = plt.subplots(2, 1, figsize=(10, 9))
        elif include_bpol_plot:
            fig, (ax_v, ax_b) = plt.subplots(2, 1, sharex=True)
        else:
            fig, ax_v = plt.subplots()
        if include_speed_map:
            _draw_speed_map(
                ax_map,
                speed_map,
                map_longitudes,
                map_latitudes,
                latitude.value,
                speed_map_title,
            )
            ax_map.set_xlabel("Carrington longitude [deg]")
        ax_v.plot(
            carr_lon,
            v_orig.to_value(u.km / u.s),
            linewidth=1.5,
            label=f"Original at {source_radius_rs:.1f} Rs",
        )
        if apply_speed_reduction:
            ax_v.plot(
                carr_lon,
                v_reduced.to_value(u.km / u.s),
                linewidth=1.5,
                linestyle="-.",
                label="WSA speed reduction: 215 to 21.5 Rs (longitude unchanged)",
            )
        ax_v.plot(
            carr_lon,
            v_mapped.to_value(u.km / u.s),
            linewidth=1.5,
            linestyle="--",
            label=(
                "Mapped to 21.5 Rs"
                if map_to_inner
                else ("Speed-reduced boundary" if apply_speed_reduction else "Original (no deceleration mapping)")
            ),
        )
        ax_v.set_xlim(0.0, 360.0)
        ax_v.set_ylabel("Vin [km/s]")
        ax_v.grid(True, alpha=0.3)
        ax_v.legend()
        if include_bpol_plot:
            ax_b.plot(
                carr_lon,
                np.asarray(b_orig),
                linewidth=1.5,
                label=f"Original bpol at {source_radius_rs:.1f} Rs",
            )
            ax_b.plot(
                carr_lon,
                np.asarray(b_mapped),
                linewidth=1.5,
                linestyle="--",
                label=(
                    "Mapped bpol to 21.5 Rs" if map_to_inner else "Original bpol (no deceleration mapping)"
                ),
            )
            ax_b.set_xlim(0.0, 360.0)
            ax_b.set_xlabel("Carrington longitude [deg]")
            ax_b.set_ylabel("bpol")
            ax_b.grid(True, alpha=0.3)
            ax_b.legend()
        else:
            ax_v.set_xlabel("Carrington longitude [deg]")
        fig.tight_layout()
        return fig

    if source == "mas":
        return plot_mas()
    if source == "wsa":
        uploaded = request.files.get("wsa_file")
        path = _save_uploaded_file(uploaded) if uploaded and uploaded.filename else _example_input_path(
            "**/*.fits",
            "Upload a WSA input file.",
        )
        return plot_file_source(
            "WSA boundary profiles",
            f"WSA speed map | {Path(path).name}",
            path,
            21.5,
            sin.get_WSA_long_profile,
            sin.get_WSA_br_long_profile,
            lambda selected_path: sin.get_WSA_maps(selected_path)[:3],
            "wsa_decelerate",
            "wsa_speed_reduction",
        )
    if source == "wsa_iswa":
        required_for = _iswa_map_datetime(
            request.form.get("iswa_map_date", ""),
            request.form.get("start_datetime", ""),
        )
        path = sin.get_WSA_from_ISWA(required_for)
        return plot_file_source(
            "WSA boundary profiles",
            f"WSA speed map | {Path(path).name}",
            Path(path),
            21.5,
            sin.get_WSA_long_profile,
            sin.get_WSA_br_long_profile,
            lambda selected_path: sin.get_WSA_maps(selected_path)[:3],
            "iswa_decelerate",
            "iswa_speed_reduction",
        )
    if source == "cortom":
        uploaded = request.files.get("cortom_file")
        path = _save_uploaded_file(uploaded) if uploaded and uploaded.filename else _example_input_path(
            "**/*.dat",
            "Upload a CORTOM input file.",
        )
        return plot_file_source(
            "CorTom boundary profiles",
            f"CorTom speed map | {Path(path).name}",
            path,
            8.0,
            sin.get_CorTom_long_profile,
            None,
            sin.get_CorTom_vr_map,
            "cortom_decelerate",
            None,
        )
    raise ValueError("Select MAS, WSA, WSA (ISWA), or CorTom before plotting.")


def _request_from_form() -> SimulationRequest:
    start = request.form.get("start_datetime") or datetime.datetime.now(
        datetime.UTC
    ).strftime("%Y-%m-%d %H:%M:%S")
    source = request.form.get("ambient_source", "user_specified")
    speed = _float("speed_kms", 400.0)
    ambient = {"source": source}
    if source == "user_specified":
        profile_text = request.form.get("speed_profile", "").strip()
        ambient["speed_profile_kms"] = (
            [float(value) for value in profile_text.split(",") if value.strip()]
            if profile_text
            else [speed] * 128
        )
    elif source == "mas":
        ambient.update(
            cr_num=int(_float("mas_cr_num", 2300)),
            decelerate_to_inner_boundary="mas_decelerate" in request.form,
        )
        if "mas_use_map_time" in request.form:
            from sunpy.coordinates import sun

            map_time = sun.carrington_rotation_time(
                float(_float("mas_cr_num", 2300))
            ).to_datetime()
            if hasattr(map_time, "item"):
                map_time = map_time.item()
            if map_time.tzinfo is not None:
                map_time = map_time.replace(tzinfo=None)
            start = map_time.strftime("%Y-%m-%d %H:%M:%S")
    elif source == "wsa":
        uploaded = request.files.get("wsa_file")
        if uploaded and uploaded.filename:
            path = _save_uploaded_file(uploaded)
        else:
            path = _example_input_path("**/*.fits", "Upload a WSA input file.")
        ambient.update(
            filepath=str(path),
            decelerate_to_inner_boundary="wsa_decelerate" in request.form,
            apply_wsa_speed_reduction="wsa_speed_reduction" in request.form,
        )
        if "wsa_use_map_time" in request.form:
            map_time = _ambient_file_start_time(source, path)
            if map_time is not None:
                start = map_time.strftime("%Y-%m-%d %H:%M:%S")
    elif source == "wsa_iswa":
        iswa_datetime = _iswa_map_datetime(request.form.get("iswa_map_date", ""), start)
        ambient.update(
            decelerate_to_inner_boundary="iswa_decelerate" in request.form,
            apply_wsa_speed_reduction="iswa_speed_reduction" in request.form,
            iswa_map_datetime=iswa_datetime.isoformat(),
        )
    elif source == "cortom":
        uploaded = request.files.get("cortom_file")
        if uploaded and uploaded.filename:
            path = _save_uploaded_file(uploaded)
        else:
            path = _example_input_path("**/*.dat", "Upload a CORTOM input file.")
        ambient.update(
            filepath=str(path),
            decelerate_to_inner_boundary="cortom_decelerate" in request.form,
        )
        if "cortom_use_map_time" in request.form:
            map_time = _ambient_file_start_time(source, path)
            if map_time is not None:
                start = map_time.strftime("%Y-%m-%d %H:%M:%S")
    elif source == "insitu_backmapped":
        ambient["mode"] = request.form.get("insitu_mode", "forecast")
    elif source == "omni":
        ambient["use_215_inner_boundary"] = "use_215_inner_boundary" in request.form

    cmes_text = request.form.get("cmes_json", "").strip()
    cmes = json.loads(cmes_text) if cmes_text else []
    if not isinstance(cmes, list):
        raise ValueError("CME JSON must contain a list of CME objects.")
    if (
        request.form.get("action") == "run"
        and "grab_donki_at_run_start" in request.form
        and abs(_float("rmin", 21.5) - 21.5) <= 1.0e-9
    ):
        model_start = datetime.datetime.fromisoformat(start.replace("T", " "))
        cmes = [cme for cme in cmes if cme.get("source") != "donki"]
        cmes.extend(
            _fetch_donki_cmes(model_start, _float("simtime_days", 10.0))
        )
    cone_file = request.files.get("cone_file")
    if cone_file and cone_file.filename:
        import numpy as np
        import surf.surf_inputs as sin
        from astropy.time import Time

        upload_dir = Path(tempfile.gettempdir()) / "surfs_up_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        cone_path = upload_dir / Path(cone_file.filename).name
        cone_file.save(cone_path)
        model_start = datetime.datetime.fromisoformat(start.replace("T", " "))
        for cone in sin.import_cone2bc_parameters(str(cone_path)).values():
            launch = Time(cone["ldates"]).to_datetime().replace(tzinfo=None)
            cmes.append(
                {
                    "longitude": float(cone.get("lon", 0)),
                    "latitude": float(cone.get("lat", 0)),
                    "speed": float(cone.get("vcld", 800)),
                    "width": float(2 * cone.get("rmajor", 30)),
                    "t_launch_day": (launch - model_start).total_seconds() / 86400,
                    "thickness_rs": 0,
                    "initial_height_rs": 21.5,
                    "cme_expansion": False,
                    "cme_fixed_duration": True,
                    "fixed_duration_hr": 12,
                    "profile_type": "square",
                    "plasma_mode": "Fraction of ambient",
                    "density_fraction": 1,
                    "temperature_fraction": 1,
                    "cme_density_pcc": np.nan,
                    "cme_temperature_k": np.nan,
                }
            )
    return SimulationRequest.from_mappings(
        {
            "solver": request.form.get("solver", "huxt"),
            "rmin": _float("rmin", 21.5),
            "rmax": _float("rmax", 240.0),
            "lon_min": _float("lon_min", 315.0),
            "lon_max": _float("lon_max", 45.0),
            "latitude": _float("latitude", 0.0),
            "is_1d": "is_1d" in request.form,
            "frame": request.form.get("frame", "sidereal"),
            "include_bpol": "include_bpol" in request.form,
            "track_cmes": "track_cmes" in request.form,
            "streak_lines_enabled": "streak_lines_enabled" in request.form,
            "streak_spacing_deg": _float("streak_spacing_deg", 10.0),
            "simtime_days": _float("simtime_days", 10.0),
            "dr_rs": _float("dr_rs", 1.5),
            "nlon": int(_float("nlon", 128)),
            "vmax_kms": _float("vmax_kms", 3000.0),
            "start_datetime": start.replace("T", " "),
            "cr_num": int(_float("cr_num", 2300)),
            "cr_lon_init_deg": _float("cr_lon_init_deg", 0.0),
        },
        ambient,
        cmes,
    )


def create_app(config: dict | None = None) -> Flask:
    """Create an app suitable for local use or a PythonAnywhere WSGI file."""
    app = Flask(__name__)
    app.config.from_mapping(MAX_CONTENT_LENGTH=1_000_000)
    if config:
        app.config.update(config)

    @app.get("/model-coordinates")
    def model_coordinates():
        """Convert between model UTC time and Carrington coordinates."""
        import astropy.units as u
        import surf.surf_inputs as sin
        from sunpy.coordinates import sun

        datetime_text = request.args.get("datetime", "").strip()
        if datetime_text:
            model_time = datetime.datetime.fromisoformat(datetime_text.replace("T", " "))
        else:
            cr_num = float(request.args["cr_num"])
            cr_lon = float(request.args["cr_lon"])
            cr_fraction = cr_num + ((360.0 - cr_lon) / 360.0)
            model_time = sun.carrington_rotation_time(cr_fraction).to_datetime()
            if hasattr(model_time, "item"):
                model_time = model_time.item()
            if model_time.tzinfo is not None:
                model_time = model_time.replace(tzinfo=None)

        cr_num, cr_lon = sin.datetime2surfinputs(model_time)
        earth_latitude = sin.get_earth_lat(model_time)
        return jsonify(
            {
                "datetime": model_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "cr_num": int(cr_num),
                "cr_lon": float(cr_lon.to_value(u.deg)),
                "earth_latitude": float(
                    earth_latitude.to_value(u.deg)
                    if hasattr(earth_latitude, "to_value")
                    else earth_latitude
                ),
            }
        )

    @app.post("/ambient-file-time")
    def ambient_file_time():
        """Infer the start time from a selected ambient file."""
        source = request.form.get("source", "")
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename:
            abort(400, "Upload a file to infer its timestamp.")

        path = _save_uploaded_file(uploaded)
        map_time = _ambient_file_start_time(source, path)
        if map_time is None:
            return jsonify({"datetime": None})
        return jsonify({"datetime": map_time.strftime("%Y-%m-%dT%H:%M:%S")})

    @app.route("/", methods=["GET", "POST"])
    def index():
        context = {
            "code": None,
            "error": None,
            "result": None,
            "run_id": None,
            "show_code_dialog": False,
        }
        if request.method == "POST":
            try:
                simulation = _request_from_form()
                context["code"] = build_generated_code(simulation)
                action = request.form.get("action")
                context["show_code_dialog"] = action == "preview"
                if action == "run":
                    progress_id = request.form.get("progress_id", "")
                    _set_run_progress(progress_id, "Grabbing and processing input data")
                    context["result"] = run_generated_code(
                        context["code"],
                        before_solve=lambda: _set_run_progress(
                            progress_id, "Running SURF"
                        ),
                    )
                    if context["result"].success and context["result"].model is not None:
                        context["run_id"] = _retain_model(
                            context["result"].model, simulation
                        )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                context["error"] = str(exc)
        return render_template("index.html", **context, **_model_defaults())

    @app.get("/run-progress/<progress_id>")
    def run_progress(progress_id: str):
        """Return the latest processing phase for an in-flight run."""
        with _RUN_PROGRESS_LOCK:
            message = _RUN_PROGRESS.get(progress_id, "")
        return jsonify({"message": message})

    @app.get("/runs/<run_id>/plot/<kind>.png")
    def plot(run_id: str, kind: str):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import astropy.units as u
        import surf.surf_analysis as sa

        retained = _run_for(run_id)
        model = retained["model"]
        with _PLOT_LOCK:
            plt.close("all")
            if kind == "map":
                plot_time = float(request.args.get("time", 1.5)) * u.day
                simulation = retained.get("simulation")
                solver = (
                    str(simulation.model["solver"]).lower()
                    if isinstance(simulation, SimulationRequest)
                    else str(getattr(model, "solver", "huxt")).lower()
                )
                options = {
                    "minimalplot": request.args.get("minimal") == "1",
                    "plotHCS": request.args.get("plot_hcs", "1") == "1",
                    "annotateplot": request.args.get("annotate", "1") == "1",
                    "plot_rmax": (
                        float(request.args["plot_rmax"])
                        if request.args.get("plot_rmax")
                        else None
                    ),
                }
                if "huxt" in solver:
                    options["trace_earth_connection"] = (
                        request.args.get("trace_earth") == "1"
                    )
                    sa.plot(model, plot_time, **options)
                else:
                    sa.plot_compressible(model, plot_time, **options)
            elif kind == "radial":
                plot_radial_profile(
                    model,
                    float(request.args.get("radial_time", 1.5)) * u.day,
                    lon=float(request.args.get("radial_lon", 0)) * u.deg,
                )
            elif kind == "timeseries":
                observer = request.args.get("observer", "custom")
                if observer == "custom":
                    plot_custom_timeseries(
                        model,
                        float(request.args.get("radius", 1)) * u.AU,
                        lon=float(request.args.get("timeseries_lon", 0)) * u.deg,
                    )
                else:
                    import numpy as np

                    series = sa.get_observer_timeseries(model, observer=observer)
                    fields = [
                        (key, label)
                        for key, label in (
                            ("vsw", "V [km/s]"),
                            ("bpol", "Bpol"),
                            ("n", "n [cm-3]"),
                            ("T", "T [K]"),
                        )
                        if key in series
                        and np.isfinite(np.asarray(series[key], dtype=float)).any()
                    ]
                    figure, axes = plt.subplots(
                        len(fields), 1, figsize=timeseries_figsize(), sharex=True
                    )
                    for axis, (key, label) in zip(np.atleast_1d(axes), fields):
                        axis.plot(series["time"], series[key], "r-")
                        axis.set_ylabel(label)
                        if key == "vsw":
                            axis.set_ylim(300, 900)
                        axis.grid(True, alpha=0.3)
                    format_datetime_axis_like_surf(
                        figure,
                        np.atleast_1d(axes),
                        series["time"],
                    )
                    figure.subplots_adjust(
                        left=0.10,
                        bottom=0.14,
                        right=0.98,
                        top=0.90,
                        hspace=0.05,
                    )
                    figure.suptitle(f"SURF time series at {observer}")
            else:
                abort(404)
            output = io.BytesIO()
            plt.gcf().savefig(output, format="png", dpi=140, bbox_inches="tight")
            plt.close("all")
        output.seek(0)
        return send_file(output, mimetype="image/png")

    @app.post("/ambient-plot.png")
    def ambient_plot():
        import matplotlib.pyplot as plt

        with _PLOT_LOCK:
            figure = _ambient_preview_figure()
            output = io.BytesIO()
            figure.savefig(output, format="png", dpi=140, bbox_inches="tight")
            plt.close("all")
        output.seek(0)
        return send_file(output, mimetype="image/png")

    @app.get("/donki-cmes")
    def donki_cmes():
        start = datetime.datetime.fromisoformat(request.args["start"].replace("T", " "))
        duration = float(request.args.get("duration", 10))
        return jsonify(_fetch_donki_cmes(start, duration))

    @app.get("/runs/<run_id>/timeseries.csv")
    def timeseries_csv(run_id: str):
        import astropy.units as u
        import surf.surf_analysis as sa

        model = _model_for(run_id)
        observer = request.args.get("observer", "Earth")
        if observer == "custom":
            radius = float(request.args.get("radius", 1)) * u.AU
            longitude = float(request.args.get("timeseries_lon", 0)) * u.deg
            series = sample_custom_timeseries(model, radius, longitude)
        else:
            series = sa.get_observer_timeseries(model, observer=observer)
        if hasattr(series, "to_csv"):
            csv_text = series.to_csv(index=False)
        else:
            columns = list(series)
            rows = zip(*(series[column] for column in columns))
            text = io.StringIO()
            writer = csv.writer(text)
            writer.writerow(columns)
            writer.writerows(rows)
            csv_text = text.getvalue()
        payload = io.BytesIO(csv_text.encode("utf-8"))
        return send_file(
            payload,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"SURF_{observer}_timeseries.csv",
        )

    @app.get("/runs/<run_id>/movie/<kind>.gif")
    def movie(run_id: str, kind: str):
        import surf.surf_analysis as sa

        model = _model_for(run_id)
        duration = float(request.args.get("duration", 10))
        fps = int(request.args.get("fps", 5))
        with tempfile.TemporaryDirectory() as directory, _PLOT_LOCK:
            path = Path(directory) / "surf_movie.gif"
            options = {
                "tag": request.args.get("tag", "gui"),
                "duration": duration,
                "fps": fps,
                "plotHCS": request.args.get("plot_hcs", "1") == "1",
                "plot_rmax": (
                    float(request.args["plot_rmax"])
                    if request.args.get("plot_rmax")
                    else None
                ),
                "outputfilepath": str(path),
            }
            if kind == "map":
                options["trace_earth_connection"] = request.args.get("trace_earth") == "1"
                animation = sa.animate
            elif kind == "timeseries":
                options["polar_var"] = request.args.get("field", "V")
                animation = sa.animate_with_ts
            else:
                abort(404)
            signature = inspect.signature(animation)
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            supported_options = (
                options
                if accepts_kwargs
                else {
                    key: value
                    for key, value in options.items()
                    if key in signature.parameters
                }
            )
            saved = animation(model, **supported_options)
            movie_path = Path(saved) if saved else path
            payload = io.BytesIO(movie_path.read_bytes())
        payload.seek(0)
        return send_file(
            payload,
            mimetype="image/gif",
            as_attachment=request.args.get("inline") != "1",
            download_name=f"SURF_{run_id[:8]}.gif",
        )

    @app.get("/runs/<run_id>/movie.gif")
    def legacy_movie(run_id: str):
        """Keep old bookmarked movie URLs working."""
        return movie(run_id, "map")

    return app


def main() -> None:
    """Run the development server."""
    create_app().run(debug=True)


if __name__ == "__main__":
    main()
