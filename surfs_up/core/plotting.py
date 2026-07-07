"""Plotting helpers shared by the desktop and web interfaces."""

from __future__ import annotations

import astropy.units as u
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np


CME_COLORS = ["r", "c", "m", "y", "deeppink", "darkorange"]
SPEED_LIMITS_KMS = (300, 900)
TIMESERIES_FIGSIZE = (10, 6.25)


def _cme_coords_at_time(cme, time_index: int):
    """Return CME coordinates for a model time index, if available."""
    coords = getattr(cme, "coords", None)
    if not isinstance(coords, dict):
        return None
    return coords.get(int(time_index))


def format_datetime_axis_like_surf(fig, axes, times):
    """Format a shared datetime x-axis using SURF's compact date style."""
    axes = np.atleast_1d(axes)
    starttime = times.iloc[0] if hasattr(times, "iloc") else times[0]
    endtime = times.iloc[-1] if hasattr(times, "iloc") else times[-1]

    for axis in axes:
        axis.set_xlim(starttime, endtime)

    duration_days = (endtime - starttime).total_seconds() / 86400
    if duration_days <= 7:
        axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    else:
        axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    fig.autofmt_xdate(rotation=0, ha="center")
    axes[-1].set_xlabel(f"DD-MM of {starttime.year}", fontsize=12, fontweight="bold")

    return starttime, endtime


def _values(data):
    """Return plain numeric values from arrays that may carry units."""
    return np.asarray(data.value if hasattr(data, "value") else data, dtype=float)


def _solver_label(model):
    solver = str(getattr(model, "solver", "huxt"))
    return f"SURF-{solver}" if getattr(model, "compressible", False) else "SURF-HUXt"


def timeseries_figsize():
    """Return the standard figure size for all time-series plots."""
    return TIMESERIES_FIGSIZE


def sample_custom_timeseries(model, radius, lon):
    """Sample model grids at the nearest custom radius and longitude."""
    radial_index = int(np.argmin(np.abs(model.r - radius)))
    if model.lon.size == 1:
        longitude_index = 0
        lon_out = float(np.asarray(model.lon.value).reshape(-1)[0])
    else:
        longitude_index = int(np.argmin(np.abs(model.lon - lon)))
        lon_out = float(model.lon[longitude_index].value)

    series = {
        "time_days": np.asarray(model.time_out.to_value(u.day), dtype=float),
        "vsw": _values(model.v_grid[:, radial_index, longitude_index]),
        "r_out": float(model.r[radial_index].value),
        "lon_out": lon_out,
    }
    if hasattr(model, "time_init"):
        series["time"] = [(model.time_init + time).datetime for time in model.time_out]
    if hasattr(model, "b_grid"):
        series["bpol"] = _values(model.b_grid[:, radial_index, longitude_index])
    if getattr(model, "compressible", False):
        m_p = 1.6726e-27
        series["n"] = _values(model.rho_grid[:, radial_index, longitude_index]) / m_p / 1e6
        series["T"] = _values(model.temp_grid[:, radial_index, longitude_index])
    return series


def plot_custom_timeseries(model, radius, lon):
    """Plot a custom-coordinate time series using the observer plot style."""
    series = sample_custom_timeseries(model, radius, lon)
    times = series.get("time", series["time_days"])
    speed = series["vsw"]
    bpol = np.asarray(series.get("bpol", np.nan), dtype=float)
    has_bpol = np.isfinite(bpol).any()
    is_compressible = getattr(model, "compressible", False)

    n_panels = 1 + int(has_bpol) + (2 if is_compressible else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=timeseries_figsize(), sharex=True)
    axes = np.atleast_1d(axes)

    panel_index = 0
    axes[panel_index].plot(times, speed, "r", label=_solver_label(model))
    axes[panel_index].set_ylim(*SPEED_LIMITS_KMS)
    axes[panel_index].set_ylabel("V [km/s]")

    if has_bpol:
        panel_index += 1
        axes[panel_index].plot(times, np.sign(bpol), "r.", label=_solver_label(model))
        axes[panel_index].set_ylabel(r"B$_{\text{POL}}$")
        axes[panel_index].set_ylim(-1.1, 1.1)

    if is_compressible:
        panel_index += 1
        axes[panel_index].semilogy(times, series["n"], "r-", label=_solver_label(model))
        axes[panel_index].set_ylabel(r"n$_\text{P}$ [cm$^{-3}$]")
        axes[panel_index].set_ylim(0.101, 999)

        panel_index += 1
        axes[panel_index].semilogy(times, series["T"], "r-", label=_solver_label(model))
        axes[panel_index].set_ylabel(r"T [K]")
        axes[panel_index].set_ylim(1e4, 9.9e6)

    for axis in axes:
        axis.grid(True, alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
    for axis in axes[:-1]:
        axis.set_xticklabels([])

    if "time" in series:
        format_datetime_axis_like_surf(fig, axes, times)
    else:
        axes[-1].set_xlim(np.nanmin(times), np.nanmax(times))
        axes[-1].set_xlabel("Time (days)")
        for axis in axes[:-1]:
            axis.set_xlim(np.nanmin(times), np.nanmax(times))
    fig.subplots_adjust(left=0.10, bottom=0.14, right=0.98, top=0.95, hspace=0.05)
    frame = str(getattr(model, "frame", "model"))
    fig.suptitle(
        f"{_solver_label(model)} | r={series['r_out']:.1f} Rs | "
        f"fixed {frame} model lon={series['lon_out']:.1f} deg",
        fontsize=16,
    )
    return fig, axes if n_panels > 1 else axes[0]


def plot_radial(model, time, lon, save: bool = False, tag: str = ""):
    """Plot a radial profile, tolerating CMEs missing the selected time index."""
    if (time < model.time_out.min()) | (time > model.time_out.max()):
        print("Error, input time outside span of model times. Defaulting to closest time")
        time_index = int(np.argmin(np.abs(model.time_out - time)))
        time = model.time_out[time_index]

    if model.lon.size != 1:
        if (lon < model.lon.min()) | (lon > model.lon.max()):
            print(
                "Error, input lon outside range of model longitudes."
                " Defaulting to closest longitude"
            )
            lon_index = int(np.argmin(np.abs(model.lon - lon)))
            lon = model.lon[lon_index]

    is_compressible = hasattr(model, "compressible") and model.compressible
    if is_compressible:
        fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(14, 7))
        axes = [ax]

    time_index = int(np.argmin(np.abs(model.time_out - time)))
    time_out = model.time_out[time_index].to(u.day).value

    if model.lon.size == 1:
        lon_index = 0
        lon_out = float(np.asarray(model.lon.value).reshape(-1)[0])
    else:
        lon_index = int(np.argmin(np.abs(model.lon - lon)))
        lon_out = model.lon[lon_index].to(u.deg).value

    ax.plot(model.r, model.v_grid[time_index, :, lon_index], "k-")
    ax.set_ylim(*SPEED_LIMITS_KMS)

    for cme_index, cme in enumerate(getattr(model, "cmes", [])):
        coords = _cme_coords_at_time(cme, time_index)
        if coords is None:
            continue

        lon_cme = coords["lon"]
        r_cme = coords["r"].to(u.solRad)
        front_id = coords["front_id"]

        id_front = front_id == 1.0
        id_back = front_id == 0.0
        if not np.any(id_front) or not np.any(id_back):
            continue

        lon_front = lon_cme[id_front]
        lon_back = lon_cme[id_back]
        r_front = r_cme[id_front]
        r_back = r_cme[id_back]

        r_front = r_front[int(np.argmin(np.abs(lon_front - lon)))]
        r_back = r_back[int(np.argmin(np.abs(lon_back - lon)))]

        id_cme = (model.r >= r_back) & (model.r <= r_front)
        if not np.any(id_cme):
            continue

        ax.plot(
            model.r[id_cme],
            model.v_grid[time_index, id_cme, lon_index],
            ".",
            color=CME_COLORS[np.mod(cme_index, len(CME_COLORS))],
            label=f"CME {cme_index:02d}",
        )

    ax.set_ylabel("V (km/s)" if is_compressible else "Solar Wind Speed (km/s)")
    ax.set_xlim(model.r.value.min(), model.r.value.max())
    if not is_compressible:
        ax.set_xlabel("Radial distance ($R_{sun}$)")

    if is_compressible:
        m_p = 1.6726e-27
        n_profile = model.rho_grid[time_index, :, lon_index].value / m_p / 1e6
        axes[1].semilogy(model.r, n_profile, "b-")
        axes[1].set_ylabel("n (protons/cm^3)", color="b")
        axes[1].tick_params(axis="y", labelcolor="b")
        axes[1].set_xlim(model.r.value.min(), model.r.value.max())
        axes[1].grid(True, alpha=0.3)

        t_profile = model.temp_grid[time_index, :, lon_index].value
        axes[2].semilogy(model.r, t_profile, "r-")
        axes[2].set_ylabel("T (K)", color="r")
        axes[2].tick_params(axis="y", labelcolor="r")
        axes[2].set_xlabel("Radial distance ($R_{sun}$)")
        axes[2].set_xlim(model.r.value.min(), model.r.value.max())
        axes[2].grid(True, alpha=0.3)

    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
    label = f"SURF Time: {time_out:3.2f} days Lon: {lon_out:3.2f}$^\\circ$"
    if is_compressible:
        axes[0].set_title(label, fontsize=20)
    else:
        ax.text(0.05, 1.02, label, color="k", fontsize=20, transform=ax.transAxes)

    for axis in axes:
        axis.grid(True, alpha=0.3)

    if ax.get_legend_handles_labels()[0]:
        ax.legend()

    if save:
        filepath = f"SURF_Radial{tag}.png"
        fig.savefig(filepath)
        return fig, ax

    return fig, ax
