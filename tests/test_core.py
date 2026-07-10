"""Fast tests for framework-neutral SURFs_UP behaviour."""

from types import SimpleNamespace

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from surfs_up.core import (
    SimulationRequest,
    build_generated_code,
    build_uniform_boundary_code,
    plot_custom_timeseries,
    plot_radial,
    run_generated_code,
    sample_custom_timeseries,
    timeseries_figsize,
)


def request() -> SimulationRequest:
    return SimulationRequest.from_mappings(
        {
            "solver": "huxt",
            "rmin": 21.5,
            "rmax": 240.0,
            "latitude": 0.0,
            "simtime_days": 5.0,
            "dr_rs": 1.5,
            "nlon": 128,
            "vmax_kms": 3000.0,
            "start_datetime": "2026-07-03 12:00:00",
            "cr_num": 2300,
            "cr_lon_init_deg": 0.0,
        },
        {"source": "user_specified", "speed_profile_kms": [400.0] * 128},
    )


def test_generated_code_is_valid_python():
    code = build_uniform_boundary_code(request())
    compile(code, "<generated>", "exec")
    assert "dr=1.5 * u.solRad" in code
    assert "nlon=128" in code
    assert "v_max=3000.0 * (u.km/u.s)" in code
    assert "track_cmes=False" in code


def test_generated_code_can_enable_cme_front_tracking():
    simulation = request()
    simulation.model["track_cmes"] = True

    code = build_uniform_boundary_code(simulation)

    assert "track_cmes=True" in code


def test_runner_reports_immediately_before_model_solve():
    code = """
print("prepared")
class Model:
    def solve(self):
        print("solved")
model = Model()
model.solve()
"""

    result = run_generated_code(code, before_solve=lambda: print("running"))

    assert result.success
    assert result.output.splitlines() == ["prepared", "running", "solved"]


def test_invalid_radial_bounds_are_rejected():
    simulation = request()
    simulation.model["rmin"] = simulation.model["rmax"]

    try:
        simulation.validate()
    except ValueError as exc:
        assert "inner radial boundary" in str(exc)
    else:
        raise AssertionError("Expected invalid radial bounds to be rejected")


def test_radial_plot_skips_cme_without_selected_time_coordinates():
    model = SimpleNamespace(
        time_out=np.array([0.0, 1.0]) * u.day,
        lon=np.array([0.0]) * u.deg,
        r=np.array([20.0, 30.0, 40.0]) * u.solRad,
        v_grid=np.array([[[400.0], [420.0], [440.0]], [[450.0], [470.0], [490.0]]]),
        cmes=[
            SimpleNamespace(
                coords={
                    0: {
                        "lon": np.array([0.0, 0.0]) * u.deg,
                        "r": np.array([25.0, 35.0]) * u.solRad,
                        "front_id": np.array([0.0, 1.0]),
                    }
                }
            )
        ],
    )

    fig, ax = plot_radial(model, 1.0 * u.day, lon=0.0 * u.deg)

    assert fig is not None
    assert ax.lines
    assert ax.get_ylim() == (300.0, 900.0)
    plt.close(fig)


def test_custom_timeseries_plots_bpol_when_grid_is_available():
    model = SimpleNamespace(
        time_out=np.array([0.0, 1.0, 2.0]) * u.day,
        lon=np.array([0.0]) * u.deg,
        r=np.array([20.0, 215.0]) * u.solRad,
        v_grid=np.array(
            [[[390.0], [400.0]], [[410.0], [420.0]], [[430.0], [440.0]]]
        )
        * (u.km / u.s),
        b_grid=np.array([[[1.0], [-1.0]], [[-1.0], [1.0]], [[1.0], [-1.0]]]),
        cmes=[],
        compressible=False,
        time_init=Time("2026-07-03T12:00:00"),
    )

    series = sample_custom_timeseries(model, 1.0 * u.AU, 0.0 * u.deg)
    fig, axes = plot_custom_timeseries(model, 1.0 * u.AU, 0.0 * u.deg)
    axes = np.atleast_1d(axes)

    assert "bpol" in series
    assert "time" in series
    assert tuple(fig.get_size_inches()) == timeseries_figsize()
    assert len(axes) == 2
    assert axes[0].get_ylim() == (300.0, 900.0)
    assert axes[1].get_ylabel() == r"B$_{\text{POL}}$"
    assert axes[-1].get_xlabel() == "DD-MM of 2026"
    plt.close(fig)


def test_general_generator_supports_cmes():
    simulation = request()
    simulation.cmes.append(
        {
            "t_launch_day": 0.5,
            "longitude": 0,
            "latitude": 0,
            "speed": 800,
            "width": 60,
        }
    )
    code = build_generated_code(simulation)

    compile(code, "<generated>", "exec")
    assert "s.ConeCME" in code
    assert "model.solve(cme_list" in code


def test_generated_code_fetches_donki_at_runtime_when_requested():
    simulation = request()
    simulation.model["grab_donki_at_run_start"] = True
    simulation.cmes.extend(
        [
            {
                "t_launch_day": 0.25,
                "longitude": 10,
                "latitude": 2,
                "speed": 900,
                "width": 45,
                "source": "donki",
                "donki_id": "downloaded-at-run",
            },
            {
                "t_launch_day": 0.5,
                "longitude": 0,
                "latitude": 0,
                "speed": 800,
                "width": 60,
                "source": "manual",
            },
        ]
    )

    code = build_generated_code(simulation)

    compile(code, "<generated>", "exec")
    assert "sin.get_DONKI_cme_list(model, start_time, donki_end_time)" in code
    assert "downloaded-at-run" not in code
    assert code.count("s.ConeCME") == 1


def test_generated_code_includes_preloaded_donki_list_when_runtime_fetch_is_off():
    simulation = request()
    simulation.cmes.append(
        {
            "t_launch_day": 0.25,
            "longitude": 10,
            "latitude": 2,
            "speed": 900,
            "width": 45,
            "source": "donki",
            "donki_id": "preloaded-donki",
        }
    )

    code = build_generated_code(simulation)

    compile(code, "<generated>", "exec")
    assert "sin.get_DONKI_cme_list" not in code
    assert "s.ConeCME" in code


def test_generated_code_passes_absolute_cme_density_as_value():
    simulation = request()
    simulation.model["solver"] = "hydro"
    simulation.cmes.append(
        {
            "t_launch_day": 0.5,
            "longitude": 0,
            "latitude": 0,
            "speed": 800,
            "width": 60,
            "plasma_mode": "Absolute values",
            "cme_density_pcc": 100,
            "cme_temperature_k": 100000,
        }
    )

    code = build_generated_code(simulation)

    compile(code, "<generated>", "exec")
    assert "cme_density=(100.0/u.cm**3*const.m_p).to_value(u.kg/u.m**3)" in code
    assert "cme_density=(100.0/u.cm**3*const.m_p).to(u.kg/u.m**3)" not in code


def test_generated_code_supports_wsa_iswa():
    simulation = request()
    simulation.ambient = {
        "source": "wsa_iswa",
        "decelerate_to_inner_boundary": True,
        "apply_wsa_speed_reduction": True,
        "iswa_map_datetime": "2024-05-06T00:00:00",
    }

    code = build_generated_code(simulation)

    compile(code, "<generated>", "exec")
    assert "datetime.datetime.fromisoformat('2024-05-06T00:00:00')" in code
    assert "get_WSA_from_ISWA(iswa_map_time)" in code
    assert "get_WSA_long_profile(wsa_path" in code
    assert "map_v_inwards" in code
    assert "acc_profile = 'huxt' if solver == 'huxt' else 'parker'" in code
    assert "acc_profile=acc_profile" in code
    assert code.index("map_v_inwards") < code.index("map_v_boundary_inwards")


def test_generated_code_uses_parker_wsa_reduction_for_non_huxt():
    simulation = request()
    simulation.model["solver"] = "hydro"
    simulation.ambient = {
        "source": "wsa_iswa",
        "decelerate_to_inner_boundary": True,
        "apply_wsa_speed_reduction": True,
        "iswa_map_datetime": "2024-05-06T00:00:00",
    }

    code = build_generated_code(simulation)

    compile(code, "<generated>", "exec")
    assert "if solver == 'huxt':" in code
    assert "sin.map_v_inwards_parker" in code
    assert "sin.map_v_inwards(" in code
    assert "v_boundary = wsa_reduction[0]" in code


def test_generated_code_passes_longitude_range_to_omni_backmapped():
    simulation = request()
    simulation.model["lon_min"] = 120.0
    simulation.model["lon_max"] = 240.0
    simulation.ambient = {
        "source": "insitu_backmapped",
        "mode": "forecast",
    }

    code = build_generated_code(simulation)

    compile(code, "<generated>", "exec")
    assert "omniSURF_forecast" in code
    assert "lon_start=120.0*u.deg" in code
    assert "lon_stop=240.0*u.deg" in code
