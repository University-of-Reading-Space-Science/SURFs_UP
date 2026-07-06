"""Fast tests for framework-neutral SURFs_UP behaviour."""

from surfs_up.core import (
    SimulationRequest,
    build_generated_code,
    build_uniform_boundary_code,
    run_generated_code,
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
    assert code.index("map_v_inwards") < code.index("map_v_boundary_inwards")


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
