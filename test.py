import datetime
import numpy as np
import astropy.units as u
import astropy.constants as const
import surf.surf as s
import surf.surf_inputs as sin
import surf.surf_insitu as sinsit

# Define settings shared by the boundary preparation and model setup.
solver = 'hydro'
acc_profile = 'huxt' if solver == 'huxt' else 'parker'
rmin = 21.5 * u.solRad
rmax = 430.0 * u.solRad
latitude = -7.25 * u.deg
simtime = 27.0 * u.day
start_time = datetime.datetime.fromisoformat('2024-03-05 08:48')
gamma = 1.5

# Prepare the selected ambient solar-wind boundary.
omni_input = sinsit.get_omni(start_time - datetime.timedelta(days=28), start_time + datetime.timedelta(days=simtime.to_value(u.day) + 28))
omni_numeric = omni_input.select_dtypes(include='number').columns
omni_input[omni_numeric] = omni_input[omni_numeric].interpolate(
    method='linear', limit_direction='both'
)
omni_input = sinsit.removeICMEs(omni_input, icme_list='CaneRichardson', pre_icme_buffer=1.0, post_icme_buffer=1.0)
model = sinsit.omniSURF_reconstruction(start_time, start_time + datetime.timedelta(days=simtime.to_value(u.day)), rmin=rmin, rmax=rmax, dr=1.5*u.solRad, nlon=128, v_max=3000.0*(u.km/u.s), dt_scale=4, solver=solver, gamma=gamma, run_2d=False, track_cmes=True, include_b_boundary=False, icme_list='CaneRichardson', omni_input=omni_input)

model.latitude = latitude.to(u.rad)

# Build the list of cone CMEs injected into the simulation.
cme_list = []
donki_end_time = start_time + datetime.timedelta(days=simtime.to_value(u.day))
try:
    donki_cmes = sin.get_DONKI_cme_list(model, start_time, donki_end_time)
except Exception as exc:
    raise RuntimeError('DONKI CME data could not be accessed') from exc
print(f'Loaded {len(donki_cmes)} DONKI cone CMEs for this run')
if solver == 'hydro':
    for donki_cme in donki_cmes:
        donki_cme.profile_type = 'sinusoidal'
cme_list.extend(donki_cmes)

# Evolve the model with the configured CMEs and optional streak lines.
model.solve(cme_list, streak_carr=np.arange(0, 360, 10.0)*u.deg)


# Plotting code run from the web interface
import matplotlib.pyplot as plt
import surf.surf_analysis as sa

# Plot the SURF Earth time series, optionally overlaid with OMNI observations.
try:
    sa.plot_earth_timeseries(model, plot_omni=True)
except Exception:
    # OMNI may be unavailable for the requested dates.
    plt.close('all')
    sa.plot_earth_timeseries(model, plot_omni=False)
plt.show()
