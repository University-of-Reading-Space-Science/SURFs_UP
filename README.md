# SURFs_UP

SURFs_UP provides desktop and web interfaces for
[SURF](https://github.com/University-of-Reading-Space-Science/SURF). It is
packaged separately so SURF can remain a GUI-independent modelling library.

Both interfaces use `surfs_up.core` for request validation and execution.
Interface code should only translate widgets or form fields into a
`SimulationRequest`; reusable modelling behaviour belongs in the core package.

## Development installation

Install SURF and SURFs_UP into the same environment. Choose the interface extras
you need:

```powershell
pip install -e ../SURF
pip install -e ".[desktop]"
```

Run the application with either command:

```powershell
surfs-up
surf-gui
```

The `surf-gui` alias is retained for users of the original bundled GUI.

## Web interface

The Flask workflow supports user-specified, MAS, WSA, CorTom, OMNI-backmapped,
and OMNI-outwards ambient boundaries, plus magnetic boundaries, streak lines,
and JSON-defined Cone CMEs. It uses the same `SimulationRequest`, general code
generator, and execution service as the desktop application:

```powershell
pip install -e ".[web]"
surfs-up-web
```

Open `http://127.0.0.1:5000`. The web form can preview generated code or run it
synchronously. Completed models can produce 2D maps, radial profiles, time
series, and downloadable GIF movies. Up to eight models are retained in the
current web process; restarting or reloading the process clears them.

Before exposing a production site to multiple users, move long model and movie
runs to a background job system and persistent result store so they do not
occupy a web worker. PythonAnywhere may run more than one worker, so the current
in-memory result store is intended for development and single-worker trials.

## PythonAnywhere

Create a virtual environment, install SURF and then install this project with
the web extra. Configure the PythonAnywhere web app's WSGI file to import the
provided application:

```python
import sys

project = "/home/YOUR_USERNAME/SURFs_UP"
if project not in sys.path:
    sys.path.insert(0, project)

from wsgi import application
```

Set the web app source directory to the repository and reload it. The repository
root [wsgi.py](wsgi.py) is intentionally small so deployment-specific settings
can later be supplied without coupling them to Flask routes.
