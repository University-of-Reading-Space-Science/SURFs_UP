# SURFs_UP

SURFs_UP provides a web interface for
[SURF](https://github.com/University-of-Reading-Space-Science/SURF). It can be
run locally or deployed to a WSGI host such as PythonAnywhere.

The Flask application uses `surfs_up.core` for request validation, code
generation, plotting, and execution.

## Development installation

Install SURF and SURFs_UP into the same environment:

```powershell
pip install -e ../SURF
pip install -e .
```

Run the local web application with either command:

```powershell
surfs-up
surfs-up-web
```

The Flask workflow supports user-specified, MAS, WSA, CorTom, OMNI-backmapped,
and OMNI-outwards ambient boundaries, plus magnetic boundaries, streak lines,
and JSON-defined Cone CMEs.

Open `http://127.0.0.1:5000`. The web form can preview generated code or run it
synchronously. Completed models can produce 2D maps, radial profiles, time
series, and downloadable MP4 movies. Up to eight models are retained in the
current web process; restarting or reloading the process clears them.

Before exposing a production site to multiple users, move long model and movie
runs to a background job system and persistent result store so they do not
occupy a web worker. PythonAnywhere may run more than one worker, so the current
in-memory result store is intended for development and single-worker trials.

## PythonAnywhere

Create a virtual environment, install SURF and then install this project.
Configure the PythonAnywhere web app's WSGI file to import the
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
