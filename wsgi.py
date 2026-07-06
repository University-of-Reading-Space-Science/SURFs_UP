"""WSGI entry point for PythonAnywhere and other WSGI hosts."""

from surfs_up.web import create_app

application = create_app()
