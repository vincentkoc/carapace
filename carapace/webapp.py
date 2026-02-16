"""Compatibility facade for web app factories."""

from carapace.webapp_routes import create_app, create_app_from_env

__all__ = ["create_app", "create_app_from_env"]
