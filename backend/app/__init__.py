"""FastAPI application package."""

from typing import Any


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Import the HTTP stack lazily for adapter-only consumers."""

    from .main import create_app as factory

    return factory(*args, **kwargs)


__all__ = ["create_app"]
