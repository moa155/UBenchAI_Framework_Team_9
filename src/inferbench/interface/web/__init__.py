"""
Web Interface Module for InferBench Framework.

Provides a Flask-based dashboard for monitoring and managing benchmarks.
"""

from inferbench.interface.web.app import create_app, run_server

__all__ = [
    "create_app",
    "run_server",
]
