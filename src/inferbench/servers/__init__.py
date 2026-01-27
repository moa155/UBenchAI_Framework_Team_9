"""
Server Module for InferBench Framework.

Provides management of containerized AI services on HPC clusters.
"""

from inferbench.servers.manager import ServerManager, get_server_manager

__all__ = [
    "ServerManager",
    "get_server_manager",
]
