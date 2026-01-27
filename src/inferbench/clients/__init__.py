"""
Client Module for InferBench Framework.

Provides benchmark client execution and results collection.
"""

from inferbench.clients.manager import ClientManager, get_client_manager

__all__ = [
    "ClientManager",
    "get_client_manager",
]
