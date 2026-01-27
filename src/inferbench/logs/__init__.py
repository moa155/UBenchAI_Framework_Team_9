"""
Logs Module for InferBench Framework.

Provides log collection, aggregation, filtering, and export.
"""

from inferbench.logs.manager import (
    LogManager,
    LogEntry,
    LogCollection,
    get_log_manager,
)

__all__ = [
    "LogManager",
    "LogEntry",
    "LogCollection",
    "get_log_manager",
]
