"""
Monitor Module for InferBench Framework.

Provides Prometheus and Grafana deployment for metrics collection.
"""

from inferbench.monitors.manager import MonitorManager, get_monitor_manager

__all__ = [
    "MonitorManager",
    "get_monitor_manager",
]
