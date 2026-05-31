"""Live monitoring dashboard: incremental detection loop, HTML report, and server.

This package is detached from the batch :class:`~spotanomaly2.application.pipeline.Pipeline`.
``LiveMonitor`` owns the live-prediction flow (incremental fetch → process →
detect with an existing model), the HTML report generation, and the
Server-Sent-Events report server used by the ``live`` CLI command.
"""

from spotanomaly2.dashboard.live_monitor import LiveMonitor

__all__ = ["LiveMonitor"]
