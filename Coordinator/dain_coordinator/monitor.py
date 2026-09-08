"""Heartbeat monitor (S2): background task enforcing the offline timeout.

Ticks every ``monitor_tick_s`` and demotes silent nodes to OFFLINE via the
NodeService state machine, so detection latency is bounded by
``offline_timeout_s + monitor_tick_s`` (spec §6/§13).
"""

from __future__ import annotations

import asyncio
import logging
import time

from dain_coordinator.nodes import NodeService
from dain_coordinator.settings import CoordinatorSettings

log = logging.getLogger("dain.coordinator.monitor")


class HeartbeatMonitor:
    def __init__(self, service: NodeService, settings: CoordinatorSettings) -> None:
        self.service = service
        self.settings = settings

    async def run(self) -> None:
        log.info(
            "monitor_started tick=%.2fs timeout=%.2fs",
            self.settings.monitor_tick_s,
            self.settings.offline_timeout_s,
        )
        while True:
            await asyncio.sleep(self.settings.monitor_tick_s)
            try:
                evicted = self.service.enforce_timeouts(time.time())
                for node_id in evicted:
                    log.warning("node_timed_out node=%s", node_id)
            except Exception:
                # The monitor must survive any transient store/service error.
                log.exception("monitor_tick_failed")
