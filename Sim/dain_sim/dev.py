"""dain_sim.dev — single source of truth for the hermetic dev-cluster secrets.

Every simulator entrypoint (cluster, chaos) and every Sim test must use these
values, never re-declare them. Keeping one definition prevents a silent secret
drift (a node joining with a stale token would fail registration only at odd
moments).
"""

from __future__ import annotations

JOIN_TOKEN = "dain-dev-join-token"
API_KEY = "dain-dev-key"
ADMIN_KEY = "dain-dev-admin-key"
ADMIN_HEADERS = {"X-Admin-Key": ADMIN_KEY}