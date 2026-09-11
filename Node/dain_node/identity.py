"""Persistent node identity (spec §11): stable node_id + issued node_token.

The token issued at first registration is stored locally so the agent can
re-register after restarts/OFFLINE without re-joining; history stays keyed to
the stable node_id.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
from dataclasses import dataclass

log = logging.getLogger("dain.node.identity")


def generate_node_id() -> str:
    host = socket.gethostname().split(".")[0] or "host"
    return f"node-{host}-{secrets.token_hex(2)}"


@dataclass
class IdentityState:
    node_id: str
    node_token: str | None = None

    @classmethod
    def load(cls, path: str) -> IdentityState | None:
        """Load persisted identity; missing or corrupt files yield None (fresh join)."""
        try:
            with open(path, encoding="utf-8-sig") as fh:
                data = json.load(fh)
            node_id, token = data["node_id"], data.get("node_token")
            if not isinstance(node_id, str) or not node_id:
                return None
            return cls(node_id=node_id, node_token=token)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("identity_state_unreadable path=%s (starting fresh)", path)
            return None

    @classmethod
    def create(cls, path: str, node_id: str | None = None) -> IdentityState:
        state = cls(node_id=node_id or generate_node_id())
        state.save(path)
        return state

    def save(self, path: str) -> None:
        """Atomic write (tmp + replace) so a crash never truncates the identity."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp = f"{path}.tmp-{secrets.token_hex(4)}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"node_id": self.node_id, "node_token": self.node_token}, fh)
        os.replace(tmp, path)
