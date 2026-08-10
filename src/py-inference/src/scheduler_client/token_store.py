"""Rotating gateway-token store.

SP self-registration issues this worker a per-worker Bearer token, and every
re-registration ROTATES it. The scheduler advertises the current token on each
GetGpuSchedule response; the listener feeds it here and the API server accepts
any held token. The last two tokens are kept so requests forwarded with a
freshly-rotated token — or ones still in flight with the previous token — both
authenticate during the ≤1 poll-interval handover window.
"""

import logging
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


class GatewayTokenStore:
    """Thread-safe holder of the accepted dynamic gateway tokens."""

    def __init__(self, keep: int = 2):
        self._keep = max(1, keep)
        self._tokens: "OrderedDict[str, None]" = OrderedDict()
        self._lock = threading.Lock()

    def update(self, token: str) -> None:
        """Record the token currently advertised by the scheduler (no-op if empty
        or already newest)."""
        if not token:
            return
        with self._lock:
            if token in self._tokens:
                self._tokens.move_to_end(token)
                return
            self._tokens[token] = None
            self._tokens.move_to_end(token)
            rotated = len(self._tokens) > 1
            while len(self._tokens) > self._keep:
                self._tokens.popitem(last=False)
        # Log OUTSIDE the lock; never log the token value itself.
        if rotated:
            logger.info("gateway auth token rotated (previous token remains valid briefly)")
        else:
            logger.info("gateway auth token received from scheduler")

    def accepted(self) -> list[str]:
        """Currently-accepted dynamic tokens, newest last."""
        with self._lock:
            return list(self._tokens.keys())
