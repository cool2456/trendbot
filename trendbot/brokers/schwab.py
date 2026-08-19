"""Schwab adapter - deliberately not implemented.

Two independent reasons, both of which make Schwab unusable for the protocol in
PREREGISTRATION.md section 7 rather than merely inconvenient:

1. **There is no paper environment.** Schwab's Trader API offers no sandbox or
   paper-trading endpoint; orders placed through it are real orders against a real
   brokerage account. BUILD_PROMPT.md's hard invariant 4 is "paper only", and
   section 7 step 6 requires a minimum of six months of paper trading before any
   decision. An adapter that could only trade live would make it possible to
   satisfy neither.

2. **The refresh token expires after 7 days.** Schwab issues a refresh token with a
   fixed seven-day lifetime and no programmatic renewal path: restoring access
   requires a human to complete an interactive browser login. A strategy that
   rebalances on the first trading day of each month would therefore be
   unauthenticated on roughly three quarters of the days it is supposed to be
   running, and would need a manual login before every single rebalance. That is
   not unattended operation, and a scheduled job that silently fails to
   authenticate three weeks out of four is worse than no job at all.

If Schwab ever ships a sandbox and a longer-lived credential, implementing
:class:`trendbot.brokers.base.Broker` here is the only change required; nothing else
in the package knows which broker it is talking to.
"""

from __future__ import annotations

from .base import Broker

__all__ = ["SchwabBroker"]

_REASON = (
    "Schwab is not supported. Its API has no paper-trading environment, so it cannot "
    "satisfy the paper-only invariant, and its refresh token expires after 7 days with "
    "no unattended renewal, so a monthly rebalance would require an interactive browser "
    "login before nearly every run. Use the Alpaca paper adapter."
)


class SchwabBroker(Broker):
    """Placeholder. Every method raises."""

    is_paper = False

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(_REASON)

    @property
    def name(self) -> str:
        raise NotImplementedError(_REASON)

    def get_positions(self):
        raise NotImplementedError(_REASON)

    def get_account(self):
        raise NotImplementedError(_REASON)

    def get_open_orders(self):
        raise NotImplementedError(_REASON)

    def submit(self, symbol, qty, side, client_order_id):
        raise NotImplementedError(_REASON)

    def get_clock(self):
        raise NotImplementedError(_REASON)

    def get_last_close(self, symbols):
        raise NotImplementedError(_REASON)
