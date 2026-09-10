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
