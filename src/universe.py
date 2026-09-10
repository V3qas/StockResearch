from __future__ import annotations

import json
from pathlib import Path


def load_universe(path: str | Path) -> dict:
    """Read an explicit frozen selection; never refresh membership during a run."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Universe must be a JSON object with a tickers list.")
    tickers = payload.get("tickers")
    if not isinstance(tickers, list) or not tickers or any(
        not isinstance(ticker, str) or not ticker.strip() for ticker in tickers
    ):
        raise ValueError("Universe tickers must be a nonempty list of symbols.")
    return payload
