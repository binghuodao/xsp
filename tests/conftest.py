"""Shared fixtures for XSP market report tests."""
import os, sys, math, json, datetime
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ── Fake moomoo (private SDK, not on pip) ──
class _FakeMoomoo:
    class OpenQuoteContext:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a, **kw): pass
    class SubType:
        ORDER_BOOK = 1
    RET_OK = 0
sys.modules['moomoo'] = _FakeMoomoo()

import app
from tests.helpers import make_option_chain, std_hs

@pytest.fixture(autouse=True)
def reset_globals():
    """Reset all module-level globals before each test."""
    app._latest_report = {}
    app._morning_report_date = ""
    app._evening_report_date = ""
    app._prev_report_score = 0
    app._prev_report_direction = None
    app._last_watchlist_clean_date = None
    app.user_watchlist = []
    app.latest_data = {
        "index": {"price": 750.0},
        "options": make_option_chain(),
    }
    app.historical_stats = std_hs()
    app.TELEGRAM_TOKEN = ""
    app.TELEGRAM_CHAT_ID = ""


@pytest.fixture
def mock_sio():
    """Mock socketio to capture emitted events."""
    sio = MagicMock()
    with patch.object(app, 'socketio', sio):
        yield sio


@pytest.fixture
def mock_now():
    """Fixture to control datetime.now for testing schedule windows.
    Yields object with .set(syd, et) to change timestamps."""
    class MockNow:
        def __init__(self):
            self.syd_dt = None
            self.et_dt = None
        def set(self, syd=None, et=None):
            self.syd_dt = syd or datetime.datetime(2026, 7, 20, 21, 30, 0)
            self.et_dt = et or datetime.datetime(2026, 7, 20, 7, 30, 0)
    mn = MockNow()
    mn.set()

    real_dt = app.datetime
    class _FakeDatetime:
        def __getattr__(self, name):
            if name == 'now':
                def _now(tz=None):
                    if tz is app.ET_TZ:
                        return mn.et_dt.replace(tzinfo=app.ET_TZ)
                    if tz is app.S_TZ:
                        return mn.syd_dt.replace(tzinfo=app.S_TZ)
                    return datetime.datetime.now(tz)
                return _now
            return getattr(real_dt, name)
    with patch.object(app, 'datetime', _FakeDatetime()):
        yield mn
