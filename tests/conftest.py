"""Shared fixtures for XSP market report tests."""
import os, sys, math, json, datetime, tempfile
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
def reset_globals(monkeypatch):
    """Reset all module-level globals before each test."""
    monkeypatch.setattr(app, '_get_xsp_prev_close', lambda: None)
    monkeypatch.setattr(app, '_get_xsp_closes', lambda: (None, None))
    monkeypatch.setattr(app, '_get_xsp_closes_with_dates', lambda: (None, None, None, None))
    app._latest_report = {}
    app._morning_report_date = ""
    app._evening_report_date = ""
    app._prev_report_score = 0
    app._prev_report_direction = None
    app._last_watchlist_clean_date = None
    app._active_position_date = None
    app._crash_entry_date = None
    app._crash_etf_scaled = False
    app._crash_half_date = None
    app._crash_reentry = False
    app._crash_reentry_date = None
    app._crash_etf_out = False
    app._crash_exit_mode = 'V4'
    app._crash_force_days = 4
    app._mr_entry_date = None
    app._mr_entry_price = None
    app._mr_etf_entry_price = None
    app._trend_opt_expiry = None
    app._trend_opt_strike = None
    app._trend_opt_strike2 = None
    app._trend_opt_entry = None
    app._trend_opt_entry_date = None
    app._trend_opt_sigma = None
    app._trend_opt_pnl = None
    app._entry_price = None
    app._peak_price = None
    app._etf_entry_price = None
    app._etf_peak_price = None
    app.POSITION_FILE = tempfile.mktemp(suffix='.json')
    app.user_watchlist = []
    app.latest_data = {
        "index": {"price": 750.0},
        "options": make_option_chain(),
    }
    app.historical_stats = std_hs()
    app.TELEGRAM_TOKEN = ""
    app.TELEGRAM_CHAT_ID = ""
    # Ensure app.datetime exists for _dte_from_yyyymmdd
    if not hasattr(app, 'datetime') or not hasattr(app.datetime, 'now'):
        app.datetime = datetime


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

    # Use the test override mechanism in app module
    app._test_time_override = lambda tz=None: mn.et_dt.replace(tzinfo=app.ET_TZ) if tz is app.ET_TZ else (mn.syd_dt.replace(tzinfo=app.S_TZ) if tz is app.S_TZ else datetime.datetime.now(tz))
    print(f"FIXTURE: Set _test_time_override = {app._test_time_override}")
    print(f"FIXTURE: _test_time_override in app.__dict__: {app.__dict__.get('_test_time_override')}")

    # Also patch app.datetime for other uses
    class _FakeDatetime:
        def __init__(self, mock_now):
            self._mock_now = mock_now
        
        def now(self, tz=None):
            if tz is app.ET_TZ:
                return self._mock_now.et_dt.replace(tzinfo=app.ET_TZ)
            if tz is app.S_TZ:
                return self._mock_now.syd_dt.replace(tzinfo=app.S_TZ)
            return datetime.datetime.now(tz)
        
        @property
        def datetime(self):
            return datetime.datetime
        
        def strptime(self, *args, **kwargs):
            return datetime.datetime.strptime(*args, **kwargs)
    
    fake_datetime = _FakeDatetime(mn)
    
    with patch.object(app, 'datetime', fake_datetime):
        print("FIXTURE: Patched app.datetime, yielding mn")
        yield mn
        print("FIXTURE: Cleaning up _test_time_override")
        app._test_time_override = None
    
    # Cleanup
    app._test_time_override = None


@pytest.fixture
def mock_sio():
    """Mock socketio to capture emitted events."""
    sio = MagicMock()
    with patch.object(app, 'socketio', sio):
        yield sio