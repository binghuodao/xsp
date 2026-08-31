"""PositionState: all mutable position state for production and backtest.

This mirrors ALL _crash_*, _mr_*, _trend_*, and other globals from app.py.
Single source of truth for position state across production and backtest.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Optional, Any


@dataclass
class PositionState:
    """All mutable position state — mirrors app.py globals exactly."""
    
    # TREND
    trend_opt_expiry: Optional[str] = None
    trend_opt_strike: Optional[int] = None
    trend_opt_strike2: Optional[int] = None
    trend_opt_entry: Optional[float] = None
    trend_opt_entry_date: Optional[date] = None
    trend_opt_sigma: Optional[float] = None
    trend_opt_pnl: float = 0.0
    active_position_date: Optional[date] = None
    entry_price: Optional[float] = None
    peak_price: Optional[float] = None
    etf_entry_price: Optional[float] = None
    etf_peak_price: Optional[float] = None
    prev_report_direction: Optional[str] = None
    prev_report_score: int = 0
    trend_etf_size: int = 5000
    
    # MR
    mr_entry_date: Optional[date] = None
    mr_entry_price: Optional[float] = None
    mr_etf_entry_price: Optional[float] = None
    mr_k: Optional[int] = None
    mr_sigma: Optional[float] = None
    mr_expiry: Optional[date] = None
    mr_opt_entry: Optional[float] = None
    
    # CRASH (all _crash_* vars from app.py)
    crash_entry_date: Optional[date] = None
    crash_entry_price: Optional[float] = None
    crash_k1: Optional[int] = None
    crash_k2: Optional[int] = None
    crash_debit: Optional[float] = None
    crash_sigma: Optional[float] = None
    crash_etf_entry: Optional[float] = None
    crash_etf_scaled: bool = False
    crash_reentry: bool = False
    crash_reentry_date: Optional[date] = None
    crash_opt_reopened: bool = False
    crash_opt_reopen_date: Optional[date] = None
    crash_yin_scaled: bool = False
    crash_yin_date: Optional[date] = None
    crash_stop_pct: float = 0.025
    crash_reentry_pct: float = 1.0
    crash_dte: int = 21
    crash_spread_w: int = 15
    crash_drop_thresh: float = 0.005
    crash_etf_stop_pct: float = 0.0
    crash_etf_out: bool = False
    crash_size_mult: float = 1.0
    crash_etf_size: int = 5000
    crash_stop_cooldown: int = 0
    crash_stop_date: Optional[date] = None
    crash_half_pct: float = 0.0625
    crash_yin_pct: float = 0.75
    crash_yin_scaled: bool = False
    crash_yin_date: Optional[date] = None
    crash_green_streak: int = 0
    crash_half_date: Optional[date] = None
    crash_reentry_date: Optional[date] = None
    crash_opt_reopened: bool = False
    crash_opt_reopen_date: Optional[date] = None
    crash_resids: list = field(default_factory=list)
    crash_etf_carry: Optional[dict] = None
    carry_display: dict = field(default_factory=dict)
    crash_green_streak: int = 0
    
    # Additional crash fields for position tracking
    half_etf: float = 0.0
    half_sh: Optional[int] = None
    keep_sh: Optional[int] = None
    yin_etf: float = 0.0
    yin_sh: Optional[int] = None
    yin_keep_sh: Optional[int] = None
    re_entry_spxl: Optional[float] = None
    re_yin_px: Optional[float] = None
    reopened: bool = False
    reopen_k1: Optional[int] = None
    reopen_k2: Optional[int] = None
    reopen_debit: Optional[float] = None
    reopen_sigma: Optional[float] = None
    reopen_date: Optional[date] = None
    opt_closed: bool = False
    resid_entry: Optional[float] = None
    resid_expiry: Optional[date] = None
    resid: Optional[dict] = None
    etf_out: bool = False
    size_mult: float = 1.0
    etf_shares: Optional[int] = None
    
    def to_dict(self) -> dict:
        """Serialize for POSITION_FILE persistence."""
        d = {}
        for k, v in asdict(self).items():
            if isinstance(v, (date, datetime)):
                d[k] = v.isoformat() if v else None
            elif isinstance(v, list):
                d[k] = [
                    {kk: (vv.isoformat() if isinstance(vv, (date, datetime)) else vv) for kk, vv in item.items()}
                    if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                d[k] = v
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PositionState':
        """Deserialize from POSITION_FILE."""
        kwargs = {}
        for k, v in d.items():
            if k in ('active_position_date', 'entry_date', 'peak_price', 'etf_entry_price', 
                     'etf_peak_price', 'mr_entry_date', 'crash_entry_date', 'crash_half_date',
                     'crash_reentry_date', 'crash_opt_reopen_date', 'crash_yin_date',
                     'crash_stop_date', 'trend_opt_entry_date', 'mr_expiry', 'mr_entry_date',
                     'crash_reentry_date', 'crash_yin_date', 'crash_stop_date', 'crash_half_date',
                     'crash_reentry_date', 'crash_opt_reopen_date', 'crash_yin_date',
                     'crash_stop_date', 'crash_half_date', 'crash_reentry_date', 'crash_opt_reopen_date',
                     'crash_yin_date', 'crash_green_streak', 'crash_stop_date'):
                # These might be dates - handle in field-specific logic below
                pass
            if v is None:
                kwargs[k] = None
            elif k.endswith('_date') or k in ('crash_stop_date', 'crash_half_date', 'crash_reentry_date', 
                                                'crash_opt_reopen_date', 'crash_yin_date', 'crash_stop_date',
                                                'trend_opt_entry_date', 'mr_entry_date', 'mr_expiry',
                                                'reopen_date', 'resid_expiry', 'resid_entry'):
                if v:
                    try:
                        kwargs[k] = date.fromisoformat(v) if isinstance(v, str) else v
                    except Exception:
                        kwargs[k] = v
                else:
                    kwargs[k] = None
            elif k == 'crash_resids':
                kwargs[k] = []
                if v:
                    for item in v:
                        new_item = {}
                        for kk, vv in item.items():
                            if kk in ('expiry', 'open') and vv:
                                try:
                                    new_item[kk] = date.fromisoformat(vv) if isinstance(vv, str) else vv
                                except Exception:
                                    new_item[kk] = vv
                            else:
                                new_item[kk] = vv
                        kwargs[k].append(new_item)
            elif k == 'carry_display':
                kwargs[k] = {}
                if v:
                    for kk, vv in v.items():
                        kwargs[k][kk] = vv
            else:
                kwargs[k] = v
        return cls(**kwargs)
    
    def copy(self) -> 'PositionState':
        """Create a deep copy."""
        return PositionState.from_dict(self.to_dict())