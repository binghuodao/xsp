"""engine.report: SINGLE SOURCE OF TRUTH for XSP report generation.

Pure logic for building market report lines/score/direction.
Used by app.py (production), sim_reports_full.py (backtest), and UI (index_update).

No side effects: no yfinance/moomoo/socketio/telegram. Caller supplies inputs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import pandas as pd

from engine.signals import calc_crash


def _s5(v: float) -> int:
    return int(round(v / 5) * 5)


def _score_ts(v, th):
    for t, s in zip(th, [100, 75, 50, 25, 0]):
        if v >= t:
            return s
    return 0


@dataclass
class ReportInputs:
    title: str
    dte_adj: int
    price: float
    hs: Dict[str, Any]
    is_crash_signal: bool
    is_mr_signal: bool
    xsp_chg_pct: float
    # for trend composite
    # price, hs already contain needed fields
    # previous report state for close_lines (not needed for pure)
    # options/watchlist for close_lines handled outside


@dataclass
class ReportResult:
    score: int
    icon: str
    slbl: str
    is_trend: bool
    direction: Optional[str]
    reason: Optional[str]
    trend_entry_blocked: bool
    lines_header: list  # first lines before 10Y etc.


def build_score_direction(price: float, hs: Dict[str, Any], active_position_date=None, trend_opt_expiry=None) -> Tuple[int, str, str, bool, Optional[str], Optional[str], bool]:
    """Pure: compute score, is_trend, direction, reason, trend_entry_blocked."""
    ema20 = hs.get("ema_20", 0)
    bbl = hs.get("support", 0)
    bbu = hs.get("resistance", 0)
    bw = bbu - bbl if (bbl and bbu and bbu > bbl) else 1

    W = {'adx': .3, 'er': .2, 'bbw': .15, 'dev': .15, 'vr': .1}
    T = {'adx': [30, 25, 20, 15, 0], 'er': [.7, .55, .35, .2, 0],
         'bbw': [45, 30, 18, 10, 0], 'dev': [3.0, 1.5, 0.8, 0.3, 0],
         'vr': [2.0, 1.3, .8, .5, 0]}
    total = 5
    for k, w in W.items():
        v = hs.get(k)
        if v is None:
            total += 5
            continue
        val = abs(v) if k == 'dev' else v
        total += _score_ts(val, T.get(k, [0])) * w
    score = round(total)
    icon = '🟢' if score >= 65 else '🟡' if score >= 35 else '🔴'
    slbl = 'Trending' if score >= 65 else 'Mixed' if score >= 35 else 'Ranging'
    is_trend = score >= 50

    # Direction fusion — exact copy of app.py:516-564
    dlow = (price - bbl) / bw * 100 if bw else 0
    di_diff = hs.get('di_diff', 0)
    vix_pct = hs.get('vix_percentile', 50)
    adx = hs.get('adx', 18)
    rsi_14 = hs.get('rsi_14', 50)
    atr14 = hs.get("atr_14")
    if atr14 and atr14 > 0:
        near_threshold = atr14 * 0.60
    else:
        near_threshold = bw * 0.10
    near_top = (bbu - price) < near_threshold if bbu else False
    near_bottom = (price - bbl) < near_threshold if bbl else False
    near_bb_overall = near_top or near_bottom
    direction, reason = None, None
    if not near_bb_overall and is_trend:
        if di_diff > 0 and hs.get('di_diff_prev', 0) > 0 and price > hs.get('sma50', 0):
            if hs.get('sma50', 0) < hs.get('sma200', 0) and price > hs.get('sma200', 0) and hs.get('sma50_slope', 999) < 2:
                direction, reason = None, 'BB 中段'
            else:
                direction, reason = 'CALL', f'DI+({di_diff:.2f})'
        elif di_diff > 0:
            direction, reason = None, 'BB 中段'
        elif di_diff < 0:
            direction, reason = None, 'BB 中段'
        else:
            direction, reason = None, 'BB 中段'
    elif near_top and score >= 50 and vix_pct > 75:
        direction, reason = None, 'BB 中段'
    elif near_bottom and score >= 50 and vix_pct > 75:
        direction, reason = 'CALL', f'贴BB下+VIX({vix_pct:.0f}%)'
    elif near_top and di_diff > 0:
        direction, reason = None, 'BB 中段'
    elif near_bottom and False:
        pass
    elif near_top and score >= 50:
        direction, reason = None, 'BB 中段'
    elif near_bottom and score >= 35 and (adx < 25 or rsi_14 < 35) and not (adx >= 25 and price < hs.get('sma50', 0)):
        direction, reason = 'CALL', f'贴BB下轨({dlow:.0f}%)'
    elif near_top or near_bottom:
        direction, reason = None, 'BB 中段'
    else:
        direction, reason = None, 'BB 中段'

    # 趋势高位过滤: BB%>80 暂缓
    trend_entry_blocked = (is_trend and direction == 'CALL' and dlow > 80
                           and active_position_date is None and trend_opt_expiry is None)

    return score, icon, slbl, is_trend, direction, reason, trend_entry_blocked


def build_report_header(title: str, price: float, hs: Dict[str, Any], direction: Optional[str], reason: Optional[str], score: int, icon: str, slbl: str, now_et_str: str) -> list:
    """Build first lines of report (pure)."""
    ema20 = hs.get("ema_20", 0)
    bbl = hs.get("support", 0)
    bbu = hs.get("resistance", 0)
    lines = [f"{title} — {now_et_str}",
             "━━━━━━━━━━━━━━━━━━━━━",
             f"{icon} 综合 {score} / {slbl}",
             f"ADX {hs.get('adx',0):.1f} | ER {hs.get('er',0):.2f} | BBW {hs.get('bbw',0):.1f}% | Dev {hs.get('dev',0):+.1f}% | VR {hs.get('vr',0):.1f}x",
             f"VIX {hs.get('vix',0):.1f} ({hs.get('vix_rank',0):.0f}%) | DI {hs.get('di_diff',0):+.2f}",
             f"EMA20 ${ema20:.2f} | 现价 ${price:.2f}",
             f"BBL ${bbl:.2f} | BBU ${bbu:.2f} | ATR14 ${hs.get('atr_14',0):.2f}",
             "", f"→ 方向: {direction} ({reason})" if direction else "→ BB中段，不开仓，等待方向明确", ""]
    return lines


def build_full_report(title: str, price: float, hs: Dict[str, Any], direction: Optional[str], reason: Optional[str], score: int, icon: str, slbl: str, now_et_str: str, xsp_dbg: str = "", y10_level=None, y10_20d=None, y10_gate_active: bool = False, y10_gate_pp: float = 0.0, close_lines: list = None) -> list:
    """Full report lines — header + XSP dbg + 10Y + close_lines. Pure, no side effects."""
    lines = build_report_header(title, price, hs, direction, reason, score, icon, slbl, now_et_str)
    # XSP 调试明细
    if xsp_dbg:
        lines.append(f"🔍 {xsp_dbg}")
    # 10Y
    if y10_level is None or y10_20d is None:
        lines.append("⚠️ 10Y 不可用（闸门自动关，崩盘照常）")
    else:
        lines.append(f"10Y {y10_level:.3f}% (20d {y10_20d:+.2f}%) | 利率闸门 {'🚫 开(拦截崩盘)' if y10_gate_active else '✓ 关'}")
    if close_lines:
        lines.append("")
        lines.append("━━━ 平仓提示 ━━━")
        lines.extend(close_lines)
    return lines
