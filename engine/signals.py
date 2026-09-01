"""engine.signals: SINGLE SOURCE OF TRUTH for crash signal (收盘对收盘).

Canonical: 崩盘信号为 ^XSP 日线收盘对收盘跌幅 <-0.5% (Close[T]/Close[T-1]-1 同源),
非盘中价对昨收。供 app.py / sim_reports_full.py / golden / helpers 共用。
"""
from __future__ import annotations
from typing import Optional, Tuple


def calc_crash(
    close_t: Optional[float],
    close_t1: Optional[float],
    drop_thresh: float = 0.005,
) -> Tuple[Optional[float], bool]:
    """收盘对收盘纯函数.

    Returns: (xsp_chg_pct or None, is_crash_signal)
    is_crash_signal = close_t1 is not None and chg < -drop_thresh
    与 RULES.md / PositionConfig.drop_thresh (0.005) 同口径。
    """
    if close_t is None or close_t1 is None or close_t1 == 0:
        return None, False
    try:
        chg = (float(close_t) - float(close_t1)) / float(close_t1)
    except Exception:
        return None, False
    signal = chg < -float(drop_thresh)
    return chg, signal


# alias for PositionCore compatibility
def check_crash_signal(close_t, close_t1, drop_thresh=0.005) -> bool:
    _, sig = calc_crash(close_t, close_t1, drop_thresh)
    return sig
