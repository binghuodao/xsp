"""engine.data: 业务逻辑 — XSP 收盘取数 (混合收盘).

Production: Close[T] 优先 moomoo 实时收盘 (当日触发), Close[T-1] 来自 yfinance.
yfinance Close[T] 到账后自动切回全 yf 同源. 回测全 yf.

Single source for app.py / sim_reports_* / helpers.
"""
from __future__ import annotations
from typing import Optional, Tuple
import pandas as pd

from engine.signals import calc_crash


def _last_two_closes_yf(xsp: pd.DataFrame, asof) -> Tuple[Optional[float], Optional[float], Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """从 yfinance xsp 按 asof 交易日对齐取 Close[T], Close[T-1] 及对应日期."""
    try:
        xsp = xsp.sort_index()
        xi = xsp[xsp.index <= pd.Timestamp(asof)]
        if len(xi) >= 2:
            return float(xi['Close'].iloc[-1]), float(xi['Close'].iloc[-2]), xi.index[-1], xi.index[-2]
        if len(xi) == 1:
            c = float(xi['Close'].iloc[-1])
            return c, c, xi.index[-1], xi.index[-1]
    except Exception:
        pass
    return None, None, None, None


def resolve_xsp_closes(
    asof,
    xsp_yf: Optional[pd.DataFrame] = None,
    moomoo_close: Optional[float] = None,
    drop_thresh: float = 0.005,
    yf_download_closes: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], bool, str]:
    """混合收盘业务逻辑.

    Returns: (close_t, close_t1, xsp_chg_pct or None, is_crash_signal, source)
    source: 'yf' | 'mix' | 'unavailable'
    - asof 交易日对齐: 优先 yfinance Close[T] 若其日期 == asof, 否则 mix 用 moomoo_close 作 Close[T].
    - 回测: 传入 xsp_yf DataFrame, moomoo_close=None → 全 yf.
    - 生产: 传入 yf_download_closes=(close_t_yf, close_t1_yf) + moomoo_close.
    """
    # 回测路径: xsp_yf DataFrame
    if xsp_yf is not None:
        close_t, close_t1, d_t, d_t1 = _last_two_closes_yf(xsp_yf, asof)
        chg, sig = calc_crash(close_t, close_t1, drop_thresh)
        src = 'yf' if close_t is not None and close_t1 is not None else 'unavailable'
        return close_t, close_t1, chg, sig, src

    # 生产路径: yf_download_closes + moomoo
    if yf_download_closes is not None:
        yf_close_t, yf_close_t1 = yf_download_closes
        # 尝试判断 yf Close[T] 是否已到账: 需要 yf index 日期, 但此处仅有值, 保守用 moomoo 优先
        # 若 yf_close_t 与 moomoo_close 接近 (<0.1% 且 yf 数据存在), 认为 yf 已到账用全 yf
        if moomoo_close is not None and yf_close_t is not None and yf_close_t1 is not None:
            try:
                if abs(float(yf_close_t) - float(moomoo_close)) / float(moomoo_close) < 0.001:
                    # yf 已收盘, 用全 yf 同源
                    chg, sig = calc_crash(yf_close_t, yf_close_t1, drop_thresh)
                    return yf_close_t, yf_close_t1, chg, sig, 'yf'
            except Exception:
                pass
        # 混合: moomoo Close[T] + yf Close[T-1]
        if moomoo_close is not None and yf_close_t1 is not None:
            chg, sig = calc_crash(float(moomoo_close), float(yf_close_t1), drop_thresh)
            return float(moomoo_close), float(yf_close_t1), chg, sig, 'mix'
        # 兜底全 yf
        if yf_close_t is not None and yf_close_t1 is not None:
            chg, sig = calc_crash(yf_close_t, yf_close_t1, drop_thresh)
            return yf_close_t, yf_close_t1, chg, sig, 'yf'
        return None, yf_close_t1, None, False, 'unavailable'

    return None, None, None, False, 'unavailable'
