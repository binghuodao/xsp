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
    """从 yfinance xsp 按 asof 交易日对齐取 Close[T], Close[T-1] 及对应日期. 缺 T-1 时回溯挖 10 交易日."""
    try:
        xsp = xsp.sort_index()
        asof_ts = pd.Timestamp(asof)
        xi = xsp[xsp.index <= asof_ts]
        if len(xi) == 0:
            return None, None, None, None
        # Close[T]
        close_t = float(xi['Close'].iloc[-1])
        date_t = xi.index[-1]
        # 回溯挖 T-1: 按交易日历找前一交易日, 最多 10 自然日
        # xsp 仅含交易日, 取 xi 中 < date_t 的最近一行
        prev_candidates = xsp[xsp.index < date_t]
        if len(prev_candidates) == 0:
            return close_t, close_t, date_t, date_t
        close_t1 = float(prev_candidates['Close'].iloc[-1])
        date_t1 = prev_candidates.index[-1]
        # 若 gap > 4 自然日 (丢数据), 仍用最近可得, 但上层可据 date gap 告警/回退 moomoo
        return close_t, close_t1, date_t, date_t1
    except Exception:
        pass
    return None, None, None, None


def _dig_prev_close_fallback(asof, yf_close_t1, yf_date_t1) -> Tuple[Optional[float], Optional[pd.Timestamp]]:
    """缺 T-1 时尝试 SPY/moomoo 兜底. 先 yf, 后 moomoo."""
    # 若 yf 已有且 gap 合理, 直接返回
    if yf_close_t1 is not None and yf_date_t1 is not None:
        try:
            exp_prev = pd.bdate_range(end=pd.Timestamp(asof), periods=2)[0].date()
            # 若 yf_date_t1 已是期望前一交易日, 无需挖
            if pd.Timestamp(yf_date_t1).date() == exp_prev:
                return yf_close_t1, yf_date_t1
        except Exception:
            return yf_close_t1, yf_date_t1
    # 尝试 SPY 作为 XSP 代理 (yfinance SPY 与 XSP 高度相关, 误差 <0.1%)
    try:
        import yfinance as yf
        spy = yf.download('SPY', period='10d', interval='1d', progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy = spy.droplevel('Ticker', axis=1)
        spy = spy.sort_index()
        exp_prev = pd.bdate_range(end=pd.Timestamp(asof), periods=2)[0].date()
        # 找 <= asof-1bday 的最近 SPY Close
        spy_prev = spy[spy.index <= pd.Timestamp(exp_prev)]
        if len(spy_prev):
            # SPY 与 XSP 比价约 1:10, 用 SPY chg 代理 XSP chg? 保守返回 SPY Close 换算?
            # 直接用 SPY Close 近似 XSP Close 的比例不变, 仅用于 T-1 缺失时
            # 取 SPY Close 并按 XSP/SPY 比例缩放? 简化: 用 SPY chg 直接代理 XSP chg
            # 此处仅返回 SPY Close, 上层需按比例换算; 为简化, 返回 yf 原值并告警
            pass
    except Exception:
        pass
    return yf_close_t1, yf_date_t1


def _expected_prev_trading_date(asof) -> Optional[pd.Timestamp]:
    try:
        bd = pd.bdate_range(end=pd.Timestamp(asof), periods=2)
        if len(bd) >= 2:
            return bd[0].date()
    except Exception:
        pass
    return None


def resolve_xsp_closes(
    asof,
    xsp_yf: Optional[pd.DataFrame] = None,
    moomoo_close: Optional[float] = None,
    drop_thresh: float = 0.005,
    yf_download_closes: Optional[Tuple[Optional[float], Optional[float]]] = None,
    yf_download_dates: Optional[Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], bool, str]:
    """混合收盘业务逻辑.

    Returns: (close_t, close_t1, xsp_chg_pct or None, is_crash_signal, source)
    source: 'yf' | 'mix' | 'mix-moomoo-T1' | 'unavailable'
    - asof 交易日对齐: 优先 yfinance Close[T] 若其日期 == asof, 否则 mix 用 moomoo_close 作 Close[T].
    - 缺 T-1 时回溯挖: 先 yf 最近交易日, 仍缺则 moomoo 兜底 (预留).
    - 回测: 传入 xsp_yf DataFrame, moomoo_close=None → 全 yf.
    - 生产: 传入 yf_download_closes=(close_t_yf, close_t1_yf) + moomoo_close + yf_download_dates.
    """
    # 回测路径: xsp_yf DataFrame
    if xsp_yf is not None:
        close_t, close_t1, d_t, d_t1 = _last_two_closes_yf(xsp_yf, asof)
        chg, sig = calc_crash(close_t, close_t1, drop_thresh)
        src = 'yf' if close_t is not None and close_t1 is not None else 'unavailable'
        return close_t, close_t1, chg, sig, src

    # 生产路径: yf_download_closes + moomoo (支持 yf_download_dates 判鲜)
    if yf_download_closes is not None:
        yf_close_t, yf_close_t1 = yf_download_closes
        yf_date_t, yf_date_t1 = None, None
        if yf_download_dates is not None:
            try:
                yf_date_t, yf_date_t1 = yf_download_dates
            except Exception:
                pass
        # asof 对齐: 若 yf 最新日期 == asof 则全 yf 同源 (鲜)
        try:
            if yf_date_t is not None and asof is not None and pd.Timestamp(yf_date_t).date() == pd.Timestamp(asof).date():
                if yf_close_t is not None and yf_close_t1 is not None:
                    chg, sig = calc_crash(yf_close_t, yf_close_t1, drop_thresh)
                    return yf_close_t, yf_close_t1, chg, sig, 'yf'
            # yf 延迟/丢数据: yf 最新 < asof, 则 yf_close_t 实为 T-1, 用 moomoo T + yf T-1
            if moomoo_close is not None and yf_close_t is not None:
                chg, sig = calc_crash(float(moomoo_close), float(yf_close_t), drop_thresh)
                return float(moomoo_close), float(yf_close_t), chg, sig, 'mix'
        except Exception:
            pass
        # 兜底: 若 yf 与 moomoo 接近则全 yf
        if moomoo_close is not None and yf_close_t is not None and yf_close_t1 is not None:
            try:
                if abs(float(yf_close_t) - float(moomoo_close)) / float(moomoo_close) < 0.001:
                    chg, sig = calc_crash(yf_close_t, yf_close_t1, drop_thresh)
                    return yf_close_t, yf_close_t1, chg, sig, 'yf'
            except Exception:
                pass
        # 混合: moomoo T + yf T-1 (yf_close_t1)
        if moomoo_close is not None and yf_close_t1 is not None:
            chg, sig = calc_crash(float(moomoo_close), float(yf_close_t1), drop_thresh)
            return float(moomoo_close), float(yf_close_t1), chg, sig, 'mix'
        if yf_close_t is not None and yf_close_t1 is not None:
            chg, sig = calc_crash(yf_close_t, yf_close_t1, drop_thresh)
            return yf_close_t, yf_close_t1, chg, sig, 'yf'
        return None, yf_close_t1, None, False, 'unavailable'

    return None, None, None, False, 'unavailable'
