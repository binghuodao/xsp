"""Pure pricing/math functions shared by production and backtest.

No external state, no side effects. Only depends on pricing.black_scholes.
"""

import pricing
import pandas as pd


def _bs_spread(price: float, k1: int, k2: int, T: float, sigma: float) -> float:
    """Black-Scholes value of a CALL spread (long k1, short k2)."""
    if T <= 0:
        T = 1 / 365.0
    if not sigma or sigma <= 0:
        sigma = 0.20
    return (
        pricing.black_scholes(price, k1, T, 0.05, sigma, 'C')
        - pricing.black_scholes(price, k2, T, 0.05, sigma, 'C')
    )


def _bs_call(price: float, K: int, T: float, sigma: float) -> float:
    """Black-Scholes value of a single CALL."""
    if T <= 0:
        T = 1 / 365.0
    if not sigma or sigma <= 0:
        sigma = 0.20
    return pricing.black_scholes(price, K, T, 0.05, sigma, 'C')


def _etf_stop_fill(entry_px: float, asof, spxl_ohlc) -> float | None:
    """SPXL 止损限价单触价结算 (生产开仓提示挂 -7.5% 触发 + 2.5% 缓冲限价单):
    
    触发线 = entry*0.925, 限价 = 触发线*0.975 (与 app 提示一致).
    盘中跌破触发线(S)时市价仍 ≥ 限价(L) → 按触发线成交 (15/16 笔与此一致);
    跳空 O ∈ (L, S] → 按开盘价成交 (与现口径一致);
    跳空 O < L 且当日 High ≥ L → 盘中回抽按限价 L 成交;
    跳空 O < L 且当日 High < L → 当日不成交, 返回 None (待同价续持并入下一崩盘开仓);
    兜底按收盘价 (理论不触发).
    
    Args:
        entry_px: SPXL entry price
        asof: date for lookup
        spxl_ohlc: SPXL DataFrame with OHLC columns, indexed by date
    
    Returns:
        Fill price, or None if not filled same day (carry to next crash open)
    """
    line = entry_px * 0.925
    limit = line * 0.975
    
    try:
        row = spxl_ohlc.loc[pd.Timestamp(asof)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
    except KeyError:
        # Fallback: use last available row up to asof
        sp = spxl_ohlc[spxl_ohlc.index <= pd.Timestamp(asof)]
        if len(sp) == 0:
            return None
        row = sp.iloc[-1]
    
    o = float(row['Open'])
    h = float(row['High'])
    l = float(row['Low'])
    c = float(row['Close'])
    
    if o <= line:
        if o >= limit:
            return o  # gap open in (L, S] → fill at open
        return limit if h >= limit else None  # gap below L → fill at L if intraday recovers, else None
    
    if l <= line:
        return line  # intraday touch → fill at trigger
    
    return c  # fallback to close (shouldn't trigger)


def _leg_round(shares: int, frac: float) -> int:
    """Round shares * fraction to integer, floor at 0."""
    return max(round(shares * frac), 0)


def compute_carry_blend(carry_entry: float, carry_shares: int, fresh_entry: float, fresh_shares: int) -> tuple[int, float]:
    """Compute blended entry after merging carry into fresh position.
    
    Returns:
        (total_shares, blended_entry_price)
    """
    total_shares = carry_shares + fresh_shares
    if total_shares == 0:
        return 0, fresh_entry
    blended = (carry_entry * carry_shares + fresh_entry * fresh_shares) / total_shares
    return total_shares, blended


def evaluate_standalone_option(state, price: float, asof, policy) -> list[dict]:
    """Evaluate standalone option TP/SL/expiry for crash position.
    
    Args:
        state: PositionState with crash_resids
        price: XSP price
        asof: current date
        policy: CrashPolicy with opt_standalone, opt_tp, opt_sl_resid, etc.
    
    Returns:
        List of TradeEvent dicts for any triggered standalone exits
    """
    events = []
    
    for t in state.crash_resids:
        if not t or t.get('indep') is not True:
            continue
        
        days_left = (t['expiry'] - asof).days
        T_rem = max(days_left / 365.0, 1 / 365.0)
        
        close_d = _bs_spread(price, t['k1'], t['k2'], T_rem, t['sigma'])
        debit = t['debit']
        
        tp_line = debit * policy.opt_tp
        sl_line = debit * policy.opt_sl_resid if policy.opt_sl_resid > 0 else 0
        
        if close_d >= tp_line:
            events.append({
                'kind': 'CRASH',
                'type': 'RESIDUAL_SETTLE',
                'reason': 'TP',
                'price': price,
                'spxl_price': None,  # filled by caller
                'details': {
                    'trade_n': t.get('trade_n'),
                    'opt_pnl': max((close_d - debit), -debit) * 100 * t.get('size_mult', 1.0) * policy.opt_mult,
                    'resid': None,
                    'opt_closed': True,
                }
            })
            t['opt_closed'] = True
            t['resid'] = None
        
        elif sl_line > 0 and close_d <= sl_line and (not policy.opt_sl_adaptive or price <= t['entry']):
            events.append({
                'kind': 'CRASH',
                'type': 'RESIDUAL_SETTLE',
                'reason': 'SL',
                'price': price,
                'spxl_price': None,
                'details': {
                    'trade_n': t.get('trade_n'),
                    'opt_pnl': max((close_d - debit), -debit) * 100 * t.get('size_mult', 1.0) * policy.opt_mult,
                    'resid': None,
                    'opt_closed': True,
                }
            })
            t['opt_closed'] = True
            t['resid'] = None
        
        elif policy.opt_sl_decay > 0:
            hold_days = (asof - t.get('open', asof)).days if t.get('open') else 0
            total_life = (t['expiry'] - t.get('open', asof)).days if t.get('open') else 0
            if total_life > 0 and hold_days > total_life * policy.opt_sl_decay and close_d < debit * policy.opt_sl_decay:
                events.append({
                    'kind': 'CRASH',
                    'type': 'RESIDUAL_SETTLE',
                    'reason': 'TIME_DECAY',
                    'price': price,
                    'spxl_price': None,
                    'details': {
                        'trade_n': t.get('trade_n'),
                        'opt_pnl': max((close_d - debit), -debit) * 100 * t.get('size_mult', 1.0) * policy.opt_mult,
                        'resid': None,
                        'opt_closed': True,
                    }
                })
                t['opt_closed'] = True
                t['resid'] = None
        
        elif days_left <= 0:
            events.append({
                'kind': 'CRASH',
                'type': 'RESIDUAL_SETTLE',
                'reason': 'EXPIRY',
                'price': price,
                'spxl_price': None,
                'details': {
                    'trade_n': t.get('trade_n'),
                    'opt_pnl': max((close_d - debit), -debit) * 100 * t.get('size_mult', 1.0) * policy.opt_mult,
                    'resid': None,
                    'opt_closed': True,
                }
            })
            t['opt_closed'] = True
            t['resid'] = None
    
    return events