"""PositionEngine: core position state machine for all three layers.

Single source of truth for position state, PnL, and trade events.
Used by both production (app.py) and backtest (sim_reports_full.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional
import copy

import pricing

from engine.policies import CrashPolicy, TrendPolicy, MRPolicy
from engine.pricing import (
    _bs_spread,
    _bs_call,
    _etf_stop_fill,
    _leg_round,
    compute_carry_blend,
    evaluate_standalone_option,
)


# ════════════════════════════════════════════════════════════════════════
# Data Classes
# ════════════════════════════════════════════════════════════════════════

@dataclass
class TradeEvent:
    """Structured trade event for ledger/backtest."""
    kind: str                    # 'TREND' | 'MR' | 'CRASH'
    type: str                    # 'OPEN' | 'CLOSE' | 'ROLL' | 'HALF' | 'YIN' | 'REENTRY' | 'RESIDUAL_SETTLE'
    date: date
    price: float                 # XSP price
    spxl_price: float
    details: dict                # All relevant fields: trade_n, pnl, result, etc.


@dataclass
class PositionState:
    """All mutable position state — mirrors app.py globals + backtest ledger state."""
    
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
    
    # MR
    mr_entry_date: Optional[date] = None
    mr_entry_price: Optional[float] = None
    mr_etf_entry_price: Optional[float] = None
    mr_k: Optional[int] = None
    mr_sigma: Optional[float] = None
    mr_expiry: Optional[date] = None
    mr_opt_entry: Optional[float] = None
    
    # CRASH
    crash_entry_date: Optional[date] = None
    crash_entry_price: Optional[float] = None
    crash_k1: Optional[int] = None
    crash_k2: Optional[int] = None
    crash_debit: Optional[float] = None
    crash_sigma: Optional[float] = None
    crash_etf_entry: Optional[float] = None
    crash_half_date: Optional[date] = None
    crash_reentry_date: Optional[date] = None
    crash_opt_reopen_date: Optional[date] = None
    crash_resids: list = field(default_factory=list)
    crash_etf_scaled: bool = False
    crash_reentry: bool = False
    crash_opt_reopened: bool = False
    crash_exit_mode: str = 'V10'
    crash_half_pct: float = 0.0625
    crash_yin_pct: float = 0.75
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
    risk_off_active: bool = False
    
    # Carry (same-price continuation)
    crash_etf_carry: Optional[dict] = None      # {'shares', 'entry', 'fresh_entry'}
    carry_display: dict = field(default_factory=dict)  # trade_n -> {'fresh', 'blend'}
    
    # Half/yin/reentry tracking
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
    opt_pnl: float = 0.0
    etf_pnl: float = 0.0
    etf_shares: Optional[int] = None
    etf_out: bool = False
    size_mult: float = 1.0
    
    def state_fp(self) -> tuple:
        """Position fingerprint for change detection (matches sim_reports_full.py)."""
        return (
            self.prev_report_direction,
            self.active_position_date,
            self.trend_opt_expiry, self.trend_opt_strike, self.trend_opt_strike2,
            self.mr_entry_date,
            self.crash_entry_date,
            self.crash_etf_scaled,
            self.crash_reentry,
            self.crash_etf_out,
            self.crash_yin_scaled,
        )
    
    def merge_carry(self, fresh_entry_px: float, total_shares: int) -> tuple[int, float]:
        """Apply carry to new crash open. Returns (total_shares, blended_entry)."""
        if self.crash_etf_carry:
            carry = self.crash_etf_carry
            add_shares = max(total_shares - carry['shares'], 0)
            blended = (carry['entry'] * carry['shares'] + fresh_entry_px * add_shares) / total_shares
            # Note: carry_display update happens in engine after trade_n is known
            self.crash_etf_carry = None
            return total_shares, blended
        return total_shares, fresh_entry_px


# ════════════════════════════════════════════════════════════════════════
# Position Engine
# ════════════════════════════════════════════════════════════════════════

class PositionEngine:
    """Core position engine — processes evening reports and emits TradeEvents."""
    
    def __init__(
        self,
        crash: CrashPolicy,
        trend: TrendPolicy,
        mr: MRPolicy,
    ):
        self.state = PositionState()
        self.crash = crash
        self.trend = trend
        self.mr = mr
        
        # Ledger for backtest compatibility
        self.ledger = {'TREND': [], 'MR': [], 'CRASH': []}
        self._open_ref = {'TREND': None, 'MR': None, 'CRASH': None}
        self.ETF_SIZE = {'TREND': trend.etf_size, 'MR': mr.etf_size, 'CRASH': 5000}
    
    # ── Public API ───────────────────────────────────────────────────────
    
    def on_evening_report(
        self,
        price: float,
        spxl_price: float,
        vix: float,
        report: dict,
        risk_gate_active: bool,
        bounce_ok: bool,
        spxl_ohlc=None,
    ) -> list[TradeEvent]:
        """Process evening report, update position state, return trade events."""
        
        events = []
        prev_fp = self.state.state_fp()
        now_fp = None
        
        # 1. CRASH signal check (with bounce + risk gate)
        crash_signal = self._check_crash_signal(report, risk_gate_active, bounce_ok)
        
        # 2. TREND layer
        if self.state.trend_opt_expiry is not None:
            if self._trend_should_close(report):
                events.append(self._close_trend(price, spxl_price, report))
        if self._trend_should_open(report, crash_signal):
            events.append(self._open_trend(price, spxl_price))
        
        # 3. MR layer
        if self.state.mr_entry_date is not None:
            if self._mr_should_close(report, price):
                events.append(self._close_mr(price, spxl_price, report))
        if self._mr_should_open(report, crash_signal):
            events.append(self._open_mr(price, spxl_price, vix))
        
        # 4. CRASH layer
        if self.state.crash_entry_date is not None:
            events.extend(self._update_crash(price, spxl_price, vix, report, risk_gate_active, spxl_ohlc))
        if self._crash_should_open(crash_signal, price, spxl_price, vix, report):
            events.append(self._open_crash(price, spxl_price, vix, report))
        
        # 5. Option standalone evaluation (tp/tp-v9)
        if self.crash.opt_standalone in ('tp', 'tp-v9'):
            ev = evaluate_standalone_option(self.state, price, report.get('asof', date.today()), self.crash)
            events.extend(ev)
        
        # 6. Residual settlement (V9/V10/V11 shared-exit residuals)
        events.extend(self._settle_residuals(price, spxl_price, report))
        
        # Update fingerprint
        now_fp = self.state.state_fp()
        
        return events
    
    def get_open_positions(self) -> dict:
        """Return currently open positions by layer."""
        return {
            'TREND': self._cur_trade('TREND'),
            'MR': self._cur_trade('MR'),
            'CRASH': self._cur_trade('CRASH'),
        }
    
    def get_closed_trades(self) -> list:
        """Return all closed trades from ledger."""
        return [t for layer in self.ledger.values() for t in layer if t.get('close')]
    
    # ── TREND Layer ─────────────────────────────────────────────────────
    
    def _trend_should_close(self, report: dict) -> bool:
        """Check if trend should close (direction change or BB middle)."""
        return (
            self.state.trend_opt_expiry is not None and
            (report.get('direction') != 'CALL' or 'BB中段' in str(report.get('close_alerts', [])))
        )
    
    def _trend_should_open(self, report: dict, crash_signal: bool) -> bool:
        """Check if trend should open."""
        if self.state.trend_opt_expiry is not None:
            return False
        if crash_signal or report.get('direction') != 'CALL':
            return False
        return report.get('direction') == 'CALL'
    
    def _open_trend(self, price: float, spxl_price: float) -> TradeEvent:
        """Open trend position."""
        n = self._open_trade('TREND', price)
        t = self._cur_trade('TREND')
        t['etf_entry'] = spxl_price
        t['etf_shares'] = max(round(self.trend.etf_size / spxl_price), 1)
        t['etf_pnl'] = 0.0
        t['opt_pnl'] = 0.0
        return TradeEvent('TREND', 'OPEN', date.today(), price, spxl_price, {'n': n})
    
    def _close_trend(self, price: float, spxl_price: float, report: dict) -> TradeEvent:
        """Close trend position."""
        n = self._cur_trade('TREND')['n'] if self._cur_trade('TREND') else None
        t = self._cur_trade('TREND')
        
        # ETF PnL
        if t and t.get('etf_entry'):
            t['etf_pnl'] = (t.get('etf_shares') or 0) * (spxl_price - t['etf_entry'])
        
        # Option PnL (spread close)
        if t and t.get('trend_opt_entry'):
            T_rem = max((t['trend_opt_expiry'] - date.today()).days / 365.0, 1 / 365.0)
            close_d = _bs_spread(price, t['trend_opt_strike'], t['trend_opt_strike2'], T_rem, t.get('trend_opt_sigma'))
            t['opt_pnl'] = max((close_d - t['trend_opt_entry']), -t['trend_opt_entry']) * 100
        
        # Result classification
        alerts = report.get('close_alerts', []) or []
        if 't+30' in str(alerts): result = 't+30强制平仓'
        elif '滚动CALL价差已平仓' in str(alerts): result = '趋势结束平价差'
        elif '跟踪' in str(alerts): result = '跟踪-3%触发'
        elif '入场硬止损' in str(alerts): result = '入场硬止损-2%'
        elif '方向已由' in str(alerts): result = '方向转变'
        elif 'BB中段' in str(alerts): result = 'BB中段/综合分不足'
        else: result = '方向转空'
        
        self._close_trade('TREND', price, result, spxl_price)
        return TradeEvent('TREND', 'CLOSE', date.today(), price, spxl_price, {'n': n, 'result': result})
    
    # ── MR Layer ────────────────────────────────────────────────────────
    
    def _mr_should_open(self, report: dict, crash_signal: bool) -> bool:
        if self.state.mr_entry_date is not None:
            return False
        if crash_signal:
            return False
        hs = report.get('historical_stats', {})
        return hs.get('rsi_14', 50) < self.mr.rsi_thresh and hs.get('vix', 0) > self.mr.vix_thresh
    
    def _mr_should_close(self, report: dict, price: float) -> bool:
        if self.state.mr_entry_date is None:
            return False
        if 'MR首阳' in str(report.get('close_alerts', [])):
            return True
        if 'MR跌穿' in str(report.get('close_alerts', [])):
            return True
        mr_days = report.get('mr_days', 0)
        return mr_days >= self.mr.force_days
    
    def _open_mr(self, price: float, spxl_price: float, vix: float) -> TradeEvent:
        n = self._open_trade('MR', price)
        t = self._cur_trade('MR')
        t['etf_entry'] = spxl_price
        t['etf_shares'] = max(round(self.mr.etf_size / spxl_price), 1)
        t['mr_k'] = self._s5(price)
        t['mr_sigma'] = vix / 100.0
        t['mr_expiry'] = date.today() + timedelta(days=self.mr.dte)
        t['mr_opt_entry'] = _bs_call(price, t['mr_k'], self.mr.dte / 365.0, t['mr_sigma'])
        return TradeEvent('MR', 'OPEN', date.today(), price, spxl_price, {'n': n})
    
    def _close_mr(self, price: float, spxl_price: float, report: dict) -> TradeEvent:
        n = self._cur_trade('MR')['n'] if self._cur_trade('MR') else None
        t = self._cur_trade('MR')
        
        if t:
            if t.get('mr_k') is not None and t.get('mr_opt_entry') is not None:
                T_rem = max((t['mr_expiry'] - date.today()).days / 365.0, 1 / 365.0)
                exit_opt = _bs_call(price, t['mr_k'], T_rem, t.get('mr_sigma'))
                t['opt_pnl'] = max((exit_opt - t['mr_opt_entry']), -t['mr_opt_entry']) * 100
            if t.get('etf_entry'):
                t['etf_pnl'] = (t.get('etf_shares') or 0) * (spxl_price - t['etf_entry'])
        
        if 'MR首阳' in str(report.get('close_alerts', [])):
            result = '首阳+0.3%'
        elif 'MR跌穿' in str(report.get('close_alerts', [])):
            result = '止损-2%'
        else:
            result = '3天强制平'
        
        self._close_trade('MR', price, result, spxl_price)
        return TradeEvent('MR', 'CLOSE', date.today(), price, spxl_price, {'n': n, 'result': result})
    
    def _s5(self, v: float) -> int:
        """Round strike to nearest 5."""
        return int(round(v / 5) * 5)
    
    # ── CRASH Layer ─────────────────────────────────────────────────────
    
    def _check_crash_signal(self, report: dict, risk_gate_active: bool, bounce_ok: bool) -> bool:
        """Determine if crash signal triggers."""
        xsp_chg = report.get('xsp_chg_pct', 0)
        if not bounce_ok:
            return False
        if risk_gate_active:
            return False
        if xsp_chg < -self.crash.drop_thresh and report.get('direction') == 'CALL':
            return True
        return False
    
    def _crash_should_open(self, crash_signal: bool, price: float, spxl_price: float, vix: float, report: dict) -> bool:
        if not crash_signal:
            return False
        if self.state.crash_entry_date is not None:
            return False
        return True
    
    def _open_crash(self, price: float, spxl_price: float, vix: float, report: dict) -> TradeEvent:
        n = self._open_trade('CRASH', price)
        t = self._cur_trade('CRASH')
        
        # Handle carry merge
        if self.state.crash_etf_carry:
            carry = self.state.crash_etf_carry
            total_shares = max(round(self.ETF_SIZE['CRASH'] / spxl_price), 1)
            add_shares = max(total_shares - carry['shares'], 0)
            t['etf_entry'] = (carry['entry'] * carry['shares'] + spxl_price * add_shares) / total_shares
            self.state.carry_display[n] = {'fresh': carry['fresh_entry'], 'blend': t['etf_entry']}
            self.state.crash_etf_carry = None
        else:
            t['etf_entry'] = spxl_price
            t['etf_shares'] = max(round(self.ETF_SIZE['CRASH'] / spxl_price), 1)
        
        # Option legs
        k1 = self._s5(price - self.crash.spread_w / 2)
        k2 = k1 + self.crash.spread_w
        sigma = vix / 100.0
        debit = _bs_spread(price, k1, k2, self.crash.dte / 365.0, sigma)
        
        t['k1'] = k1
        t['k2'] = k2
        t['debit'] = debit
        t['sigma'] = sigma
        t['size_mult'] = self.crash.risk_mult if self.state.risk_off_active else 1.0
        t['resid_entry'] = price
        t['resid_expiry'] = date.today() + timedelta(days=self.crash.dte)
        
        # Sync to PositionState for report generation
        self.state.crash_entry_date = date.today()
        self.state.crash_entry_price = price
        self.state.crash_k1 = k1
        self.state.crash_k2 = k2
        self.state.crash_debit = debit
        self.state.crash_sigma = sigma
        self.state.crash_etf_entry = spxl_price
        self.state.crash_etf_scaled = False
        self.state.crash_reentry = False
        self.state.crash_opt_reopened = False
        self.state.crash_exit_mode = self.crash.mode
        self.state.crash_half_pct = self.crash.half_pct
        self.state.crash_yin_pct = self.crash.yin_pct
        self.state.crash_yin_scaled = False
        self.state.crash_stop_pct = self.crash.stop_pct
        self.state.crash_reentry_pct = self.crash.reentry_pct
        self.state.crash_dte = self.crash.dte
        self.state.crash_spread_w = self.crash.spread_w
        self.state.crash_drop_thresh = self.crash.drop_thresh
        self.state.crash_etf_stop_pct = self.crash.etf_stop_pct
        self.state.crash_etf_out = False
        self.state.crash_size_mult = self.crash.risk_mult if self.state.risk_off_active else 1.0
        self.state.crash_etf_size = int(self.ETF_SIZE['CRASH'] * (self.crash.risk_mult if self.state.risk_off_active else 1.0))
        self.state.crash_stop_cooldown = self.crash.stop_cooldown
        
        return TradeEvent('CRASH', 'OPEN', date.today(), price, spxl_price, {'n': n})
    
    def _update_crash(self, price: float, spxl_price: float, vix: float, report: dict, risk_gate_active: bool, spxl_ohlc) -> list[TradeEvent]:
        """Update existing crash position — handles all exit types."""
        events = []
        t = self._cur_trade('CRASH')
        if not t:
            return events
        
        alerts = report.get('close_alerts', []) or []
        sm = t.get('size_mult', 1.0)
        osm = sm * self.crash.opt_mult
        cal = (date.today() - t['open']).days
        T_rem = max(self.crash.dte / 365.0 - cal / 365.0, 1 / 365.0)
        
        # Option standalone (tp) handling
        if self.crash.opt_standalone == 'tp' and not t.get('opt_closed') and not t.get('resid') and t.get('k1') and t.get('k2') and t.get('debit') is not None:
            t['resid'] = {'k1': t['k1'], 'k2': t['k2'], 'debit': t['debit'],
                          'sigma': t.get('sigma') or 0.20,
                          'entry': t['resid_entry'], 'expiry': t['resid_expiry'],
                          'indep': True}
            events.append(TradeEvent('CRASH', 'RESIDUAL_SETTLE', date.today(), price, spxl_price, 
                                   {'reason': 'OPTION_TO_INDEPENDENT', 'trade_n': t['n']}))
        
        # ETF出场分支
        if t.get('etf_out'):
            # 已退半/首阴/止损后的ETF剩余部分
            if t.get('half_date') is None and t.get('k1') and t.get('k2') and t.get('debit') is not None and not t.get('opt_closed') and not t.get('resid'):
                close_d = _bs_spread(price, t['k1'], t['k2'], T_rem, t.get('sigma'))
                t['opt_pnl'] = max((close_d - t['debit']), -t['debit']) * 100 * osm
        elif t.get('half_date'):
            # 首阳退半后续持
            base = t.get('re_entry_spxl') or t.get('etf_entry')
            keep = t.get('keep_sh')
            if keep is None:
                sh = t.get('etf_shares') or 0
                keep = sh - _leg_round(sh, self.crash.half_pct)
                t['keep_sh'] = keep
            fill = _etf_stop_fill(base, date.today(), spxl_ohlc) if ('崩盘跌穿' in str(alerts) and base) else spxl_price
            fill = fill if fill is not None else spxl_price
            rem = keep * (fill - base) if base else 0
            t['etf_pnl'] = ((t.get('half_etf') or 0) + rem) * sm
        elif t.get('yin_sh') is not None:
            # 首阴清仓
            yin_pct = t.get('yin_pct') or self.crash.yin_pct
            sh = t.get('etf_shares') or 0
            yin_sh = t.get('yin_sh')
            if yin_sh is None:
                yin_sh = _leg_round(sh, yin_pct)
                t['yin_sh'] = yin_sh
                t['yin_keep_sh'] = sh - yin_sh
            keep = t.get('yin_keep_sh')
            if keep is None:
                keep = sh - yin_sh
            if t.get('re_yin_px'):
                rem = keep * (t['re_yin_px'] - t['etf_entry']) * sm
                full_fill = _etf_stop_fill(t['re_yin_px'], date.today(), spxl_ohlc) if '崩盘跌穿' in str(alerts) else spxl_price
                full_fill = full_fill if full_fill is not None else spxl_price
                full = sh * (full_fill - t['re_yin_px']) * sm
                t['etf_pnl'] = t['yin_etf'] + rem + full
            else:
                final = _etf_stop_fill(t['etf_entry'], date.today(), spxl_ohlc) if '崩盘跌穿' in str(alerts) else spxl_price
                final = final if final is not None else spxl_price
                t['etf_pnl'] = t['yin_etf'] + keep * (final - t['etf_entry']) * sm
            if not t.get('re_yin_px') and t.get('k1') and t.get('k2') and t.get('debit') is not None and not t.get('opt_closed') and not t.get('resid'):
                if self.crash.mode in ('V9', 'V10', 'V11') and '崩盘跌穿' in str(alerts):
                    t['resid'] = {'k1': t['k1'], 'k2': t['k2'], 'debit': t['debit'],
                                  'sigma': t.get('sigma') or 0.20,
                                  'entry': t['resid_entry'], 'expiry': t['resid_expiry']}
                else:
                    close_d = _bs_spread(price, t['k1'], t['k2'], T_rem, t.get('sigma'))
                    t['opt_pnl'] = max((close_d - t['debit']), -t['debit']) * 100 * osm
        else:
            # 纯崩盘持仓（未退半/未首阴）
            if t.get('k1') and t.get('k2') and t.get('debit') is not None and not t.get('opt_closed') and not t.get('resid'):
                if self.crash.mode in ('V9', 'V10', 'V11') and '崩盘跌穿' in str(alerts):
                    t['resid'] = {'k1': t['k1'], 'k2': t['k2'], 'debit': t['debit'],
                                  'sigma': t.get('sigma') or 0.20,
                                  'entry': t['resid_entry'], 'expiry': t['resid_expiry']}
                else:
                    close_d = _bs_spread(price, t['k1'], t['k2'], T_rem, t.get('sigma'))
                    t['opt_pnl'] = max((close_d - t['debit']), -t['debit']) * 100 * osm
            if t.get('etf_entry'):
                fill = _etf_stop_fill(t['etf_entry'], date.today(), spxl_ohlc) if '崩盘跌穿' in str(alerts) else spxl_price
                if fill is None:
                    # 止损限价单当日不成交 → 同价续持
                    self.state.crash_etf_carry = {'shares': t.get('etf_shares') or 0,
                                                  'entry': t['etf_entry'],
                                                  'fresh_entry': spxl_price}
                    t['etf_pnl'] = 0.0
                else:
                    t['etf_pnl'] = (t.get('etf_shares') or 0) * (fill - t['etf_entry']) * sm
        
        # 重开期权 (reopened)
        if t.get('reopened'):
            rcal = (date.today() - t['reopen_date']).days
            rT_rem = max(self.crash.dte / 365.0 - rcal / 365.0, 1 / 365.0)
            rclose = _bs_spread(price, t['reopen_k1'], t['reopen_k2'], rT_rem, t['reopen_sigma'])
            ropnl = max((rclose - t['reopen_debit']), -t['reopen_debit']) * 100 * osm
            t['opt_pnl'] = (t.get('opt_pnl') or 0) + ropnl
        
        # 出场类型判定
        result = None
        if '崩盘首阴' in str(alerts) and '崩盘首阴续持' not in str(alerts):
            result = '首阴清仓'
        elif '崩盘二次首阳' in str(alerts):
            result = '二次首阳清仓'
        elif '崩盘首阳' in str(alerts):
            result = '首阳退半'
        elif '崩盘跌穿' in str(alerts):
            result = '止损-2.5%'
        elif t.get('crash_days', 0) >= 4:
            result = '4天强制平'
        
        if result:
            self._close_trade('CRASH', price, result, spxl_price)
            events.append(TradeEvent('CRASH', 'CLOSE', date.today(), price, spxl_price, 
                                   {'n': t['n'], 'result': result}))
        
        return events
    
    def _settle_residuals(self, price: float, spxl_price: float, report: dict) -> list[TradeEvent]:
        """V9/V10/V11: settle crash options riding past XSP stop-loss."""
        events = []
        for t in self.ledger['CRASH']:
            r = t.get('resid')
            if not r:
                continue
            days_left = (r['expiry'] - date.today()).days
            if self.crash.opt_standalone == 'tp' and r.get('indep'):
                # Handled by evaluate_standalone_option
                continue
            if price <= r['entry'] and days_left > 0:
                continue
            T_rem = max(days_left / 365.0, 1 / 365.0)
            close_d = _bs_spread(price, r['k1'], r['k2'], T_rem, r['sigma'])
            t['opt_pnl'] = max((close_d - r['debit']), -r['debit']) * 100 * t.get('size_mult', 1.0) * self.crash.opt_mult
            reason = '残期收复' if price > r['entry'] else '到期兜底'
            t['resid'] = None
            events.append(TradeEvent('CRASH', 'RESIDUAL_SETTLE', date.today(), price, spxl_price,
                                   {'trade_n': t['n'], 'reason': reason, 'opt_pnl': t['opt_pnl']}))
        return events
    
    # ── Helpers ──────────────────────────────────────────────────────────
    
    def _open_trade(self, kind: str, asof_price: float) -> int:
        n = len(self.ledger[kind]) + 1
        self.ledger[kind].append({
            'n': n, 'kind': kind, 'open': date.today(), 'open_p': asof_price,
            'close': None, 'close_p': None, 'result': None,
            'etf_entry': None, 'etf_close': None, 'etf_shares': None,
            'etf_pnl': 0.0, 'opt_pnl': 0.0, 'segs': [],
            'half_etf': 0.0, 'half_sh': None, 'keep_sh': None, 'half_date': None,
            'yin_etf': 0.0, 'yin_sh': None, 'yin_keep_sh': None,
            'mr_K': None, 'mr_sigma': None, 'mr_expiry': None, 'mr_opt_entry': None,
            'k1': None, 'k2': None, 'debit': None, 'sigma': None,
            'etf_out': False, 'size_mult': 1.0, 'opt_closed': False,
            'resid_entry': None, 'resid_expiry': None, 'resid': None,
        })
        self._open_ref[kind] = n
        return n
    
    def _close_trade(self, kind: str, price: float, result: str, etf_p: float):
        t = self._cur_trade(kind)
        if t:
            t['close'] = date.today()
            t['close_p'] = price
            t['result'] = result
            t['etf_close'] = etf_p
            t['hold_days'] = max((date.today() - t['open']).days, 0)
        self._open_ref[kind] = None
    
    def _cur_trade(self, kind: str):
        if self._open_ref[kind] is None:
            return None
        for t in self.ledger[kind]:
            if t['n'] == self._open_ref[kind]:
                return t
        return None