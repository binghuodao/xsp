"""Policy dataclasses for PositionEngine configuration.

All user-tunable parameters live here. Production uses singletons with
CLI-default values; experiments create modified instances for grid sweeps.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CrashPolicy:
    """Collapse/崩盘策略参数."""
    
    # 核心模式
    mode: str = 'V10'                    # V9 | V10 | V11 | V1 | V4 | V5 | V6 | V7 | V8 | V0
    layer_priority: str = 'crash_mr_trend'  # crash_mr_trend | mr_crash_trend
    
    # 入场过滤
    bounce_thresh: Optional[float] = 1.003  # SPY close/low intraday bounce; None = disabled
    drop_thresh: float = 0.005             # XSP single-day drop threshold (0.5%)
    
    # 仓位管理
    half_pct: float = 0.0625               # 首阳退半比例
    yin_pct: float = 0.75                  # 首阴清仓比例
    reentry_pct: float = 1.0               # 再进触发线 = entry * reentry_pct
    
    # 止损/出场
    stop_pct: float = 0.025                # XSP 止损 -2.5%
    dte: int = 21                          # 期权 DTE
    spread_w: int = 15                     # 价差宽度
    etf_stop_pct: float = 0.0              # SPXL 独立止损 (0=off)
    stop_cooldown: int = 0                 # 止损后冷却天数
    
    # 规模/风控
    risk_gate: str = 'b200slope'           # none | b200 | b200slope | vix80 | macd | engulf | s3red
    risk_mult: float = 1.0                 # 门亮时仓位倍数
    y10_gate: float = 0.0                  # 10Y收益率20d变化阈值(pp); 0=off
    opt_mult: float = 1.0                  # 期权腿PnL倍数
    
    # 独立期权管理 (实验性)
    opt_standalone: str = 'off'            # off | tp | tp-v9
    opt_tp: float = 1.70                   # 止盈线 = debit * opt_tp
    opt_sl_resid: float = 0.0              # 残值止损线 = debit * opt_sl_resid
    opt_sl_decay: float = 0.0              # 时间衰减早退: hold_days > DTE*this AND value < debit*this
    opt_sl_adaptive: bool = False          # 止损需 XSP <= entry 确认
    
    def resolve_residuals(self, state, price: float, asof) -> list[dict]:
        """Mode-specific residual settlement. Returns list of events.
        
        This is called by PositionEngine after main crash update.
        Override in subclasses for custom residual logic.
        """
        from engine.pricing import evaluate_standalone_option
        
        events = []
        
        if self.opt_standalone in ('tp', 'tp-v9'):
            events.extend(evaluate_standalone_option(state, price, asof, self))
        
        # V9/V10/V11 shared-exit residuals
        if self.mode in ('V9', 'V10', 'V11'):
            # Delegate to engine's _settle_residuals which has full logic
            pass  # Engine calls _settle_residuals directly
        
        return events
    
    def should_block_crash_signal(self, report: dict, risk_gate_active: bool, bounce_ok: bool) -> bool:
        """Determine if crash signal should be blocked."""
        if not bounce_ok:
            return True
        if risk_gate_active:
            return True
        if report.get('direction') != 'CALL':
            return True
        return False


@dataclass
class TrendPolicy:
    """趋势/Trend策略参数."""
    
    etf_size: int = 5000
    dte: int = 14
    stop_pct: float = 0.05                 # 5% (1% if near BB)
    trail_pct: float = 0.03                # 3% trailing
    force_days: int = 30                   # T+30 强制平仓
    near_bb_stop_pct: float = 0.01         # 1% if near BB
    bb_proximity_mult: float = 0.10        # near BB if (BBU - price) < ATR*0.6


@dataclass
class MRPolicy:
    """均值回归/MR策略参数."""
    
    etf_size: int = 2000
    dte: int = 7
    stop_pct: float = 0.02                 # -2%
    green_pct: float = 0.003               # +0.3% 首阳
    force_days: int = 3                    # 3天强制平
    rsi_thresh: int = 30                   # RSI < 30
    vix_thresh: float = 20.0               # VIX > 20


# ── Production Singletons ──
# Match current CLI defaults exactly

ProductionCrashPolicy = CrashPolicy(
    mode='V10',
    layer_priority='crash_mr_trend',
    bounce_thresh=1.003,
    drop_thresh=0.005,
    half_pct=0.0625,
    yin_pct=0.75,
    reentry_pct=1.0,
    stop_pct=0.025,
    dte=21,
    spread_w=15,
    etf_stop_pct=0.0,
    stop_cooldown=0,
    risk_gate='b200slope',
    risk_mult=1.0,
    y10_gate=0.0,
    opt_mult=1.0,
    opt_standalone='off',
    opt_tp=1.70,
    opt_sl_resid=0.0,
    opt_sl_decay=0.0,
    opt_sl_adaptive=False,
)

ProductionTrendPolicy = TrendPolicy(
    etf_size=5000,
    dte=14,
    stop_pct=0.05,
    trail_pct=0.03,
    force_days=30,
    near_bb_stop_pct=0.01,
    bb_proximity_mult=0.10,
)

ProductionMRPolicy = MRPolicy(
    etf_size=2000,
    dte=7,
    stop_pct=0.02,
    green_pct=0.003,
    force_days=3,
    rsi_thresh=30,
    vix_thresh=20.0,
)