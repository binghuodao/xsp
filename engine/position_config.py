"""PositionConfig: all strategy parameters as a single dataclass.

Maps CLI argparse args to structured config. Single source for both
production and backtest configuration.
"""

from dataclasses import dataclass
from typing import Optional
import argparse


@dataclass
class PositionConfig:
    """All strategy parameters in one place."""
    
    # Core modes
    crash_mode: str = 'V10'
    crash_exit_mode: str = 'V10'
    layer_priority: str = 'crash_mr_trend'
    
    # Crash entry/exit
    crash_half_pct: float = 0.0625
    crash_yin_pct: float = 0.75
    stop_pct: float = 0.025
    drop_thresh: float = 0.005
    reentry_pct: float = 1.0
    dte: int = 21
    spread_w: int = 15
    etf_stop_pct: float = 0.0
    stop_cooldown: int = 0
    crash_etf_size: int = 5000
    
    # Risk gates
    risk_gate: str = 'b200slope'
    risk_mult: float = 1.0
    y10_gate: float = 0.0
    
    # Option standalone
    opt_mult: float = 1.0
    opt_standalone: str = 'off'
    opt_tp: float = 1.70
    opt_sl_resid: float = 0.0
    opt_sl_decay: float = 0.0
    opt_sl_adaptive: bool = False
    
    # Bounce filter (intraday)
    bounce_thresh: Optional[float] = 1.003
    
    # Trend
    trend_etf_size: int = 5000
    trend_dte: int = 14
    trend_stop_pct: float = 0.05
    trend_trail_pct: float = 0.03
    trend_force_days: int = 30
    
    # MR
    mr_etf_size: int = 2000
    mr_dte: int = 7
    mr_stop_pct: float = 0.02
    mr_green_pct: float = 0.003
    mr_force_days: int = 3
    mr_rsi_thresh: int = 30
    mr_vix_thresh: float = 20.0
    
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'PositionConfig':
        """Map CLI argparse.Namespace → PositionConfig."""
        return cls(
            crash_mode=args.crash_mode,
            crash_exit_mode=args.crash_mode,
            layer_priority=args.layer_priority,
            crash_half_pct=args.crash_half,
            crash_yin_pct=args.crash_yin,
            stop_pct=args.stop_pct,
            drop_thresh=args.drop_thresh,
            reentry_pct=args.reentry_pct,
            dte=args.dte,
            spread_w=args.spread_w,
            etf_stop_pct=args.etf_stop,
            stop_cooldown=args.stop_cooldown,
            risk_gate=args.risk_gate,
            risk_mult=args.risk_mult,
            y10_gate=args.crash_y10_gate,
            opt_mult=args.opt_mult,
            opt_standalone=args.opt_standalone,
            opt_tp=args.opt_tp,
            opt_sl_resid=args.opt_sl_resid,
            opt_sl_decay=args.opt_sl_decay,
            opt_sl_adaptive=args.opt_sl_adaptive,
            bounce_thresh=args.crash_bounce_thresh,
            trend_etf_size=5000,
            mr_etf_size=2000,
            crash_etf_size=5000,
        )
    
    def to_production_singletons(self):
        """Create ProductionCrashPolicy, ProductionTrendPolicy, ProductionMRPolicy for backward compat."""
        from engine.policies import CrashPolicy, TrendPolicy, MRPolicy
        return (
            CrashPolicy(
                mode=self.crash_mode,
                layer_priority=self.layer_priority,
                bounce_thresh=self.bounce_thresh,
                half_pct=self.crash_half_pct,
                yin_pct=self.crash_yin_pct,
                stop_pct=self.stop_pct,
                drop_thresh=self.drop_thresh,
                reentry_pct=self.reentry_pct,
                dte=self.dte,
                spread_w=self.spread_w,
                etf_stop_pct=self.etf_stop_pct,
                stop_cooldown=self.stop_cooldown,
                risk_gate=self.risk_gate,
                risk_mult=self.risk_mult,
                y10_gate=self.y10_gate,
                opt_mult=self.opt_mult,
                opt_standalone=self.opt_standalone,
                opt_tp=self.opt_tp,
                opt_sl_resid=self.opt_sl_resid,
                opt_sl_decay=self.opt_sl_decay,
                opt_sl_adaptive=self.opt_sl_adaptive,
            ),
            TrendPolicy(
                etf_size=self.trend_etf_size,
                dte=self.trend_dte,
                stop_pct=self.trend_stop_pct,
                trail_pct=0.03,
                force_days=self.trend_force_days,
            ),
            MRPolicy(
                etf_size=self.mr_etf_size,
                dte=self.mr_dte,
                stop_pct=self.mr_stop_pct,
                green_pct=self.mr_green_pct,
                force_days=self.mr_force_days,
                rsi_thresh=self.mr_rsi_thresh,
                vix_thresh=self.mr_vix_thresh,
            )
        )