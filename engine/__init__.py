"""Engine package: shared position logic for production and backtest."""

from engine.policies import (
    CrashPolicy,
    TrendPolicy,
    MRPolicy,
    ProductionCrashPolicy,
    ProductionTrendPolicy,
    ProductionMRPolicy,
)
from engine.pricing import (
    _bs_spread,
    _bs_call,
    _etf_stop_fill,
    _leg_round,
    compute_carry_blend,
    evaluate_standalone_option,
)
from engine.position import PositionEngine, PositionState, TradeEvent
from engine.position_state import PositionState
from engine.position_config import PositionConfig
from engine.position_core import (
    process_position_report,
    ReportInputs,
    TradeEvent,
)

__all__ = [
    # Policies
    'CrashPolicy',
    'TrendPolicy', 
    'MRPolicy',
    'ProductionCrashPolicy',
    'ProductionTrendPolicy',
    'ProductionMRPolicy',
    # Pricing
    '_bs_spread',
    '_bs_call',
    '_etf_stop_fill',
    '_leg_round',
    'compute_carry_blend',
    'evaluate_standalone_option',
    # Core
    'PositionEngine',
    'PositionState',
    'PositionConfig',
    'process_position_report',
    'ReportInputs',
    'TradeEvent',
]