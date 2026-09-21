"""get_xsp_anchor_price 降级链 — 单元测试 (2026-09-21: fast_info 偶发 KeyError 'exchangeTimezoneName')."""
import math
import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import app


def _ticker(fast_info_val=None, fast_info_exc=None, hist_close=None, hist_exc=None):
    tk = MagicMock()
    fi = MagicMock()
    if fast_info_exc is not None:
        fi.__getitem__.side_effect = fast_info_exc
    else:
        fi.__getitem__.return_value = fast_info_val
    tk.fast_info = fi
    if hist_exc is not None:
        tk.history.side_effect = hist_exc
    else:
        tk.history.return_value = pd.DataFrame({'Close': [hist_close]})
    return tk


def _run(tk):
    with patch.object(app, 'yf') as mock_yf:
        mock_yf.Ticker.return_value = tk
        return app.get_xsp_anchor_price()


class TestAnchorPrice:
    def test_fast_info_ok(self):
        assert _run(_ticker(fast_info_val=755.18, hist_close=750.0)) == 755.18

    def test_fast_info_keyerror_falls_back_to_history(self):
        # 日志里的 'exchangeTimezoneName' 案: 不应回 0, 应拿日线收盘顶住
        assert _run(_ticker(fast_info_exc=KeyError('exchangeTimezoneName'),
                            hist_close=750.0)) == 750.0

    def test_fast_info_nan_falls_back(self):
        assert _run(_ticker(fast_info_val=float('nan'), hist_close=750.0)) == 750.0

    def test_fast_info_zero_falls_back(self):
        assert _run(_ticker(fast_info_val=0, hist_close=750.0)) == 750.0

    def test_all_fail_returns_zero(self, capsys):
        assert _run(_ticker(fast_info_exc=KeyError('x'),
                            hist_exc=Exception('net down'))) == 0
        assert 'stale' in capsys.readouterr().out
