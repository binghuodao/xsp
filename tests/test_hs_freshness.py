"""hs 新鲜度行 — 单元测试 (2026-09-21: 0917 晨报 VIX 静默 stale + 鲜价混算出幻影方向)."""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.report import _freshness_line, build_full_report


def _hs(**kw):
    hs = {'vix_date': None, 'spy_date': None, 'xsp_date': None, 'tnx_date': None}
    hs.update(kw)
    return hs


class TestFreshnessLine:
    def test_all_fresh_no_warning(self):
        line = _freshness_line(_hs(vix_date='2026-09-16', spy_date='2026-09-16',
                                   xsp_date='2026-09-16', tnx_date='2026-09-16'))
        assert 'VIX 09/16' in line and 'SPY 09/16' in line
        assert 'XSP 09/16' in line and '10Y 09/16' in line
        assert '⚠️' not in line

    def test_stale_vix_flagged(self):
        # 0917 晨报复现: VIX 老 5 天, 其余是鲜的
        line = _freshness_line(_hs(vix_date='2026-09-11', spy_date='2026-09-16',
                                   xsp_date='2026-09-16', tnx_date='2026-09-16'))
        assert '⚠️' in line and 'VIX滞后5天' in line

    def test_one_day_skew_no_warning(self):
        # yf 各源到货时差 1 天属正常, 不告警 (避免盘中天天误报)
        line = _freshness_line(_hs(vix_date='2026-09-15', spy_date='2026-09-16',
                                   xsp_date='2026-09-16', tnx_date='2026-09-16'))
        assert '⚠️' not in line

    def test_all_none_no_crash(self):
        # 重启后未刷新: 显示 --, 不告警不崩
        line = _freshness_line(_hs())
        assert '--' in line and '⚠️' not in line

    def test_none_excluded_from_compare(self):
        # 闸门关闭时无 10Y: 不参与比较, 其余一致不告警
        line = _freshness_line(_hs(vix_date='2026-09-16', spy_date='2026-09-16',
                                   xsp_date='2026-09-16', tnx_date=None))
        assert '10Y --' in line and '⚠️' not in line

    def test_bad_format_no_crash(self):
        line = _freshness_line(_hs(vix_date='not-a-date', spy_date='2026-09-16',
                                   xsp_date='2026-09-16', tnx_date='2026-09-16'))
        assert 'VIX --' in line and '⚠️' not in line


class TestFullReportFreshness:
    def test_freshness_in_full_report(self):
        hs = _hs(vix_date='2026-09-16', spy_date='2026-09-16',
                 xsp_date='2026-09-16', tnx_date='2026-09-16')
        hs.update({'adx': 12.0, 'er': 0.2, 'bbw': 2.5, 'dev': -1.4, 'vr': 1.5,
                   'vix': 17.0, 'vix_rank': 50.0, 'di_diff': -0.1,
                   'ema_20': 760.0, 'support': 753.0, 'resistance': 772.0,
                   'atr_14': 6.5})
        lines = build_full_report('t', 755.0, hs, None, 'BB 中段', 20, '🔴', 'Ranging', 'now')
        fresh = [l for l in lines if l.startswith('🕐 数据')]
        assert len(fresh) == 1 and '⚠️' not in fresh[0]
