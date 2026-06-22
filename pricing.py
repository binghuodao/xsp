"""Black-Scholes pricing for XSP Christmas Tree Monitor"""
import math

RISK_FREE_RATE = 0.05


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = max(S - K, 0) if opt_type == 'C' else max(K - S, 0)
        return intrinsic
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt_type == 'C':
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def xmas_theory_price(S, k_short, k_mid, k_long, T, r, iv_short, iv_mid, iv_long, opt_type):
    """Fair value of entering +1S/-3M/+2L position.
    Positive = net credit (you receive money to enter)."""
    bs_short = black_scholes(S, k_short, T, r, iv_short, opt_type)
    bs_mid = black_scholes(S, k_mid, T, r, iv_mid, opt_type)
    bs_long = black_scholes(S, k_long, T, r, iv_long, opt_type)
    return bs_short + 2 * bs_long - 3 * bs_mid


def xmas_payoff_extrema(combo_bid, strikes, opt_type, width, atr_val=10.0):
    """Compute max profit, max loss, and breakevens at expiration
    for +1S/-3M/+2L entered at combo_bid (credit received)."""
    k_short, k_mid, k_long = strikes
    lower = min(k_short, k_long)
    upper = max(k_short, k_long)
    mid = k_mid

    scan_range = max(atr_val * 3, width * 3)
    min_px = lower - scan_range
    max_px = upper + scan_range
    step = 0.1

    max_profit = -float('inf')
    max_loss = float('inf')
    be_points = []
    prev_pnl = None

    px = min_px
    while px <= max_px:
        if opt_type == 'P':
            intr_short = max(k_short - px, 0) if k_short > px else 0.0
            intr_mid = max(k_mid - px, 0) if k_mid > px else 0.0
            intr_long = max(k_long - px, 0) if k_long > px else 0.0
        else:
            intr_short = max(px - k_short, 0) if px > k_short else 0.0
            intr_mid = max(px - k_mid, 0) if px > k_mid else 0.0
            intr_long = max(px - k_long, 0) if px > k_long else 0.0

        pnl = -combo_bid + intr_short - 3 * intr_mid + 2 * intr_long

        if pnl > max_profit:
            max_profit = pnl
        if pnl < max_loss:
            max_loss = pnl

        if prev_pnl is not None:
            crossed_up = prev_pnl <= 0 and pnl >= 0
            crossed_down = prev_pnl >= 0 and pnl <= 0
            if crossed_up or crossed_down:
                if abs(pnl - prev_pnl) > 1e-10:
                    be = (px - step) + step * (0 - prev_pnl) / (pnl - prev_pnl)
                    be_points.append(round(be, 2))

        prev_pnl = pnl
        px += step

    be_lower = min(be_points) if be_points else None
    be_upper = max(be_points) if be_points else None
    if be_lower == be_upper:
        be_upper = None

    risk_reward = max_profit / abs(max_loss) if max_loss < 0 else None

    return {
        'max_profit': round(max_profit, 2),
        'max_loss': round(max_loss, 2),
        'be_lower': be_lower,
        'be_upper': be_upper,
        'risk_reward': round(risk_reward, 2) if risk_reward is not None else None,
    }


def xmas_scenarios(S, k_short, k_mid, k_long, T, r, iv_short, iv_mid, iv_long, opt_type, combo_mid):
    """Scenario analysis returning {key: {price, change}}."""
    base_theory = xmas_theory_price(
        S, k_short, k_mid, k_long, T, r,
        iv_short, iv_mid, iv_long, opt_type
    )

    out = {
        'current': {
            'price': round(base_theory, 2),
            'change': 0.0,
            'mid': round(combo_mid, 2),
            'edge': round(combo_mid - base_theory, 2),
        }
    }

    for pct in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
        s2 = S * (1 + pct / 100.0)
        theory = xmas_theory_price(
            s2, k_short, k_mid, k_long, T, r,
            iv_short, iv_mid, iv_long, opt_type
        )
        out[f'price_{pct:+.1f}pct'] = {
            'price': round(theory, 2),
            'change': round(theory - combo_mid, 2)
        }

    for d_days in [-5, -3, -1]:
        t2 = max(T + d_days / 365.0, 0.001)
        theory = xmas_theory_price(
            S, k_short, k_mid, k_long, t2, r,
            iv_short, iv_mid, iv_long, opt_type
        )
        out[f'dte_{d_days}d'] = {
            'price': round(theory, 2),
            'change': round(theory - combo_mid, 2)
        }

    for d_iv in [-5, -2, -1, 1, 2, 5]:
        iv_s = max(0.05, iv_short + d_iv / 100.0)
        iv_m = max(0.05, iv_mid + d_iv / 100.0)
        iv_l = max(0.05, iv_long + d_iv / 100.0)
        theory = xmas_theory_price(
            S, k_short, k_mid, k_long, T, r,
            iv_s, iv_m, iv_l, opt_type
        )
        out[f'iv_{d_iv:+.0f}pct'] = {
            'price': round(theory, 2),
            'change': round(theory - combo_mid, 2)
        }

    return out
