"""
IQLBL Backend — IQL + Dynamic Black-Litterman model wrapper for dashboard integration.
Wraps two_stage_DYBL_portfolio.py classes for API use.
"""

import os
import sys
import time
import threading
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Add IQLBL folder to path
IQLBL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IQLBL_v1_Dashboard")
sys.path.insert(0, IQLBL_DIR)

from two_stage_DYBL_portfolio import (
    load_config, get_default_config, setup_seed, setup_device,
    load_all_coin_data, Stage1IQLAgent, DynamicBlackLittermanOptimizer,
    PositionTracker, PortfolioEnvironment, evaluate_portfolio,
    evaluate_baselines, create_stage1_dataset, train_stage1_agent,
)
from backtest import compute_metrics

# ── IQLBL constants ──
IQLBL_COINS = ['BTC', 'DOGE', 'ETH', 'NEAR', 'SOL', 'WLD']

# ── Cache ──
_iqlbl_cache = {}
_iqlbl_lock = threading.Lock()

# ── Loaded model state ──
_iqlbl_agents = None
_iqlbl_config = None
_iqlbl_env = None
_iqlbl_bl_optimizer = None
_iqlbl_loaded = False


def _load_iqlbl_config():
    """Load IQLBL config from bl_portfolio_config.yaml."""
    config_path = os.path.join(IQLBL_DIR, "bl_portfolio_config.yaml")
    if os.path.exists(config_path):
        config = load_config(config_path)
    else:
        config = get_default_config()

    # Set defaults
    if 'portfolio' not in config:
        config['portfolio'] = {}
    config['portfolio'].setdefault('initial_value', 10000)
    config['portfolio'].setdefault('fee_rate', 0.001)
    config['portfolio'].setdefault('allow_short', True)
    config['portfolio'].setdefault('max_leverage', 1.0)

    if 'black_litterman' not in config:
        config['black_litterman'] = {}
    config['black_litterman'].setdefault('tau', 0.05)
    config['black_litterman'].setdefault('risk_aversion', 2.5)
    config['black_litterman'].setdefault('view_confidence', 0.6)
    config['black_litterman'].setdefault('lookback', 60)

    if 'dynamic_bl' not in config:
        config['dynamic_bl'] = {}
    config['dynamic_bl'].setdefault('vol_lookback', 20)
    config['dynamic_bl'].setdefault('vol_threshold_high', 0.04)
    config['dynamic_bl'].setdefault('vol_threshold_low', 0.015)
    config['dynamic_bl'].setdefault('confidence_min', 0.2)
    config['dynamic_bl'].setdefault('confidence_max', 0.8)
    config['dynamic_bl'].setdefault('trend_lookback', 10)
    config['dynamic_bl'].setdefault('trend_threshold', 0.05)
    config['dynamic_bl'].setdefault('regime_weight', 0.3)
    config['dynamic_bl'].setdefault('momentum_decay', 0.5)

    return config


def _ensure_iqlbl_loaded():
    """Ensure IQLBL model is trained and ready."""
    global _iqlbl_agents, _iqlbl_config, _iqlbl_env, _iqlbl_bl_optimizer, _iqlbl_loaded

    if _iqlbl_loaded:
        return True

    with _iqlbl_lock:
        if _iqlbl_loaded:
            return True

        try:
            config = _load_iqlbl_config()
            _iqlbl_config = config

            setup_seed(config.get('seed', 42))
            setup_device(config.get('gpu', {}).get('use_cuda', True))

            # Load data
            all_data = load_all_coin_data(IQLBL_DIR, IQLBL_COINS)
            if len(all_data) < 2:
                print("IQLBL: Not enough coin data found")
                return False

            # Create environment
            env = PortfolioEnvironment(all_data, IQLBL_COINS, config)
            _iqlbl_env = env

            # Find date indices
            train_end = pd.to_datetime(config['data']['train_end_date'])
            test_start = pd.to_datetime(config['data']['test_start_date'])

            train_end_idx = None
            test_start_idx = None
            for i, date in enumerate(env.dates):
                if date <= train_end:
                    train_end_idx = i
                if date >= test_start and test_start_idx is None:
                    test_start_idx = i

            # Train Stage 1 agents
            window_size = config['environment']['window_size']
            stage1_agents = {}
            for coin in IQLBL_COINS:
                if coin not in all_data:
                    continue
                setup_seed(config.get('seed', 42))
                agent = Stage1IQLAgent(state_dim=window_size, action_dim=1, config=config.get('iql', {}))
                dataset = create_stage1_dataset(all_data[coin], config, train_end_idx)
                if len(dataset['states']) > 0:
                    agent = train_stage1_agent(agent, dataset, config)
                stage1_agents[coin] = agent

            _iqlbl_agents = stage1_agents

            # Initialize BL optimizer
            bl_optimizer = DynamicBlackLittermanOptimizer(n_assets=len(IQLBL_COINS), config=config)
            _iqlbl_bl_optimizer = bl_optimizer

            _iqlbl_loaded = True
            print("IQLBL: Model loaded and trained successfully")
            return True

        except Exception as e:
            print(f"IQLBL: Load failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def iqlbl_run_portfolio():
    """Run IQLBL portfolio evaluation on test period."""
    from app import _is_cached, _set_cache

    ok, cached = _is_cached("iqlbl_portfolio")
    if ok:
        return cached

    if not _ensure_iqlbl_loaded():
        return {"error": "IQLBL model not available. Check IQLBL_v1_Dashboard folder."}

    config = _iqlbl_config
    env = _iqlbl_env

    # Find test start index
    test_start = pd.to_datetime(config['data']['test_start_date'])
    test_start_idx = None
    for i, date in enumerate(env.dates):
        if date >= test_start and test_start_idx is None:
            test_start_idx = i

    if test_start_idx is None:
        return {"error": "Could not find test start date in data"}

    # Reset BL optimizer state for fresh evaluation
    bl_optimizer = DynamicBlackLittermanOptimizer(n_assets=len(IQLBL_COINS), config=config)

    # Evaluate portfolio
    results = evaluate_portfolio(env, _iqlbl_agents, bl_optimizer, config,
                                  test_start_idx, len(env.dates))

    # Evaluate baselines
    baselines = evaluate_baselines(env, config, test_start_idx, len(env.dates))

    # Format results for API
    dates = results['dates']
    portfolio_values = results['portfolio_values']
    initial_value = config['portfolio'].get('initial_value', 10000)

    # Normalize portfolio values (like SDELP does)
    norm_values = [v / initial_value for v in portfolio_values]
    norm_bah = [v / initial_value for v in baselines['equal_weight']]

    min_len = min(len(norm_values), len(norm_bah), len(dates))
    date_labels = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d) for d in dates[:min_len]]

    # Compute metrics
    iqlbl_metrics = compute_metrics(norm_values[:min_len])
    bah_metrics = compute_metrics(norm_bah[:min_len])

    # Weight history
    weights = results['weights']
    step = max(1, len(weights) // 200)
    weight_dates = date_labels[::step]
    weights_sampled = [w.tolist() if hasattr(w, 'tolist') else list(w) for w in weights[::step]]

    # Daily returns
    vals = np.array(norm_values[:min_len])
    daily_returns = (vals[1:] / vals[:-1] - 1).tolist() if len(vals) > 1 else []

    # Current weights
    current_weights = weights[-1].tolist() if hasattr(weights[-1], 'tolist') else list(weights[-1])

    # IQLBL-specific data
    regimes = results.get('regime', [])
    confidences = results.get('view_confidence', [])
    drawdowns = results.get('drawdown', [])
    defensive_modes = results.get('defensive_mode', [])

    result = {
        "dates": date_labels,
        "sdelp_values": [float(v) for v in norm_values[:min_len]],
        "bah_values": [float(v) for v in norm_bah[:min_len]],
        "sdelp_metrics": {k: round(float(v), 4) for k, v in iqlbl_metrics.items()},
        "bah_metrics": {k: round(float(v), 4) for k, v in bah_metrics.items()},
        "current_weights": current_weights,
        "weight_labels": IQLBL_COINS,
        "weight_dates": weight_dates,
        "weights_sampled": weights_sampled,
        "daily_returns": daily_returns,
        "daily_return_dates": date_labels[1:],
        "tickers": IQLBL_COINS,
        "test_start": config['data'].get('test_start_date', ''),
        "test_end": dates[-1].strftime("%Y-%m-%d") if hasattr(dates[-1], 'strftime') else str(dates[-1]),
        # IQLBL-specific
        "regimes": regimes[:min_len] if regimes else [],
        "confidences": [float(c) for c in confidences[:min_len]] if confidences else [],
        "drawdowns": [float(d) for d in drawdowns[:min_len]] if drawdowns else [],
        "defensive_modes": [bool(d) for d in defensive_modes[:min_len]] if defensive_modes else [],
    }

    _set_cache("iqlbl_portfolio", result)
    return result


def iqlbl_run_train_portfolio():
    """Run IQLBL model on training period (for train chart)."""
    from app import _is_cached, _set_cache

    ok, cached = _is_cached("iqlbl_train_portfolio")
    if ok:
        return cached

    if not _ensure_iqlbl_loaded():
        return {"error": "IQLBL model not available"}

    config = _iqlbl_config
    env = _iqlbl_env

    # Find train end index
    train_end = pd.to_datetime(config['data']['train_end_date'])
    window_size = config['environment']['window_size']

    train_end_idx = None
    for i, date in enumerate(env.dates):
        if date <= train_end:
            train_end_idx = i

    if train_end_idx is None:
        return {"error": "Could not find train end date"}

    # Start from after warmup
    start_idx = window_size + 20

    # Reset BL optimizer
    bl_optimizer = DynamicBlackLittermanOptimizer(n_assets=len(IQLBL_COINS), config=config)

    results = evaluate_portfolio(env, _iqlbl_agents, bl_optimizer, config,
                                  start_idx, train_end_idx)
    baselines = evaluate_baselines(env, config, start_idx, train_end_idx)

    dates = results['dates']
    portfolio_values = results['portfolio_values']
    initial_value = config['portfolio'].get('initial_value', 10000)

    norm_values = [v / initial_value for v in portfolio_values]
    norm_bah = [v / initial_value for v in baselines['equal_weight']]
    min_len = min(len(norm_values), len(norm_bah), len(dates))
    date_labels = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d) for d in dates[:min_len]]

    iqlbl_metrics = compute_metrics(norm_values[:min_len])
    bah_metrics = compute_metrics(norm_bah[:min_len])

    result = {
        "dates": date_labels,
        "sdelp_values": [float(v) for v in norm_values[:min_len]],
        "bah_values": [float(v) for v in norm_bah[:min_len]],
        "sdelp_metrics": {k: round(float(v), 4) for k, v in iqlbl_metrics.items()},
        "bah_metrics": {k: round(float(v), 4) for k, v in bah_metrics.items()},
        "train_start": dates[0].strftime("%Y-%m-%d") if dates and hasattr(dates[0], 'strftime') else "",
        "train_end": config['data']['train_end_date'],
    }

    _set_cache("iqlbl_train_portfolio", result)
    return result


def _fetch_binance_5min(symbol, start_ms=None):
    """Fetch 5-min klines from Binance public API.
    If start_ms is None, defaults to this week's Monday KST 00:00.
    Automatically paginates for ranges exceeding 1000 candles."""
    import requests as req
    from datetime import datetime, timezone, timedelta
    from app import _get_week_start_ms

    KST = timezone(timedelta(hours=9))
    end_ms = int(time.time() * 1000)
    if start_ms is None:
        start_ms = _get_week_start_ms()

    all_data = []
    cursor = start_ms
    while cursor < end_ms:
        url = (f"https://api.binance.com/api/v3/klines"
               f"?symbol={symbol}&interval=5m&startTime={cursor}&endTime={end_ms}&limit=1000")
        resp = req.get(url, timeout=10)
        batch = resp.json()
        if not isinstance(batch, list) or len(batch) == 0:
            break
        all_data.extend(batch)
        cursor = batch[-1][0] + 1
        if len(batch) < 1000:
            break

    if len(all_data) == 0:
        return None
    times = [datetime.fromtimestamp(k[0] / 1000, tz=KST) for k in all_data]
    closes = [float(k[4]) for k in all_data]
    return pd.Series(closes, index=times)


# Map IQLBL coins to Binance symbols
_IQLBL_BINANCE_MAP = {
    'BTC': 'BTCUSDT', 'DOGE': 'DOGEUSDT', 'ETH': 'ETHUSDT',
    'NEAR': 'NEARUSDT', 'SOL': 'SOLUSDT', 'WLD': 'WLDUSDT',
}


def iqlbl_run_live_5min():
    """Fetch 5-min candle data from Binance and apply IQLBL model weights."""
    from app import _is_cached, _set_cache

    ok, cached = _is_cached("iqlbl_live_5min")
    if ok:
        return cached

    frames = {}
    for coin in IQLBL_COINS:
        symbol = _IQLBL_BINANCE_MAP.get(coin)
        if not symbol:
            continue
        try:
            series = _fetch_binance_5min(symbol)
            if series is not None and len(series) > 0:
                frames[coin] = series
        except Exception:
            pass

    if len(frames) < 2:
        return {"error": "Not enough 5-min data from Binance for IQLBL"}

    prices = pd.DataFrame(frames).dropna()
    if len(prices) < 2:
        return {"error": "Not enough aligned data"}

    timestamps = [t.strftime("%m/%d %H:%M") for t in prices.index]
    base = prices.iloc[0]
    normed = prices.div(base)

    # B&H: equal weight
    bah = normed.mean(axis=1)
    bah_pct = ((bah / bah.iloc[0]) - 1) * 100

    # IQLBL: use model weights from portfolio evaluation
    portfolio_data = iqlbl_run_portfolio()
    if "error" not in portfolio_data and "current_weights" in portfolio_data:
        weights = portfolio_data["current_weights"]
        coins_list = portfolio_data.get("tickers", IQLBL_COINS)

        w = []
        for col in prices.columns:
            if col in coins_list:
                idx = coins_list.index(col)
                w.append(abs(weights[idx]))  # Use absolute weight for return calc
            else:
                w.append(1.0 / len(prices.columns))
        w = np.array(w)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum

        iqlbl = (normed * w).sum(axis=1)
        iqlbl_pct = ((iqlbl / iqlbl.iloc[0]) - 1) * 100
    else:
        iqlbl_pct = bah_pct

    result = {
        "dates": timestamps,
        "sdelp_values": [float(v) for v in iqlbl_pct.values],
        "bah_values": [float(v) for v in bah_pct.values],
    }

    _set_cache("iqlbl_live_5min", result)
    return result


def iqlbl_get_model_info():
    """Return IQLBL model hyperparameters."""
    config = _load_iqlbl_config()
    iql = config.get('iql', {})
    bl = config.get('black_litterman', {})
    dyn = config.get('dynamic_bl', {})
    portfolio = config.get('portfolio', {})

    return {
        "model_name": "IQL + Dynamic Black-Litterman",
        "coins": IQLBL_COINS,
        "num_assets": len(IQLBL_COINS),
        "window_size": config.get('environment', {}).get('window_size', 120),
        "iql_gamma": iql.get('gamma', 0.99),
        "iql_tau": iql.get('tau', 0.005),
        "iql_expectile": iql.get('expectile', 0.8),
        "iql_temperature": iql.get('temperature', 3.0),
        "iql_lr": iql.get('lr', 0.0003),
        "n_epochs": config.get('training', {}).get('n_epochs', 100),
        "batch_size": config.get('training', {}).get('batch_size', 256),
        "learning_window": config.get('training', {}).get('learning_window', 5),
        "bl_tau": bl.get('tau', 0.05),
        "risk_aversion": bl.get('risk_aversion', 2.5),
        "view_confidence": bl.get('view_confidence', 0.6),
        "bl_lookback": bl.get('lookback', 60),
        "allow_short": portfolio.get('allow_short', True),
        "fee_rate": portfolio.get('fee_rate', 0.001),
        "initial_value": portfolio.get('initial_value', 10000),
        "train_period": f"~ {config['data']['train_end_date']}",
        "test_period": f"{config['data']['test_start_date']} ~",
        "vol_lookback": dyn.get('vol_lookback', 20),
        "drawdown_threshold": dyn.get('drawdown_threshold', 0.10),
        "data_source": "Binance Futures (XLSX)",
    }


def iqlbl_get_training_history():
    """IQLBL doesn't have episode-based training history like DDPG.
    Return a placeholder indicating offline learning."""
    return {
        "error": "IQLBL uses offline learning (no episode-based training curve).",
        "training_type": "offline",
    }


def iqlbl_clear_cache():
    """Clear IQLBL-specific cache entries."""
    keys_to_remove = [k for k in _iqlbl_cache if k.startswith("iqlbl_")]
    for k in keys_to_remove:
        del _iqlbl_cache[k]
