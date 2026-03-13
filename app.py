"""
SDELP-DDPG Portfolio Dashboard — Flask Backend
실시간 암호화폐 포트폴리오 모니터링 웹 대시보드
"""

import os
import sys
import time
import threading
import numpy as np
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, render_template, send_from_directory, request
from config import *
from environment import PortfolioEnv
from agent import SDELPDDPGAgent
from backtest import compute_metrics

app = Flask(__name__)

# ── IQLBL backend (lazy import to avoid slow startup) ──
_iqlbl_module = None

def _get_iqlbl():
    global _iqlbl_module
    if _iqlbl_module is None:
        import iqlbl_backend
        _iqlbl_module = iqlbl_backend
    return _iqlbl_module


@app.after_request
def add_no_cache_headers(response):
    """Prevent browser from caching API responses."""
    if response.content_type and 'json' in response.content_type:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
    return response


# ── Cache ──
_cache = {}
CACHE_TTL = 300  # 5 minutes (test mode)
_yf_lock = threading.Lock()  # Prevent concurrent yfinance downloads


def _is_cached(key):
    if key in _cache:
        ts, data = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return True, data
    return False, None


def _set_cache(key, data):
    _cache[key] = (time.time(), data)


# ── Model (loaded once at startup) ──
_agent = None
_model_loaded = False


def _get_agent(state_dim, action_dim):
    global _agent, _model_loaded
    if _model_loaded and _agent is not None:
        return _agent
    model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        return None
    _agent = SDELPDDPGAgent(state_dim, action_dim)
    _agent.load(model_path)
    _model_loaded = True
    return _agent


def _run_portfolio_simulation():
    """Run model inference on test period and return results."""
    ok, cached = _is_cached("portfolio")
    if ok:
        return cached

    with _yf_lock:
        # Double-check cache after acquiring lock
        ok, cached = _is_cached("portfolio")
        if ok:
            return cached

        try:
            env = PortfolioEnv(tickers=TICKERS, start=TEST_START, end=TEST_END,
                               window=WINDOW_SIZE)
        except Exception as e:
            return {"error": f"Data loading failed: {str(e)}"}

    agent = _get_agent(env.state_dim, env.action_dim)
    if agent is None:
        return {"error": "Model not found (outputs/best_model.pt)"}

    # Run backtest
    state = env.reset()
    while True:
        action = agent.select_action(state, explore=False)
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            break

    sdelp_values = env.portfolio_values
    bah_values = env.get_buy_and_hold_values()
    sdelp_metrics = compute_metrics(sdelp_values)
    bah_metrics = compute_metrics(bah_values)

    # Date labels
    dates = env.dates[env.window:]
    min_len = min(len(sdelp_values), len(bah_values), len(dates))
    date_labels = [d.strftime("%Y-%m-%d") for d in dates[:min_len].to_pydatetime()]

    # Weight history
    weight_history = np.array(env.weight_history)
    weight_labels = ["Cash"] + env.tickers
    # Downsample weights for chart (every 5 days)
    step = max(1, len(weight_history) // 200)
    weight_dates = date_labels[::step]
    weights_sampled = weight_history[::step].tolist()

    # Daily returns
    vals = np.array(sdelp_values[:min_len])
    daily_returns = (vals[1:] / vals[:-1] - 1).tolist()

    # Current (latest) weights
    current_weights = weight_history[-1].tolist()

    result = {
        "dates": date_labels,
        "sdelp_values": [float(v) for v in sdelp_values[:min_len]],
        "bah_values": [float(v) for v in bah_values[:min_len]],
        "sdelp_metrics": {k: round(float(v), 4) for k, v in sdelp_metrics.items()},
        "bah_metrics": {k: round(float(v), 4) for k, v in bah_metrics.items()},
        "current_weights": current_weights,
        "weight_labels": weight_labels,
        "weight_dates": weight_dates,
        "weights_sampled": weights_sampled,
        "daily_returns": daily_returns,
        "daily_return_dates": date_labels[1:],
        "tickers": env.tickers,
        "test_start": TEST_START,
        "test_end": TEST_END,
    }

    _set_cache("portfolio", result)
    return result


def _get_prices():
    """Fetch recent crypto prices from yfinance."""
    ok, cached = _is_cached("prices")
    if ok:
        return cached

    import yfinance as yf
    try:
        df = yf.download(TICKERS, period="6mo", auto_adjust=True)
        close = df["Close"] if len(TICKERS) > 1 else df["Close"].to_frame()
        close = close.ffill().bfill()

        dates = [d.strftime("%Y-%m-%d") for d in close.index.to_pydatetime()]
        prices = {}
        for ticker in close.columns:
            col = close[ticker].tolist()
            prices[ticker] = [round(float(v), 2) if not np.isnan(v) else None for v in col]

        # Latest prices
        latest = {}
        for ticker in close.columns:
            val = close[ticker].dropna()
            if len(val) > 0:
                latest[ticker] = round(float(val.iloc[-1]), 2)
                prev = float(val.iloc[-2]) if len(val) > 1 else float(val.iloc[-1])
                latest[f"{ticker}_change"] = round((latest[ticker] - prev) / prev * 100, 2)

        result = {
            "dates": dates,
            "prices": prices,
            "latest": latest,
            "tickers": list(close.columns),
        }
        _set_cache("prices", result)
        return result
    except Exception as e:
        return {"error": str(e)}


def _run_train_simulation():
    """Run best model inference on training period data."""
    ok, cached = _is_cached("train_portfolio")
    if ok:
        return cached

    with _yf_lock:
        ok, cached = _is_cached("train_portfolio")
        if ok:
            return cached
        try:
            env = PortfolioEnv(tickers=TICKERS, start=TRAIN_START, end=TRAIN_END,
                               window=WINDOW_SIZE)
        except Exception as e:
            return {"error": f"Train data loading failed: {str(e)}"}

    agent = _get_agent(env.state_dim, env.action_dim)
    if agent is None:
        return {"error": "Model not found (outputs/best_model.pt)"}

    state = env.reset()
    while True:
        action = agent.select_action(state, explore=False)
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            break

    sdelp_values = env.portfolio_values
    bah_values = env.get_buy_and_hold_values()

    dates = env.dates[env.window:]
    min_len = min(len(sdelp_values), len(bah_values), len(dates))
    date_labels = [d.strftime("%Y-%m-%d") for d in dates[:min_len].to_pydatetime()]

    sdelp_metrics = compute_metrics(sdelp_values[:min_len])
    bah_metrics = compute_metrics(bah_values[:min_len])

    result = {
        "dates": date_labels,
        "sdelp_values": [float(v) for v in sdelp_values[:min_len]],
        "bah_values": [float(v) for v in bah_values[:min_len]],
        "sdelp_metrics": {k: round(float(v), 4) for k, v in sdelp_metrics.items()},
        "bah_metrics": {k: round(float(v), 4) for k, v in bah_metrics.items()},
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
    }

    _set_cache("train_portfolio", result)
    return result


def _run_live_simulation():
    """Run forward simulation using trained model on most recent market data (last 30 days)."""
    ok, cached = _is_cached("live_sim")
    if ok:
        return cached

    from datetime import datetime, timedelta

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=150)).strftime("%Y-%m-%d")

    with _yf_lock:
        ok, cached = _is_cached("live_sim")
        if ok:
            return cached
        try:
            env = PortfolioEnv(tickers=TICKERS, start=start_date, end=end_date,
                               window=WINDOW_SIZE)
        except Exception as e:
            return {"error": f"Live data loading failed: {str(e)}"}

    agent = _get_agent(env.state_dim, env.action_dim)
    if agent is None:
        return {"error": "Model not found (outputs/best_model.pt)"}

    state = env.reset()
    last_action = None
    while True:
        action = agent.select_action(state, explore=False)
        last_action = action
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            break

    sdelp_values = env.portfolio_values
    bah_values = env.get_buy_and_hold_values()
    dates = env.dates[env.window:]
    min_len = min(len(sdelp_values), len(bah_values), len(dates))
    date_labels = [d.strftime("%Y-%m-%d") for d in dates[:min_len].to_pydatetime()]

    n = min(30, min_len)
    result = {
        "dates": date_labels[-n:],
        "sdelp_values": [float(v) for v in list(sdelp_values[:min_len])[-n:]],
        "bah_values": [float(v) for v in list(bah_values[:min_len])[-n:]],
        "current_weights": [float(w) for w in last_action],
        "tickers": ["CASH"] + env.tickers,
    }

    _set_cache("live_sim", result)
    return result


def _get_week_start_ms():
    """Return this week's Monday KST 00:00 as millisecond timestamp."""
    from datetime import datetime, timezone, timedelta
    KST = timezone(timedelta(hours=9))
    now_kst = datetime.now(KST)
    monday = now_kst - timedelta(days=now_kst.weekday())
    monday_midnight = monday.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(monday_midnight.timestamp() * 1000)


def _fetch_binance_5min(symbol, start_ms=None):
    """Fetch 5-min klines from Binance public API.
    If start_ms is None, defaults to this week's Monday KST 00:00.
    Automatically paginates for ranges exceeding 1000 candles."""
    import requests as req
    import pandas as pd
    from datetime import datetime, timezone, timedelta

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
        # Move cursor past last candle
        cursor = batch[-1][0] + 1
        if len(batch) < 1000:
            break

    if len(all_data) == 0:
        return None
    times = [datetime.fromtimestamp(k[0] / 1000, tz=KST) for k in all_data]
    closes = [float(k[4]) for k in all_data]
    return pd.Series(closes, index=times)


# Map SDELP yfinance tickers to Binance symbols
_SDELP_BINANCE_MAP = {
    'BTC-USD': 'BTCUSDT', 'ETH-USD': 'ETHUSDT', 'DOGE-USD': 'DOGEUSDT',
    'LTC-USD': 'LTCUSDT', 'XEM-USD': 'XEMUSDT', 'XLM-USD': 'XLMUSDT',
    'SOL-USD': 'SOLUSDT', 'XRP-USD': 'XRPUSDT',
}


def _run_live_5min():
    """Fetch 5-min interval price data from Binance and compute portfolio returns."""
    ok, cached = _is_cached("live_5min")
    if ok:
        return cached

    import pandas as pd

    frames = {}
    for ticker in TICKERS:
        symbol = _SDELP_BINANCE_MAP.get(ticker)
        if not symbol:
            continue  # skip USDT-USD (stablecoin, always ~1.0)
        try:
            series = _fetch_binance_5min(symbol)
            if series is not None and len(series) > 0:
                frames[ticker] = series
        except Exception:
            pass

    if len(frames) < 2:
        return {"error": "Not enough 5-min data from Binance"}

    prices = pd.DataFrame(frames).dropna()
    if len(prices) < 2:
        return {"error": "Not enough aligned data"}

    timestamps = [t.strftime("%m/%d %H:%M") for t in prices.index]
    base = prices.iloc[0]
    normed = prices.div(base)

    # B&H: equal weight
    bah = normed.mean(axis=1)
    bah_pct = ((bah / bah.iloc[0]) - 1) * 100

    # SDELP: model weights from daily simulation
    daily = _run_live_simulation()
    if "error" not in daily and "current_weights" in daily:
        weights = daily["current_weights"]
        ticker_names = daily["tickers"]

        w = []
        for t in prices.columns:
            if t in ticker_names:
                w.append(weights[ticker_names.index(t)])
            else:
                w.append(1.0 / len(prices.columns))
        w = np.array(w)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum

        sdelp = (normed * w).sum(axis=1)
        sdelp_pct = ((sdelp / sdelp.iloc[0]) - 1) * 100
    else:
        sdelp_pct = bah_pct

    result = {
        "dates": timestamps,
        "sdelp_values": [float(v) for v in sdelp_pct.values],
        "bah_values": [float(v) for v in bah_pct.values],
    }

    _set_cache("live_5min", result)
    return result


def _run_combined_simulation():
    """Combine train + test portfolio simulations into one timeline."""
    ok, cached = _is_cached("combined_portfolio")
    if ok:
        return cached

    train = _run_train_simulation()
    test = _run_portfolio_simulation()

    if "error" in train or "error" in test:
        return {"error": train.get("error") or test.get("error")}

    # Scale test values so they start where train ended
    train_end_sdelp = train["sdelp_values"][-1]
    train_end_bah = train["bah_values"][-1]

    test_sdelp_scaled = [v * train_end_sdelp for v in test["sdelp_values"]]
    test_bah_scaled = [v * train_end_bah for v in test["bah_values"]]

    result = {
        "train_dates": train["dates"],
        "test_dates": test["dates"],
        "dates": train["dates"] + test["dates"],
        "sdelp_values": train["sdelp_values"] + test_sdelp_scaled,
        "bah_values": train["bah_values"] + test_bah_scaled,
        "train_end_index": len(train["dates"]) - 1,
        "train_sdelp_metrics": train["sdelp_metrics"],
        "train_bah_metrics": train["bah_metrics"],
        "test_sdelp_metrics": test["sdelp_metrics"],
        "test_bah_metrics": test["bah_metrics"],
        "train_period": f"{TRAIN_START} ~ {TRAIN_END}",
        "test_period": f"{TEST_START} ~ {TEST_END}",
    }

    _set_cache("combined_portfolio", result)
    return result


# ── Routes ──

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/portfolio")
def api_portfolio():
    model = request.args.get("model", "sdelp")
    if model == "iqlbl":
        return jsonify(_get_iqlbl().iqlbl_run_portfolio())
    return jsonify(_run_portfolio_simulation())


@app.route("/api/train-portfolio")
def api_train_portfolio():
    model = request.args.get("model", "sdelp")
    if model == "iqlbl":
        return jsonify(_get_iqlbl().iqlbl_run_train_portfolio())
    return jsonify(_run_train_simulation())


@app.route("/api/combined-portfolio")
def api_combined_portfolio():
    return jsonify(_run_combined_simulation())


@app.route("/api/live-simulation")
def api_live_simulation():
    return jsonify(_run_live_simulation())


@app.route("/api/live-5min")
def api_live_5min():
    model = request.args.get("model", "sdelp")
    if model == "iqlbl":
        return jsonify(_get_iqlbl().iqlbl_run_live_5min())
    return jsonify(_run_live_5min())


@app.route("/api/prices")
def api_prices():
    return jsonify(_get_prices())


@app.route("/api/metrics")
def api_metrics():
    model = request.args.get("model", "sdelp")
    if model == "iqlbl":
        result = _get_iqlbl().iqlbl_run_portfolio()
    else:
        result = _run_portfolio_simulation()
    if "error" in result:
        return jsonify(result), 500
    return jsonify({
        "sdelp": result["sdelp_metrics"],
        "bah": result["bah_metrics"],
    })


@app.route("/api/model-info")
def api_model_info():
    model = request.args.get("model", "sdelp")
    if model == "iqlbl":
        return jsonify(_get_iqlbl().iqlbl_get_model_info())
    return jsonify({
        "levy_alpha": LEVY_ALPHA,
        "sde_steps": SDE_STEPS,
        "beta": BETA,
        "window_size": WINDOW_SIZE,
        "transaction_cost": TRANSACTION_COST,
        "gamma": GAMMA,
        "tau": TAU,
        "batch_size": BATCH_SIZE,
        "buffer_size": BUFFER_SIZE,
        "actor_lr": ACTOR_LR,
        "critic_lr": CRITIC_LR,
        "num_assets": NUM_ASSETS,
        "tickers": TICKERS,
        "train_period": f"{TRAIN_START} ~ {TRAIN_END}",
        "test_period": f"{TEST_START} ~ {TEST_END}",
        "device": str(DEVICE),
    })


@app.route("/api/models")
def api_models():
    """Return available models."""
    return jsonify({
        "models": [
            {"id": "sdelp", "name": "SDELP-DDPG", "description": "SDE Lévy Process + DDPG"},
            {"id": "iqlbl", "name": "IQL-BL", "description": "IQL + Dynamic Black-Litterman"},
        ]
    })


@app.route("/api/refresh")
def api_refresh():
    """Clear cache and force refresh."""
    _cache.clear()
    global _model_loaded
    _model_loaded = False
    return jsonify({"status": "ok", "message": "Cache cleared"})


@app.route("/api/training-history")
def api_training_history():
    """Return training history for interactive Chart.js charts."""
    model = request.args.get("model", "sdelp")
    if model == "iqlbl":
        return jsonify(_get_iqlbl().iqlbl_get_training_history())
    import json
    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    if not os.path.exists(history_path):
        return jsonify({"error": "training_history.json not found. Run training first."}), 404
    with open(history_path, "r") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    """Serve training output images."""
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    print("=" * 60)
    print("  RL Portfolio Dashboard (SDELP-DDPG + IQL-BL)")
    port = int(os.environ.get("PORT", 5000))
    print(f"  http://localhost:{port}")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=port)
