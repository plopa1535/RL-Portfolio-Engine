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
    """Load pre-computed test period results from portfolio_cache.json."""
    ok, cached = _is_cached("portfolio")
    if ok:
        return cached

    cache_path = os.path.join(OUTPUT_DIR, "portfolio_cache.json")
    if not os.path.exists(cache_path):
        return {"error": "portfolio_cache.json not found. Run gen_backtest_cache.py locally first."}

    with open(cache_path, "r") as f:
        result = json.load(f)

    _set_cache("portfolio", result)
    return result


def _get_prices():
    """Fetch recent 6-month daily prices from Binance API."""
    ok, cached = _is_cached("prices")
    if ok:
        return cached

    import requests as req
    import pandas as pd

    # Binance daily klines, 180 days
    binance_map = {
        'BTC-USD': 'BTCUSDT', 'ETH-USD': 'ETHUSDT', 'DOGE-USD': 'DOGEUSDT',
        'LTC-USD': 'LTCUSDT', 'XEM-USD': 'XEMUSDT', 'XLM-USD': 'XLMUSDT',
        'SOL-USD': 'SOLUSDT', 'XRP-USD': 'XRPUSDT',
    }

    prices = {}
    latest = {}
    all_dates = None
    tickers_out = []

    for ticker in TICKERS:
        symbol = binance_map.get(ticker)
        if symbol is None:
            continue
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=180"
            resp = req.get(url, timeout=10)
            data = resp.json()
            if not isinstance(data, list) or len(data) == 0:
                continue
            dates = [pd.Timestamp(k[0], unit='ms').strftime("%Y-%m-%d") for k in data]
            closes = [round(float(k[4]), 2) for k in data]
            if all_dates is None:
                all_dates = dates
            prices[ticker] = closes
            latest[ticker] = closes[-1]
            latest[f"{ticker}_change"] = round((closes[-1] - closes[-2]) / closes[-2] * 100, 2) if len(closes) > 1 else 0.0
            tickers_out.append(ticker)
        except Exception:
            continue

    if not prices:
        return {"error": "Failed to fetch prices from Binance"}

    result = {
        "dates": all_dates or [],
        "prices": prices,
        "latest": latest,
        "tickers": tickers_out,
    }
    _set_cache("prices", result)
    return result


def _run_train_simulation():
    """Load pre-computed training results from training_history.json (avoids heavy PyTorch inference on server)."""
    ok, cached = _is_cached("train_portfolio")
    if ok:
        return cached

    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    if not os.path.exists(history_path):
        return {"error": "training_history.json not found. Run training first."}

    with open(history_path, "r") as f:
        history = json.load(f)

    sdelp_values = history.get("last_portfolio_values", [])
    bah_values = history.get("bah_values", [])

    if not sdelp_values:
        return {"error": "No portfolio values in training_history.json"}

    min_len = min(len(sdelp_values), len(bah_values))
    # Generate date labels for training period
    import pandas as pd
    dates = pd.date_range(start=TRAIN_START, periods=min_len, freq="B")
    date_labels = [d.strftime("%Y-%m-%d") for d in dates]

    sdelp_arr = [float(v) for v in sdelp_values[:min_len]]
    bah_arr = [float(v) for v in bah_values[:min_len]]

    sdelp_metrics = compute_metrics(sdelp_arr)
    bah_metrics = compute_metrics(bah_arr)

    result = {
        "dates": date_labels,
        "sdelp_values": sdelp_arr,
        "bah_values": bah_arr,
        "sdelp_metrics": {k: round(float(v), 4) for k, v in sdelp_metrics.items()},
        "bah_metrics": {k: round(float(v), 4) for k, v in bah_metrics.items()},
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
    }

    _set_cache("train_portfolio", result)
    return result


def _run_live_simulation():
    """Load pre-computed recent simulation from live_sim_cache.json."""
    ok, cached = _is_cached("live_sim")
    if ok:
        return cached

    cache_path = os.path.join(OUTPUT_DIR, "live_sim_cache.json")
    if not os.path.exists(cache_path):
        return {"error": "live_sim_cache.json not found. Run gen_backtest_cache.py locally first."}

    with open(cache_path, "r") as f:
        result = json.load(f)

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
