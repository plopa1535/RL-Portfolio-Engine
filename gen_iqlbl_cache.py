"""Pre-compute IQL-BL portfolio results and save as JSON cache files.
Run this locally before deploying to Render.
Outputs: outputs/iqlbl_portfolio_cache.json, outputs/iqlbl_train_cache.json
"""
import sys, os, json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IQLBL_DIR = os.path.join(BASE_DIR, "IQLBL_v1_Dashboard")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, IQLBL_DIR)

import yaml
from two_stage_DYBL_portfolio import (
    load_all_coin_data, PortfolioEnvironment,
    Stage1IQLAgent, DynamicBlackLittermanOptimizer,
    evaluate_portfolio, evaluate_baselines,
    train_stage1_agent, create_stage1_dataset, setup_seed
)

# Load config
config_path = os.path.join(IQLBL_DIR, "bl_portfolio_config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

IQLBL_COINS = ['BTC', 'DOGE', 'ETH', 'NEAR', 'SOL', 'WLD']

# Load data
print("Loading data...")
all_data = load_all_coin_data(IQLBL_DIR, IQLBL_COINS)
env = PortfolioEnvironment(all_data, IQLBL_COINS, config)

# Find date indices
test_start = pd.to_datetime(config['data']['test_start_date'])
train_end_date = pd.to_datetime(config['data']['train_end_date'])
test_start_idx = None
train_end_idx = None
for i, date in enumerate(env.dates):
    if date >= test_start and test_start_idx is None:
        test_start_idx = i
    if date <= train_end_date:
        train_end_idx = i

print(f"test_start_idx={test_start_idx}, train_end_idx={train_end_idx}")

# Train Stage 1 agents
window_size = config['environment']['window_size']
print("Training IQL agents...")
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
print(f"Trained {len(stage1_agents)} agents")


def compute_metrics(values):
    vals = np.array(values, dtype=float)
    if len(vals) < 2:
        return {}
    total_return = vals[-1] / vals[0] - 1
    daily_returns = vals[1:] / vals[:-1] - 1
    ann_return = (1 + total_return) ** (252 / len(vals)) - 1
    ann_vol = float(np.std(daily_returns) * np.sqrt(252))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
    downside = daily_returns[daily_returns < 0]
    sortino_d = float(np.std(downside) * np.sqrt(252)) if len(downside) > 0 else 0.0
    sortino = float(ann_return / sortino_d) if sortino_d > 0 else 0.0
    peak = np.maximum.accumulate(vals)
    mdd = float(np.min((vals - peak) / peak))
    return {
        "CRR": round(float(vals[-1] / vals[0]), 4),
        "AR (%)": round(ann_return * 100, 4),
        "AV (%)": round(ann_vol * 100, 4),
        "Sharpe": round(sharpe, 4),
        "Sortino": round(sortino, 4),
        "MDD (%)": round(abs(mdd) * 100, 4),
    }


def build_result(results, baselines, config, coins):
    dates = results['dates']
    portfolio_values = results['portfolio_values']
    initial_value = config['portfolio'].get('initial_value', 10000)
    norm_values = [v / initial_value for v in portfolio_values]
    norm_bah = [v / initial_value for v in baselines['equal_weight']]
    min_len = min(len(norm_values), len(norm_bah), len(dates))
    date_labels = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d) for d in dates[:min_len]]

    iqlbl_metrics = compute_metrics(norm_values[:min_len])
    bah_metrics = compute_metrics(norm_bah[:min_len])

    weights = results['weights']
    step = max(1, len(weights) // 200)
    weight_dates = date_labels[::step]
    weights_sampled = [w.tolist() if hasattr(w, 'tolist') else list(w) for w in weights[::step]]
    vals = np.array(norm_values[:min_len])
    daily_returns = (vals[1:] / vals[:-1] - 1).tolist() if len(vals) > 1 else []
    current_weights = weights[-1].tolist() if hasattr(weights[-1], 'tolist') else list(weights[-1])

    regimes = results.get('regime', [])
    confidences = results.get('view_confidence', [])
    drawdowns = results.get('drawdown', [])
    defensive_modes = results.get('defensive_mode', [])

    return {
        "dates": date_labels,
        "sdelp_values": [float(v) for v in norm_values[:min_len]],
        "bah_values": [float(v) for v in norm_bah[:min_len]],
        "sdelp_metrics": {k: round(float(v), 4) for k, v in iqlbl_metrics.items()},
        "bah_metrics": {k: round(float(v), 4) for k, v in bah_metrics.items()},
        "current_weights": current_weights,
        "weight_labels": coins,
        "weight_dates": weight_dates,
        "weights_sampled": weights_sampled,
        "daily_returns": daily_returns,
        "daily_return_dates": date_labels[1:],
        "tickers": coins,
        "test_start": config['data'].get('test_start_date', ''),
        "test_end": date_labels[-1] if date_labels else "",
        "regimes": regimes[:min_len] if regimes else [],
        "confidences": [float(c) for c in confidences[:min_len]] if confidences else [],
        "drawdowns": [float(d) for d in drawdowns[:min_len]] if drawdowns else [],
        "defensive_modes": [bool(d) for d in defensive_modes[:min_len]] if defensive_modes else [],
    }


# --- Test period cache ---
print("Evaluating test period...")
bl_test = DynamicBlackLittermanOptimizer(n_assets=len(IQLBL_COINS), config=config)
results_test = evaluate_portfolio(env, stage1_agents, bl_test, config, test_start_idx, len(env.dates))
baselines_test = evaluate_baselines(env, config, test_start_idx, len(env.dates))
result_test = build_result(results_test, baselines_test, config, IQLBL_COINS)
out1 = os.path.join(OUTPUT_DIR, "iqlbl_portfolio_cache.json")
with open(out1, "w") as f:
    json.dump(result_test, f)
print(f"Saved: {out1} ({len(result_test['dates'])} days)")

# --- Train period cache ---
print("Evaluating train period...")
start_idx = window_size + 20
bl_train = DynamicBlackLittermanOptimizer(n_assets=len(IQLBL_COINS), config=config)
results_train = evaluate_portfolio(env, stage1_agents, bl_train, config, start_idx, train_end_idx)
baselines_train = evaluate_baselines(env, config, start_idx, train_end_idx)
result_train = build_result(results_train, baselines_train, config, IQLBL_COINS)
result_train["train_start"] = result_train["dates"][0] if result_train["dates"] else ""
result_train["train_end"] = config['data']['train_end_date']
out2 = os.path.join(OUTPUT_DIR, "iqlbl_train_cache.json")
with open(out2, "w") as f:
    json.dump(result_train, f)
print(f"Saved: {out2} ({len(result_train['dates'])} days)")
print("Done.")
