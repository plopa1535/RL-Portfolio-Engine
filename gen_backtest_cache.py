"""Pre-compute all heavy simulation results and save as JSON cache files.
Run this locally before deploying to Render.
Outputs: outputs/portfolio_cache.json, outputs/live_sim_cache.json
"""
import json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from environment import PortfolioEnv
from agent import SDELPDDPGAgent
from datetime import datetime, timedelta


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
    sortino_denom = float(np.std(downside) * np.sqrt(252)) if len(downside) > 0 else 0.0
    sortino = float(ann_return / sortino_denom) if sortino_denom > 0 else 0.0
    peak = np.maximum.accumulate(vals)
    drawdown = (vals - peak) / peak
    mdd = float(np.min(drawdown))
    crr = float(vals[-1] / vals[0])
    return {
        "CRR": round(crr, 4),
        "Annualized Return": round(ann_return, 4),
        "Annualized Volatility": round(ann_vol, 4),
        "Sharpe Ratio": round(sharpe, 4),
        "Sortino Ratio": round(sortino, 4),
        "MDD": round(mdd, 4),
    }


def run_simulation(env, agent):
    state = env.reset()
    last_action = None
    while True:
        action = agent.select_action(state, explore=False)
        last_action = action
        next_state, reward, done, info = env.step(action)
        state = next_state
        if done:
            break
    return last_action


# --- Test period (portfolio_cache.json) ---
print("Running test period simulation...")
env_test = PortfolioEnv(tickers=TICKERS, start=TEST_START, end=TEST_END, window=WINDOW_SIZE)
agent = SDELPDDPGAgent(env_test.state_dim, env_test.action_dim)
agent.load(os.path.join(OUTPUT_DIR, "best_model.pt"))

last_action = run_simulation(env_test, agent)

sdelp_values = env_test.portfolio_values
bah_values = env_test.get_buy_and_hold_values()
dates = env_test.dates[env_test.window:]
min_len = min(len(sdelp_values), len(bah_values), len(dates))
date_labels = [d.strftime("%Y-%m-%d") for d in dates[:min_len].to_pydatetime()]

weight_history = np.array(env_test.weight_history)
weight_labels = ["Cash"] + env_test.tickers
step = max(1, len(weight_history) // 200)
weight_dates = date_labels[::step]
weights_sampled = weight_history[::step].tolist()

vals = np.array(sdelp_values[:min_len])
daily_returns = (vals[1:] / vals[:-1] - 1).tolist()
current_weights = weight_history[-1].tolist()

portfolio_cache = {
    "dates": date_labels,
    "sdelp_values": [float(v) for v in sdelp_values[:min_len]],
    "bah_values": [float(v) for v in bah_values[:min_len]],
    "sdelp_metrics": compute_metrics(sdelp_values[:min_len]),
    "bah_metrics": compute_metrics(bah_values[:min_len]),
    "current_weights": current_weights,
    "weight_labels": weight_labels,
    "weight_dates": weight_dates,
    "weights_sampled": weights_sampled,
    "daily_returns": daily_returns,
    "daily_return_dates": date_labels[1:],
    "tickers": env_test.tickers,
    "test_start": TEST_START,
    "test_end": TEST_END,
}

out_path = os.path.join(OUTPUT_DIR, "portfolio_cache.json")
with open(out_path, "w") as f:
    json.dump(portfolio_cache, f)
print(f"Saved: {out_path} ({len(date_labels)} days)")

# --- Recent 30-day simulation (live_sim_cache.json) ---
print("Running recent 30-day simulation...")
end_date = datetime.now().strftime("%Y-%m-%d")
start_date = (datetime.now() - timedelta(days=150)).strftime("%Y-%m-%d")

env_live = PortfolioEnv(tickers=TICKERS, start=start_date, end=end_date, window=WINDOW_SIZE)
agent2 = SDELPDDPGAgent(env_live.state_dim, env_live.action_dim)
agent2.load(os.path.join(OUTPUT_DIR, "best_model.pt"))

last_action_live = run_simulation(env_live, agent2)

sv = env_live.portfolio_values
bv = env_live.get_buy_and_hold_values()
d_live = env_live.dates[env_live.window:]
ml = min(len(sv), len(bv), len(d_live))
dl = [d.strftime("%Y-%m-%d") for d in d_live[:ml].to_pydatetime()]

n = min(30, ml)
live_sim_cache = {
    "dates": dl[-n:],
    "sdelp_values": [float(v) for v in list(sv[:ml])[-n:]],
    "bah_values": [float(v) for v in list(bv[:ml])[-n:]],
    "current_weights": [float(w) for w in last_action_live],
    "tickers": ["CASH"] + env_live.tickers,
}

out_path2 = os.path.join(OUTPUT_DIR, "live_sim_cache.json")
with open(out_path2, "w") as f:
    json.dump(live_sim_cache, f)
print(f"Saved: {out_path2}")
print("Done. Commit outputs/portfolio_cache.json and outputs/live_sim_cache.json to git.")
