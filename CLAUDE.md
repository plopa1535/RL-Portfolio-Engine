# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SDELP-DDPG: Reinforcement learning for cryptocurrency portfolio management using Stochastic Differential Equations with Lévy Processes and Deep Deterministic Policy Gradient. Based on a paper published in Expert Systems With Applications (2025). Documentation is in Korean; code comments are in English.

## Commands

```bash
# Full pipeline (noise analysis → training → output listing)
python run_all.py

# Individual components
python compare_levy_gaussian.py   # Lévy vs Gaussian noise comparison
python train.py                   # Train the agent
python backtest.py                # Evaluate on test period (requires outputs/best_model.pt)
```

All outputs (model checkpoints, plots) are saved to `outputs/`.

## Dependencies

```bash
pip install torch numpy matplotlib yfinance scipy pandas pyyaml
```

## Architecture

**Data flow**: `config.yaml` → `config.py` → `environment.py` / `networks.py` → `agent.py` → `train.py` / `backtest.py`

- All modules do `from config import *` to get shared constants. `config.yaml` is the only file users should edit (episode count, date ranges, tickers). Other hyperparameters live as constants in `config.py`.
- `run_all.py` runs noise comparison + training but does **not** include backtesting. Run `backtest.py` separately after training.

### Environment (`environment.py`)
- **PortfolioEnv**: MDP with state = (50-day price return window × 9 crypto assets) + current portfolio weights = 460-dim. Action = softmax-normalized portfolio weights over cash + 9 assets.
- **Reward**: `R = log(portfolio_return) - β × KL_divergence(new_weights || old_weights)`. The KL term penalizes excessive rebalancing.
- **Transaction cost**: proportional to turnover `c_s × Σ|Δw|`.
- Downloads data from Yahoo Finance via `yfinance`.

### Networks (`networks.py`)
- **SDEActor**: Conv2D feature extraction → initial action → K-step Euler-Maruyama SDE loop (`a_{k+1} = a_k + drift·Δt + diffusion·(Δt)^{1/α}·Lévy_noise`) → MultiheadAttention selects best trajectory point → softmax output.
- **DriftNet/DiffusionNet**: Use LayerNorm (not BatchNorm) because batch_size=1 during inference. However, **ConvFeatureExtractor** (used in both Actor and Critic) and **Critic's ResidualBlocks** use BatchNorm2d — so `actor.eval()`/`actor.train()` toggling in `agent.py:select_action` is critical for correct single-sample inference.
- **Lévy noise**: α-stable distribution from `scipy.stats.levy_stable`, clipped to [-5, 5].
- **Critic**: Conv2D + ResidualBlocks for state path, dense layers for action path, merged via concatenation → Q-value.

### Agent (`agent.py`)
- **SDELPDDPGAgent**: Online + target networks for both actor and critic. Ornstein-Uhlenbeck process for exploration noise. Gradient clipping (norm ≤ 1.0). Soft target update with τ=0.001.
- **ReplayBuffer**: Deque-based, capacity 50,000.

### Key Hyperparameters (in `config.yaml` and `config.py`)
- `LEVY_ALPHA` (1.4): Controls tail heaviness of exploration noise. Lower = more aggressive jumps.
- `SDE_STEPS` (5): Number of Euler-Maruyama discretization steps in actor.
- `BETA` (0.05): Risk penalty weight in reward function.
- `WINDOW_SIZE` (50): Lookback period for price returns.
- Tickers: BTC, ETH, DOGE, LTC, USDT, XEM, XLM, SOL, XRP.
- Train period: 2018-01-01 to 2022-12-31. Test period: 2023-01-01 to 2026-01-31.

## Key Implementation Details

- All matplotlib plots use `Agg` backend (non-interactive, server-safe).
- `best_model.pt` is saved based on highest portfolio value during training.
- Backtest metrics: CRR, Annualized Return, Annualized Volatility, Sharpe Ratio, Sortino Ratio, MDD. Compared against equal-weight buy-and-hold benchmark.
- Device auto-selects CUDA if available.
- `environment.py` monkey-patches `curl_cffi` to disable SSL verification for `yfinance` data downloads. This is a workaround for certificate issues in some environments.
- State dimension = `NUM_ASSETS × WINDOW_SIZE + TOTAL_ASSETS` = 9×50 + 10 = 460. If tickers are dropped due to missing data, dimensions change dynamically and the model architecture adapts via `env.state_dim`/`env.action_dim`.
