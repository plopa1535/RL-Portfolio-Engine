"""
SDELP-DDPG Backtesting (테스트 기간 전용)
논문 Table 3과의 정확한 비교를 위해 Test 기간만 평가
X축을 실제 날짜로 표시
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from environment import PortfolioEnv
from agent import SDELPDDPGAgent


def compute_metrics(portfolio_values):
    """논문 Table 3의 평가지표 계산"""
    values = np.array(portfolio_values)
    T = len(values) - 1
    years = T / 252

    daily_returns = values[1:] / values[:-1] - 1
    crr = values[-1] / values[0]
    ar = (crr ** (1 / years) - 1) * 100
    av = np.std(daily_returns) * np.sqrt(252) * 100

    mean_daily = np.mean(daily_returns)
    std_daily = np.std(daily_returns)
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0

    downside = daily_returns[daily_returns < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 1e-8
    sortino = (mean_daily / downside_std) * np.sqrt(252)

    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    mdd = np.max(drawdown) * 100

    return {
        "CRR": crr,
        "AR (%)": ar,
        "AV (%)": av,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MDD (%)": mdd,
    }


def backtest():
    """테스트 기간 백테스팅"""
    print("=" * 60)
    print("  SDELP-DDPG Backtesting (Test Period Only)")
    print(f"  기간: {TEST_START} ~ {TEST_END}")
    print("=" * 60)

    # ── 테스트 환경 생성 ──
    print("\n[1/3] 테스트 환경 초기화...")
    test_env = PortfolioEnv(
        tickers=TICKERS,
        start=TEST_START,
        end=TEST_END,
        window=WINDOW_SIZE,
    )
    print(f"  자산: {test_env.tickers}")
    print(f"  State 차원: {test_env.state_dim}")

    # ── 모델 로드 ──
    print("\n[2/3] 학습된 모델 로드...")
    model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"  ❌ 모델 파일 없음: {model_path}")
        return

    agent = SDELPDDPGAgent(test_env.state_dim, test_env.action_dim)
    agent.load(model_path)
    print(f"  ✅ 모델 로드 완료: {model_path}")

    # ── 백테스팅 실행 ──
    print("\n[3/3] 백테스팅 실행...")
    state = test_env.reset()
    while True:
        action = agent.select_action(state, explore=False)
        next_state, reward, done, info = test_env.step(action)
        state = next_state
        if done:
            break

    # ── 결과 계산 ──
    sdelp_values = test_env.portfolio_values
    bah_values = test_env.get_buy_and_hold_values()
    sdelp_metrics = compute_metrics(sdelp_values)
    bah_metrics = compute_metrics(bah_values)

    # ── 날짜 인덱스 생성 ──
    # 포트폴리오 값은 window 이후부터 시작
    all_dates = test_env.dates[test_env.window:]
    # portfolio_values는 [초기값] + 스텝별 값이므로 길이가 num_steps+1
    min_len = min(len(sdelp_values), len(bah_values))
    # 날짜도 길이 맞춤 (첫 번째는 초기값이므로 날짜 하나 앞에 추가)
    if len(all_dates) >= min_len:
        plot_dates = all_dates[:min_len].to_pydatetime()
    else:
        # fallback: 날짜가 부족하면 마지막 날짜를 채움
        plot_dates = all_dates.to_pydatetime()
        min_len = len(plot_dates)

    # ── 결과 출력 ──
    print("\n" + "=" * 80)
    print("  BACKTEST RESULTS — Crypto Dataset")
    print("=" * 80)
    header = f"{'Model':<25} {'AR(%)':<10} {'Sharpe':<10} {'AV(%)':<10} {'Sortino':<10} {'MDD(%)':<10} {'CRR':<10}"
    print(header)
    print("-" * 80)
    m = sdelp_metrics
    print(f"{'SDELP-DDPG (ours)':<25} {m['AR (%)']:<10.2f} {m['Sharpe']:<10.3f} {m['AV (%)']:<10.2f} {m['Sortino']:<10.3f} {m['MDD (%)']:<10.2f} {m['CRR']:<10.4f}")
    m = bah_metrics
    print(f"{'BAH (ours)':<25} {m['AR (%)']:<10.2f} {m['Sharpe']:<10.3f} {m['AV (%)']:<10.2f} {m['Sortino']:<10.3f} {m['MDD (%)']:<10.2f} {m['CRR']:<10.4f}")
    print("=" * 80)

    # ── 시각화 (실제 날짜 X축) ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        f"SDELP-DDPG Backtest: {TEST_START} ~ {TEST_END}\n"
        f"Crypto 9 Assets | {NUM_EPISODES} Trajectories",
        fontsize=14, fontweight="bold"
    )

    # (1) 포트폴리오 가치 곡선 — 날짜 X축
    ax1 = axes[0]
    ax1.plot(plot_dates, sdelp_values[:min_len], color="crimson", linewidth=2,
             label=f"SDELP-DDPG (CRR={sdelp_metrics['CRR']:.2f})")
    ax1.plot(plot_dates, bah_values[:min_len], color="gray", linewidth=2,
             linestyle="--",
             label=f"Buy & Hold (CRR={bah_metrics['CRR']:.2f})")
    ax1.axhline(y=1.0, color="black", linewidth=0.5, linestyle=":")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Value")
    ax1.set_title("Portfolio Value Over Time")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 날짜 포맷 설정
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # (2) 지표 비교 바 차트
    ax2 = axes[1]
    metrics_names = ["AR (%)", "Sharpe", "MDD (%)", "CRR"]
    x = np.arange(len(metrics_names))
    width = 0.3

    ours_vals = [sdelp_metrics[k] for k in metrics_names]
    bah_vals_chart = [bah_metrics[k] for k in metrics_names]

    bars1 = ax2.bar(x - width/2, ours_vals, width, label="SDELP-DDPG",
                    color="crimson", alpha=0.8)
    bars2 = ax2.bar(x + width/2, bah_vals_chart, width, label="Buy & Hold",
                    color="gray", alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_names)
    ax2.set_title("Metrics Comparison")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "backtest_results.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 그래프 저장: {save_path}")

    return sdelp_metrics, bah_metrics


if __name__ == "__main__":
    backtest()
