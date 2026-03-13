"""
SDELP-DDPG Training & Visualization
학습 실행 + 성과 그래프 생성
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 서버 환경 호환
import torch
import time
import os
from config import *
from environment import PortfolioEnv
from agent import SDELPDDPGAgent


def train():
    """메인 학습 루프"""

    print("=" * 60)
    print("  SDELP-DDPG: Portfolio Management with Levy Process")
    print("  강화학습 학회 Session 3 | Live Implementation")
    print("=" * 60)

    # ── 환경 초기화 ──
    print("\n[1/4] 환경 초기화...")
    env = PortfolioEnv()
    print(f"  자산: {TICKERS}")
    print(f"  학습 기간: {TRAIN_START} ~ {TRAIN_END}")
    print(f"  State 차원: {env.state_dim}")
    print(f"  Action 차원: {env.action_dim}")
    print(f"  에피소드 길이: ~{env.num_steps} 스텝")

    # ── 에이전트 초기화 ──
    print("\n[2/4] SDELP-DDPG 에이전트 초기화...")
    agent = SDELPDDPGAgent(env.state_dim, env.action_dim)
    print(f"  SDE Steps (K): {SDE_STEPS}")
    print(f"  Levy α: {LEVY_ALPHA}")
    print(f"  리스크 페널티 β: {BETA}")
    print(f"  디바이스: {DEVICE}")

    # Actor 파라미터 수
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    print(f"  Actor 파라미터: {actor_params:,}")
    print(f"  Critic 파라미터: {critic_params:,}")

    # ── 학습 ──
    print(f"\n[3/4] 학습 시작 ({NUM_EPISODES} 에피소드)...")
    print("-" * 60)

    episode_rewards = []
    episode_values = []
    all_portfolio_values = []
    best_value = 0.0
    start_time = time.time()

    for ep in range(NUM_EPISODES):
        state = env.reset()
        agent.ou_process.reset()
        ep_reward = 0.0
        step = 0

        while True:
            # 행동 선택
            action = agent.select_action(state, explore=True)

            # 환경 스텝
            next_state, reward, done, info = env.step(action)

            # 경험 저장
            agent.replay_buffer.push(
                state, action, reward, next_state, float(done)
            )

            # 네트워크 업데이트
            agent.update()

            ep_reward += reward
            state = next_state
            step += 1

            if done:
                break

        final_value = info.get("portfolio_value", env.portfolio_value)
        episode_rewards.append(ep_reward)
        episode_values.append(final_value)
        all_portfolio_values.append(env.portfolio_values.copy())

        # 최고 성과 기록
        if final_value > best_value:
            best_value = final_value
            agent.save(os.path.join(OUTPUT_DIR, "best_model.pt"))

        # 로그 출력
        if (ep + 1) % 10 == 0 or ep == 0:
            elapsed = time.time() - start_time
            avg_reward = np.mean(episode_rewards[-10:])
            avg_value = np.mean(episode_values[-10:])
            print(
                f"  Episode {ep+1:>3}/{NUM_EPISODES} | "
                f"Reward: {ep_reward:>8.4f} | "
                f"Portfolio: {final_value:>6.4f} | "
                f"Avg(10): {avg_value:>6.4f} | "
                f"Best: {best_value:>6.4f} | "
                f"Buffer: {len(agent.replay_buffer):>5} | "
                f"Time: {elapsed:>5.1f}s"
            )

    print("-" * 60)
    total_time = time.time() - start_time
    print(f"  학습 완료! 총 소요: {total_time:.1f}s")
    print(f"  최종 포트폴리오: {episode_values[-1]:.4f}")
    print(f"  최고 포트폴리오: {best_value:.4f}")

    # ── Save training history as JSON (for dashboard) ──
    import json
    history = {
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_values": [float(v) for v in episode_values],
        "actor_losses": [float(l) for l in agent.actor_losses[-2000:]],
        "critic_losses": [float(l) for l in agent.critic_losses[-2000:]],
        "best_value": float(best_value),
        "total_time": round(total_time, 1),
        "num_episodes": NUM_EPISODES,
        "last_portfolio_values": [float(v) for v in all_portfolio_values[-1]],
        "bah_values": [float(v) for v in env.get_buy_and_hold_values()],
    }
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f)
    print(f"  학습 히스토리 JSON 저장 완료")

    # ── 시각화 ──
    print(f"\n[4/4] 결과 시각화...")
    plot_results(episode_rewards, episode_values, all_portfolio_values, env, agent)
    print(f"  그래프 저장 완료 → {OUTPUT_DIR}/")

    return agent, env, episode_rewards, episode_values


def plot_results(rewards, values, all_pv, env, agent):
    """종합 결과 시각화 (4-panel)"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "SDELP-DDPG: Portfolio Management with Levy Process\n"
        f"Assets: {TICKERS} | α={LEVY_ALPHA} | β={BETA} | K={SDE_STEPS}",
        fontsize=14, fontweight="bold"
    )

    # ── (1) 학습 곡선: 에피소드별 누적 보상 ──
    ax1 = axes[0, 0]
    ax1.plot(rewards, alpha=0.3, color="steelblue", label="Episode Reward")
    # 이동평균
    if len(rewards) >= 10:
        ma = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        ax1.plot(range(9, len(rewards)), ma, color="navy", linewidth=2,
                 label="Moving Avg (10)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.set_title("① Learning Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── (2) 포트폴리오 가치 vs Buy & Hold ──
    ax2 = axes[0, 1]
    # 마지막 에피소드의 포트폴리오 가치
    last_pv = all_pv[-1] if all_pv else [1.0]
    bh_values = env.get_buy_and_hold_values()
    min_len = min(len(last_pv), len(bh_values))

    ax2.plot(last_pv[:min_len], color="crimson", linewidth=2,
             label=f"SDELP-DDPG ({last_pv[-1]:.4f})")
    ax2.plot(bh_values[:min_len], color="gray", linewidth=2, linestyle="--",
             label=f"Buy & Hold ({bh_values[min_len-1]:.4f})")
    ax2.axhline(y=1.0, color="black", linewidth=0.5, linestyle=":")
    ax2.set_xlabel("Trading Day")
    ax2.set_ylabel("Portfolio Value")
    ax2.set_title("② Portfolio Value (Last Episode)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── (3) 에피소드별 최종 포트폴리오 가치 ──
    ax3 = axes[1, 0]
    ax3.plot(values, color="forestgreen", alpha=0.7, label="Final Portfolio Value")
    if len(values) >= 10:
        ma_v = np.convolve(values, np.ones(10) / 10, mode="valid")
        ax3.plot(range(9, len(values)), ma_v, color="darkgreen", linewidth=2,
                 label="Moving Avg (10)")
    ax3.axhline(y=1.0, color="red", linewidth=1, linestyle="--", label="Break-even")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Final Portfolio Value")
    ax3.set_title("③ Portfolio Value per Episode")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # ── (4) 포트폴리오 가중치 변화 (마지막 에피소드) ──
    ax4 = axes[1, 1]
    weight_history = np.array(env.weight_history)  # (steps, total_assets)
    labels = ["Cash"] + TICKERS
    colors = ["#95a5a6", "#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for i in range(min(weight_history.shape[1], len(labels))):
        ax4.fill_between(
            range(len(weight_history)),
            np.sum(weight_history[:, :i], axis=1) if i > 0 else 0,
            np.sum(weight_history[:, :i + 1], axis=1),
            alpha=0.7, label=labels[i],
            color=colors[i % len(colors)]
        )
    ax4.set_xlabel("Trading Day")
    ax4.set_ylabel("Weight")
    ax4.set_title("④ Portfolio Allocation (Last Episode)")
    ax4.legend(loc="upper right")
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_results.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # ── Actor/Critic Loss ──
    if agent.actor_losses and agent.critic_losses:
        fig2, (ax_a, ax_c) = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle("Training Losses", fontsize=13, fontweight="bold")

        ax_a.plot(agent.actor_losses, alpha=0.3, color="purple")
        if len(agent.actor_losses) >= 50:
            ma_a = np.convolve(agent.actor_losses, np.ones(50) / 50, mode="valid")
            ax_a.plot(range(49, len(agent.actor_losses)), ma_a,
                      color="darkviolet", linewidth=2)
        ax_a.set_title("Actor Loss")
        ax_a.set_xlabel("Update Step")
        ax_a.grid(True, alpha=0.3)

        ax_c.plot(agent.critic_losses, alpha=0.3, color="darkorange")
        if len(agent.critic_losses) >= 50:
            ma_c = np.convolve(agent.critic_losses, np.ones(50) / 50, mode="valid")
            ax_c.plot(range(49, len(agent.critic_losses)), ma_c,
                      color="red", linewidth=2)
        ax_c.set_title("Critic Loss")
        ax_c.set_xlabel("Update Step")
        ax_c.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "losses.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    agent, env, rewards, values = train()
