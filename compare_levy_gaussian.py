"""
Lévy vs Gaussian Noise Comparison
Phase 2에서 시연할 핵심 시각화

"이것이 암호화폐 시장에서 58.37% 수익을 낸 비결입니다"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from scipy.stats import levy_stable, norm
import os
from config import OUTPUT_DIR


def compare_noise_distributions():
    """Lévy vs Gaussian 노이즈 분포 비교"""

    np.random.seed(42)
    n_samples = 10000

    # ── 노이즈 생성 ──
    gaussian_samples = np.random.randn(n_samples)
    levy_samples_15 = levy_stable.rvs(alpha=1.5, beta=0, size=n_samples)
    levy_samples_15 = np.clip(levy_samples_15, -10, 10)
    levy_samples_18 = levy_stable.rvs(alpha=1.8, beta=0, size=n_samples)
    levy_samples_18 = np.clip(levy_samples_18, -10, 10)

    # ── 시각화 ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Lévy Process vs Gaussian: Why Fat-Tails Matter for Financial RL",
        fontsize=14, fontweight="bold"
    )

    # (1) 히스토그램 비교
    ax1 = axes[0, 0]
    ax1.hist(gaussian_samples, bins=100, alpha=0.5, density=True,
             color="steelblue", label="Gaussian (α=2)")
    ax1.hist(levy_samples_15, bins=100, alpha=0.5, density=True,
             color="crimson", label="Lévy (α=1.5)")
    ax1.set_xlim(-8, 8)
    ax1.set_title("① Distribution Comparison")
    ax1.set_xlabel("Noise Value")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.annotate("Fat Tail →\nOccasional Large Jumps",
                 xy=(4, 0.02), fontsize=10, color="crimson",
                 fontweight="bold")

    # (2) 로그 스케일 tail 비교
    ax2 = axes[0, 1]
    for data, label, color in [
        (gaussian_samples, "Gaussian (α=2)", "steelblue"),
        (levy_samples_18, "Lévy (α=1.8)", "orange"),
        (levy_samples_15, "Lévy (α=1.5)", "crimson"),
    ]:
        counts, edges = np.histogram(data, bins=200, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        mask = counts > 0
        ax2.semilogy(centers[mask], counts[mask], ".", markersize=3,
                     color=color, alpha=0.7, label=label)
    ax2.set_xlim(-8, 8)
    ax2.set_title("② Log-Scale Tail Comparison")
    ax2.set_xlabel("Noise Value")
    ax2.set_ylabel("Log Density")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.annotate("← Heavier tails = more exploration",
                 xy=(-7, 1e-3), fontsize=10, color="crimson")

    # (3) 시계열 비교 (탐색 궤적)
    ax3 = axes[1, 0]
    n_steps = 200
    gaussian_walk = np.cumsum(np.random.randn(n_steps) * 0.1)
    levy_walk = np.cumsum(
        np.clip(levy_stable.rvs(1.5, 0, size=n_steps), -5, 5) * 0.1
    )

    ax3.plot(gaussian_walk, color="steelblue", linewidth=1.5,
             label="Gaussian Walk", alpha=0.8)
    ax3.plot(levy_walk, color="crimson", linewidth=1.5,
             label="Lévy Walk (α=1.5)", alpha=0.8)

    # 점프 하이라이트
    levy_diff = np.abs(np.diff(levy_walk))
    jump_threshold = np.percentile(levy_diff, 95)
    jumps = np.where(levy_diff > jump_threshold)[0]
    for j in jumps[:5]:
        ax3.axvline(x=j, color="crimson", alpha=0.2, linewidth=3)

    ax3.set_title("③ Exploration Trajectory (Random Walk)")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Cumulative Position")
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.annotate("Jump!\n(Local Optima Escape)",
                 xy=(jumps[0] if len(jumps) > 0 else 50, levy_walk[jumps[0]] if len(jumps) > 0 else 0),
                 fontsize=10, color="crimson", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="crimson"))

    # (4) α 값에 따른 탐색 범위
    ax4 = axes[1, 1]
    alphas = [1.2, 1.5, 1.8, 2.0]
    colors = ["red", "crimson", "orange", "steelblue"]
    positions = []

    for i, (alpha, color) in enumerate(zip(alphas, colors)):
        if alpha == 2.0:
            samples = np.random.randn(1000)
        else:
            samples = levy_stable.rvs(alpha=alpha, beta=0, size=1000)
            samples = np.clip(samples, -10, 10)

        parts = ax4.violinplot([samples], positions=[i], showmeans=True,
                               showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

    ax4.set_xticks(range(len(alphas)))
    ax4.set_xticklabels([f"α={a}" for a in alphas])
    ax4.set_title("④ Exploration Range by α")
    ax4.set_ylabel("Noise Range")
    ax4.grid(True, alpha=0.3)
    ax4.annotate("α ↓ = More aggressive exploration\nα=2 = Standard Gaussian",
                 xy=(0.5, 0.02), xycoords="axes fraction", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "levy_vs_gaussian.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  → 저장: {OUTPUT_DIR}/levy_vs_gaussian.png")


def compare_action_generation():
    """SDE Actor에서 Lévy vs Gaussian 행동 분포 비교"""

    np.random.seed(42)
    n_samples = 500
    action_dim = 3  # Cash + BTC + ETH

    # 가상의 drift (결정적 방향)
    drift = np.array([0.2, 0.5, 0.3])  # BTC 선호

    # ── Gaussian 기반 행동 ──
    gaussian_actions = []
    for _ in range(n_samples):
        noise = np.random.randn(action_dim) * 0.3
        action = drift + noise
        action = np.exp(action) / np.sum(np.exp(action))  # softmax
        gaussian_actions.append(action)
    gaussian_actions = np.array(gaussian_actions)

    # ── Lévy 기반 행동 ──
    levy_actions = []
    for _ in range(n_samples):
        noise = levy_stable.rvs(1.5, 0, size=action_dim)
        noise = np.clip(noise, -5, 5) * 0.3
        action = drift + noise
        action = np.exp(action) / np.sum(np.exp(action))
        levy_actions.append(action)
    levy_actions = np.array(levy_actions)

    # ── 시각화 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Action Distribution: Lévy vs Gaussian Exploration",
        fontsize=14, fontweight="bold"
    )
    labels = ["Cash", "BTC", "ETH"]

    for i, (label, ax) in enumerate(zip(labels, axes)):
        ax.hist(gaussian_actions[:, i], bins=40, alpha=0.5, density=True,
                color="steelblue", label="Gaussian")
        ax.hist(levy_actions[:, i], bins=40, alpha=0.5, density=True,
                color="crimson", label="Lévy (α=1.5)")
        ax.axvline(x=np.mean(gaussian_actions[:, i]), color="steelblue",
                   linestyle="--", linewidth=2)
        ax.axvline(x=np.mean(levy_actions[:, i]), color="crimson",
                   linestyle="--", linewidth=2)
        ax.set_title(f"{label} Weight Distribution")
        ax.set_xlabel("Portfolio Weight")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "action_distribution.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  → 저장: {OUTPUT_DIR}/action_distribution.png")


if __name__ == "__main__":
    print("=" * 50)
    print("  Lévy vs Gaussian Comparison")
    print("=" * 50)
    print("\n[1/2] 노이즈 분포 비교...")
    compare_noise_distributions()
    print("\n[2/2] 행동 분포 비교...")
    compare_action_generation()
    print("\n✅ 완료!")
