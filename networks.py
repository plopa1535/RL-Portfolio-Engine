"""
SDELP-DDPG Networks (논문 구조 충실 버전)
논문 Section 3.3-3.4: Conv2D Actor + Residual Critic

주요 변경사항 (vs 간소화 버전):
  1. Actor InitNet: Conv2D 기반 상태 처리
  2. DriftNet: a_k만 입력 (논문 수식 준수) + LayerNorm
  3. DiffusionNet: BatchNorm + Dropout 추가
  4. SDE 노이즈 스케일링: (Δt)^{1/α} (논문 Eq. 8)
  5. Critic: Conv2D + Residual Block × 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import levy_stable
from config import *


# ═══════════════════════════════════════════════════════
# Lévy / Gaussian Noise Generators
# ═══════════════════════════════════════════════════════

class LevyNoiseGenerator:
    """α-stable Lévy 노이즈 생성기"""

    def __init__(self, alpha=LEVY_ALPHA, beta=LEVY_BETA_PARAM):
        self.alpha = alpha
        self.beta = beta

    def sample(self, shape):
        samples = levy_stable.rvs(
            alpha=self.alpha, beta=self.beta, size=shape
        )
        samples = np.clip(samples, -5.0, 5.0)
        return torch.FloatTensor(samples).to(DEVICE)


class GaussianNoiseGenerator:
    def sample(self, shape):
        return torch.randn(shape).to(DEVICE)


# ═══════════════════════════════════════════════════════
# Conv2D Feature Extractor (상태 처리용)
# ═══════════════════════════════════════════════════════

class ConvFeatureExtractor(nn.Module):
    """
    논문: Conv2D로 가격 텐서에서 특성 추출
    입력: (batch, 1, n_assets, window)
    출력: (batch, out_dim)
    """

    def __init__(self, n_assets, window, out_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((n_assets, 1))
        self.fc = nn.Sequential(
            nn.Linear(64 * n_assets, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ═══════════════════════════════════════════════════════
# Drift-Net & Diffusion-Net
# ═══════════════════════════════════════════════════════

class DriftNet(nn.Module):
    """
    Drift 네트워크: u(a_k)
    논문: a_k만 입력 (state가 아님!)
    Conv2D → BatchNorm → ReLU → Flatten (논문)
    a_k가 저차원이므로 FC+LayerNorm 사용 (BatchNorm은 batch=1에서 에러)
    """

    def __init__(self, action_dim, hidden=DRIFT_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )

    def forward(self, action):
        return self.net(action)


class DiffusionNet(nn.Module):
    """
    Diffusion 네트워크: ϑ(a_1)
    논문: BatchNorm → ReLU → Dropout → Dense
    a_1(초기 행동)만 입력 — 고정값으로 SDE 발산 방지
    """

    def __init__(self, action_dim, hidden=DIFFUSION_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, action_dim),
            nn.Softplus()
        )

    def forward(self, action_first):
        return self.net(action_first)


# ═══════════════════════════════════════════════════════
# SDE Actor (Conv2D + SDE + Attention) — 논문 핵심
# ═══════════════════════════════════════════════════════

class SDEActor(nn.Module):
    """
    논문 구조 SDE 기반 Actor:
      1. Conv2D로 상태에서 초기 행동 추출 (a_0)
      2. SDE 루프: a_{k+1} = a_k + u(a_k)·Δt + ϑ(a_1)·(Δt)^{1/α}·L_k
      3. Attention으로 최적 행동 선택
      4. Softmax → 포트폴리오 가중치
    """

    def __init__(self, state_dim, action_dim, sde_steps=SDE_STEPS):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_assets = action_dim - 1  # action_dim = total_assets = n_assets + 1 (cash)
        self.window = WINDOW_SIZE
        self.sde_steps = sde_steps
        self.dt = 1.0 / sde_steps
        self.levy_alpha = LEVY_ALPHA

        # ── Conv2D 상태 특성 추출기 ──
        self.conv_extractor = ConvFeatureExtractor(
            self.n_assets, self.window, out_dim=128
        )

        # ── 초기 행동 생성기 (a_0): Conv features + weights → action ──
        self.init_net = nn.Sequential(
            nn.Linear(128 + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        # ── Drift & Diffusion (논문: a_k만 입력!) ──
        self.drift_net = DriftNet(action_dim)
        self.diffusion_net = DiffusionNet(action_dim)

        # ── Lévy Noise ──
        self.levy_noise = LevyNoiseGenerator()

        # ── Attention: K개 후보 행동 중 최적 선택 ──
        self.attention = nn.MultiheadAttention(
            embed_dim=action_dim, num_heads=1, batch_first=True
        )
        self.attention_proj = nn.Linear(action_dim, action_dim)

    def _split_state(self, state):
        """state를 price tensor와 portfolio weights로 분리"""
        batch = state.shape[0]
        price_flat = state[:, :self.n_assets * self.window]
        weights = state[:, self.n_assets * self.window:]

        # Conv2D 용: (batch, 1, n_assets, window)
        price_tensor = price_flat.view(batch, self.n_assets, self.window)
        price_tensor = price_tensor.unsqueeze(1)

        return price_tensor, weights

    def forward(self, state, deterministic=False):
        batch_size = state.shape[0]

        # Step 1: Conv2D 상태 특성 추출
        price_tensor, weights = self._split_state(state)
        features = self.conv_extractor(price_tensor)  # (batch, 128)

        # Step 2: 초기 행동 (a_0)
        init_input = torch.cat([features, weights], dim=-1)
        a_k = self.init_net(init_input)

        # Step 3: SDE 궤적 — K 스텝 Euler-Maruyama
        trajectory = [a_k]
        a_first = a_k  # Diffusion용 고정값

        for k in range(self.sde_steps):
            # Drift: u(a_k) · Δt (논문: a_k만!)
            drift = self.drift_net(a_k) * self.dt

            if deterministic:
                a_k = a_k + drift
            else:
                # Diffusion: ϑ(a_1) · (Δt)^{1/α} · L_k  (논문 Eq. 8)
                sigma = self.diffusion_net(a_first)
                levy_noise = self.levy_noise.sample((batch_size, self.action_dim))
                noise_scale = self.dt ** (1.0 / self.levy_alpha)
                diffusion = sigma * noise_scale * levy_noise

                a_k = a_k + drift + diffusion

            trajectory.append(a_k)

        # Step 4: Attention — 최적 행동 선택
        traj_tensor = torch.stack(trajectory, dim=1)  # (batch, K+1, action_dim)
        attn_out, _ = self.attention(traj_tensor, traj_tensor, traj_tensor)
        selected = attn_out[:, -1, :]
        selected = self.attention_proj(selected)

        # Step 5: Softmax → 유효 포트폴리오 가중치
        action = F.softmax(selected, dim=-1)

        return action

    def get_trajectory_for_viz(self, state):
        """시각화용 SDE 궤적"""
        batch_size = state.shape[0]
        price_tensor, weights = self._split_state(state)
        features = self.conv_extractor(price_tensor)
        init_input = torch.cat([features, weights], dim=-1)
        a_k = self.init_net(init_input)
        a_first = a_k

        trajectory = [a_k.detach().cpu().numpy()]

        for k in range(self.sde_steps):
            drift = self.drift_net(a_k) * self.dt
            sigma = self.diffusion_net(a_first)
            levy_noise = self.levy_noise.sample((batch_size, self.action_dim))
            noise_scale = self.dt ** (1.0 / self.levy_alpha)
            diffusion = sigma * noise_scale * levy_noise
            a_k = a_k + drift + diffusion
            trajectory.append(a_k.detach().cpu().numpy())

        return np.array(trajectory)


# ═══════════════════════════════════════════════════════
# Residual Block (Critic용)
# ═══════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """논문: Conv2D → ReLU → Conv2D → Add(Skip) → ReLU"""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(self.block(x) + x)  # Skip connection


# ═══════════════════════════════════════════════════════
# Critic Network (Conv2D + Residual Blocks)
# ═══════════════════════════════════════════════════════

class Critic(nn.Module):
    """
    논문 Critic: 다층 합성곱 잔차 신경망
      State 경로: Conv2D + ReLU → ResBlock × p → Conv2D → Flatten → Dense
      Action 경로: Dense
      Merge: State + Action → Dense → Q-value
    """

    def __init__(self, state_dim, action_dim, hidden=CRITIC_HIDDEN,
                 num_res_blocks=3):
        super().__init__()
        self.n_assets = action_dim - 1  # action_dim = total_assets = n_assets + 1 (cash)
        self.window = WINDOW_SIZE

        # ── State 경로: Conv2D + Residual ──
        self.state_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(32) for _ in range(num_res_blocks)]
        )
        self.state_conv_out = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.state_pool = nn.AdaptiveAvgPool2d((self.n_assets, 1))
        # state features: 64*n_assets + action_dim (weights)
        self.state_fc = nn.Linear(64 * self.n_assets + action_dim, hidden)

        # ── Action 경로 ──
        self.action_fc = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.ReLU(),
        )

        # ── Merge → Q-value ──
        self.merge = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        batch = state.shape[0]

        # State 경로
        price_flat = state[:, :self.n_assets * self.window]
        weights = state[:, self.n_assets * self.window:]
        price_tensor = price_flat.view(batch, self.n_assets, self.window).unsqueeze(1)

        s = self.state_conv(price_tensor)
        s = self.res_blocks(s)
        s = self.state_conv_out(s)
        s = self.state_pool(s)
        s = s.view(batch, -1)
        s = torch.cat([s, weights], dim=-1)
        s = F.relu(self.state_fc(s))

        # Action 경로
        a = self.action_fc(action)

        # Merge
        x = torch.cat([s, a], dim=-1)
        return self.merge(x)
