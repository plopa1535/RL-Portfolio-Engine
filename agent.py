"""
SDELP-DDPG Agent
논문 Algorithm 1: DDPG + SDE Actor + OU Process + Experience Replay
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from config import *
from networks import SDEActor, Critic


# ═══════════════════════════════════════════════════════
# Ornstein-Uhlenbeck Process
# ═══════════════════════════════════════════════════════

class OUProcess:
    """
    Ornstein-Uhlenbeck Process (탐색 노이즈)

    dX_t = θ(μ - X_t)dt + σdW_t

    평균회귀 특성 → 모멘텀 있는 탐색
    금융에서 mean-reversion을 모사
    """

    def __init__(self, size, theta=OU_THETA, sigma=OU_SIGMA, mu=0.0):
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.mu = mu * np.ones(size)
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state.copy()


# ═══════════════════════════════════════════════════════
# Experience Replay Buffer
# ═══════════════════════════════════════════════════════

class ReplayBuffer:
    """경험 리플레이 버퍼"""

    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.FloatTensor(np.array(actions)).to(DEVICE),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════
# SDELP-DDPG Agent
# ═══════════════════════════════════════════════════════

class SDELPDDPGAgent:
    """
    SDELP-DDPG 에이전트 (논문 Algorithm 1)

    핵심 구조:
    - Actor (Online + Target): SDE 기반 정책 생성기
    - Critic (Online + Target): Q-value 평가기
    - OU Process: 추가 탐색 노이즈
    - Replay Buffer: 경험 저장 및 샘플링

    학습 알고리즘:
    1. Actor가 SDE 궤적으로 행동 생성
    2. OU 노이즈 추가 → 환경에서 실행
    3. (s, a, r, s') 경험 저장
    4. 미니배치 샘플링
    5. Critic 업데이트: MSE(Q(s,a), r + γQ'(s', π'(s')))
    6. Actor 업데이트: -E[Q(s, π(s))]
    7. Target 네트워크 소프트 업데이트
    """

    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # ── Actor (Online + Target) ──
        self.actor = SDEActor(state_dim, action_dim).to(DEVICE)
        self.actor_target = SDEActor(state_dim, action_dim).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()  # Target 네트워크는 항상 eval
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)

        # ── Critic (Online + Target) ──
        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()  # Target 네트워크는 항상 eval
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        # ── Replay Buffer & OU Process ──
        self.replay_buffer = ReplayBuffer()
        self.ou_process = OUProcess(action_dim)

        # ── 학습 통계 ──
        self.actor_losses = []
        self.critic_losses = []

    def select_action(self, state, explore=True):
        """
        행동 선택

        Args:
            state: numpy array (state_dim,)
            explore: True면 OU 노이즈 추가

        Returns:
            action: numpy array (action_dim,) — softmax 정규화된 가중치
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        self.actor.eval()  # BatchNorm eval 모드 (batch=1 대응)
        with torch.no_grad():
            action = self.actor(state_tensor, deterministic=not explore)
            action = action.cpu().numpy()[0]
        self.actor.train()  # 학습 모드 복원

        if explore:
            # OU Process 노이즈 추가
            noise = self.ou_process.sample() * 0.1  # 스케일 조절
            action = action + noise
            # 음수 방지 후 재정규화
            action = np.clip(action, 0.0, None)
            action_sum = action.sum()
            if action_sum > 0:
                action = action / action_sum
            else:
                action = np.ones(self.action_dim) / self.action_dim

        return action

    def update(self):
        """
        DDPG 업데이트 (Algorithm 1의 핵심)

        Returns:
            critic_loss, actor_loss
        """
        if len(self.replay_buffer) < BATCH_SIZE:
            return None, None

        # ── 미니배치 샘플링 ──
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample()

        # ── Critic 업데이트 ──
        # Target Q-value: y = r + γ × Q'(s', π'(s'))
        with torch.no_grad():
            next_actions = self.actor_target(next_states, deterministic=True)
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + GAMMA * target_q * (1 - dones)

        # Current Q-value
        current_q = self.critic(states, actions)

        # Critic Loss: MSE
        critic_loss = nn.MSELoss()(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ── Actor 업데이트 ──
        # Policy Gradient: ∇θ J ≈ -E[Q(s, π_θ(s))]
        predicted_actions = self.actor(states, deterministic=True)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ── Target 네트워크 소프트 업데이트 ──
        # θ' ← τθ + (1-τ)θ'
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        # 통계 기록
        self.critic_losses.append(critic_loss.item())
        self.actor_losses.append(actor_loss.item())

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, source, target):
        """소프트 업데이트: θ' ← τθ + (1-τ)θ'"""
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(TAU * src_param.data + (1 - TAU) * tgt_param.data)

    def save(self, path):
        """모델 저장"""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
