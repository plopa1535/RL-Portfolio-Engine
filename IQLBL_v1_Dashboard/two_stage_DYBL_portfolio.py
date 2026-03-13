"""
Two-Stage IQL + Dynamic Black-Litterman Portfolio Optimization (v2)

Stage 1: 개별 종목 IQL 학습 (매수/매도 신호 생성)
Stage 2: Dynamic Black-Litterman 모델을 사용한 포트폴리오 비중 최적화

핵심 개선사항 v2:
1. Adaptive View Confidence: 시장 변동성에 따른 동적 조정
2. Proactive Regime Detection: 선행 지표 기반 조기 레짐 감지
3. Drawdown Protection: 손실 제한 및 현금화 메커니즘
4. Momentum Reversal Detection: 추세 반전 조기 감지
5. Dynamic Position Sizing: 변동성 기반 포지션 크기 조정

Usage:
    python two_stage_DYBL_portfolio.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import os
import yaml
import warnings
from copy import deepcopy
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Configuration
# ============================================================

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_default_config():
    return {
        'data': {
            'train_end_date': '2024-12-31',
            'test_start_date': '2025-01-01'
        },
        'environment': {
            'window_size': 90
        },
        'iql': {
            'gamma': 0.99,
            'tau': 0.005,
            'expectile': 0.8,
            'temperature': 3.0,
            'lr': 0.0003,
            'hidden_dims': [256, 256]
        },
        'training': {
            'n_epochs': 100,
            'batch_size': 256,
            'behavior_policies': ['oracle', 'momentum', 'reversal', 'random'],
            'learning_window': 30
        },
        'portfolio': {
            'initial_value': 10000,
            'fee_rate': 0.001,
            'risk_weight': 0.5,
            'turnover_penalty': 0.01
        },
        'black_litterman': {
            'tau': 0.05,
            'risk_aversion': 2.5,
            'view_confidence': 0.6,  # 기본 view_confidence (동적 조정의 중심값)
            'lookback': 60
        },
        'dynamic_bl': {
            'vol_lookback': 20,           # 변동성 계산 기간
            'vol_threshold_high': 0.04,   # 고변동성 임계값 (일별 4%)
            'vol_threshold_low': 0.015,   # 저변동성 임계값 (일별 1.5%)
            'confidence_min': 0.2,        # 최소 view_confidence
            'confidence_max': 0.8,        # 최대 view_confidence
            'trend_lookback': 10,         # 추세 감지 기간
            'trend_threshold': 0.05,      # 추세 판단 임계값 (5%)
            'regime_weight': 0.3,         # 레짐 감지의 신뢰도 조정 가중치
            'momentum_decay': 0.5,        # 모멘텀 신호 약화 시 감쇠
            # v2 추가 파라미터
            'drawdown_threshold': 0.10,   # 드로우다운 방어 임계값 (10%)
            'cash_ratio_max': 0.5,        # 최대 현금 비율 (50%)
            'reversal_lookback': 5,       # 반전 감지 기간
            'reversal_threshold': 0.03,   # 반전 감지 임계값 (3%)
            'vol_scaling': True,          # 변동성 기반 포지션 조정
            'fast_ma': 5,                 # 빠른 이동평균
            'slow_ma': 20,                # 느린 이동평균
            'rsi_period': 14,             # RSI 기간
            'rsi_overbought': 70,         # RSI 과매수
            'rsi_oversold': 30            # RSI 과매도
        },
        'seed': 42,
        'gpu': {'use_cuda': True}
    }

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = None

def setup_device(use_cuda=True):
    global device
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

# ============================================================
# Data Loading
# ============================================================

def load_data(filepath):
    df = pd.read_excel(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df['return'] = df['close'].pct_change().fillna(0).clip(-0.5, 0.5)
    return df

def load_all_coin_data(folder_path, coins):
    """모든 코인 데이터 로드"""
    all_data = {}
    for coin in coins:
        filepath = os.path.join(folder_path, f"Binance_{coin}_futures.xlsx")
        if os.path.exists(filepath):
            all_data[coin] = load_data(filepath)
            print(f"  {coin}: {len(all_data[coin])} days loaded")
        else:
            print(f"  {coin}: File not found!")
    return all_data

# ============================================================
# Stage 1: Individual Coin IQL
# ============================================================

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        input_dim = state_dim + action_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        layers.extend([nn.Linear(input_dim, action_dim), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)

class Stage1IQLAgent:
    """Stage 1: 개별 종목 IQL 에이전트"""
    def __init__(self, state_dim, action_dim=1, config=None):
        global device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)
        self.expectile = self.config.get('expectile', 0.8)
        self.temperature = self.config.get('temperature', 3.0)
        self.lr = self.config.get('lr', 3e-4)
        self.hidden_dims = self.config.get('hidden_dims', [256, 256])
        self._build_networks()

    def _build_networks(self):
        global device
        self.value_net = ValueNetwork(self.state_dim, self.hidden_dims).to(device)
        self.q_net1 = QNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(device)
        self.q_net2 = QNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(device)
        self.q_target1 = QNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(device)
        self.q_target2 = QNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(device)
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dims).to(device)

        self.q_target1.load_state_dict(self.q_net1.state_dict())
        self.q_target2.load_state_dict(self.q_net2.state_dict())

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.q_optimizer = optim.Adam(list(self.q_net1.parameters()) + list(self.q_net2.parameters()), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def reset_networks(self):
        self._build_networks()

    def expectile_loss(self, diff, expectile):
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return (weight * (diff ** 2)).mean()

    def update(self, batch):
        global device
        states = torch.FloatTensor(batch['states']).to(device)
        actions = torch.FloatTensor(batch['actions']).to(device)
        rewards = torch.FloatTensor(batch['rewards']).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(batch['next_states']).to(device)
        dones = torch.FloatTensor(batch['dones']).unsqueeze(1).to(device)

        with torch.no_grad():
            q_min = torch.min(self.q_target1(states, actions), self.q_target2(states, actions))
        v = self.value_net(states)
        value_loss = self.expectile_loss(q_min - v, self.expectile)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        with torch.no_grad():
            target_q = rewards + self.gamma * (1 - dones) * self.value_net(next_states)
        q1, q2 = self.q_net1(states, actions), self.q_net2(states, actions)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            advantage = torch.min(self.q_net1(states, actions), self.q_net2(states, actions)) - self.value_net(states)
            weights = torch.clamp(torch.exp(self.temperature * advantage), max=100.0)
        policy_loss = (weights * ((self.policy_net(states) - actions) ** 2)).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for param, target_param in zip(self.q_net1.parameters(), self.q_target1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q_net2.parameters(), self.q_target2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def select_action(self, state, deterministic=True):
        global device
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.policy_net(state)
        return action.cpu().numpy().flatten()[0]

def create_stage1_dataset(df, config, end_idx):
    """Stage 1용 오프라인 데이터셋 생성"""
    window_size = config['environment']['window_size']
    behavior_policies = config['training'].get('behavior_policies', ['oracle', 'momentum', 'reversal', 'random'])

    states, actions, rewards, next_states, dones = [], [], [], [], []
    returns = df['return'].values

    for policy in behavior_policies:
        for i in range(window_size, end_idx - 1):
            state = returns[i - window_size:i].copy()
            next_state = returns[i - window_size + 1:i + 1].copy()
            future_return = returns[i]

            if policy == 'oracle':
                action = (1.0 if future_return > 0 else -1.0) + np.random.normal(0, 0.1)
            elif policy == 'momentum':
                recent = np.mean(state[-5:])
                action = (np.sign(recent) if abs(recent) > 0.005 else np.random.choice([-1, 1])) + np.random.normal(0, 0.2)
            elif policy == 'reversal':
                recent = np.mean(state[-5:])
                action = (-np.sign(recent) if abs(recent) > 0.005 else np.random.choice([-1, 1])) + np.random.normal(0, 0.2)
            else:
                action = np.random.uniform(-1, 1)

            action = np.clip(action, -1, 1)
            reward = 1.0 if (action > 0 and future_return > 0) or (action < 0 and future_return < 0) else 0.0

            states.append(state)
            actions.append([action])
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(0.0)

    return {
        'states': np.array(states) if states else np.array([]).reshape(0, window_size),
        'actions': np.array(actions) if actions else np.array([]).reshape(0, 1),
        'rewards': np.array(rewards),
        'next_states': np.array(next_states) if next_states else np.array([]).reshape(0, window_size),
        'dones': np.array(dones)
    }

def train_stage1_agent(agent, dataset, config):
    """Stage 1 에이전트 학습"""
    n_epochs = config['training']['n_epochs']
    batch_size = config['training']['batch_size']
    n_samples = len(dataset['states'])

    if n_samples == 0:
        return agent

    batch_size = min(batch_size, max(1, n_samples // 2))
    n_batches = max(1, n_samples // batch_size)

    for _ in range(n_epochs):
        indices = np.random.permutation(n_samples)
        for i in range(n_batches):
            batch_idx = indices[i * batch_size:(i + 1) * batch_size]
            batch = {k: v[batch_idx] for k, v in dataset.items()}
            agent.update(batch)

    return agent

# ============================================================
# Stage 2: Dynamic Black-Litterman Model with Adaptive Confidence
# ============================================================

class DynamicBlackLittermanOptimizer:
    """
    Dynamic Black-Litterman 포트폴리오 최적화 (v2)

    핵심 기능:
    1. Adaptive View Confidence: 시장 변동성에 따른 동적 조정
    2. Proactive Regime Detection: 선행 지표 기반 조기 레짐 감지
    3. Drawdown Protection: 손실 제한 및 현금화 메커니즘
    4. Momentum Reversal Detection: 추세 반전 조기 감지
    5. Dynamic Position Sizing: 변동성 기반 포지션 크기 조정
    """

    def __init__(self, n_assets, config):
        self.n_assets = n_assets
        self.config = config

        # Black-Litterman 기본 파라미터
        bl_config = config.get('black_litterman', {})
        self.tau = bl_config.get('tau', 0.05)
        self.risk_aversion = bl_config.get('risk_aversion', 2.5)
        self.base_view_confidence = bl_config.get('view_confidence', 0.6)

        # Dynamic BL 파라미터
        dyn_config = config.get('dynamic_bl', {})
        self.vol_lookback = dyn_config.get('vol_lookback', 20)
        self.vol_threshold_high = dyn_config.get('vol_threshold_high', 0.04)
        self.vol_threshold_low = dyn_config.get('vol_threshold_low', 0.015)
        self.confidence_min = dyn_config.get('confidence_min', 0.2)
        self.confidence_max = dyn_config.get('confidence_max', 0.8)
        self.trend_lookback = dyn_config.get('trend_lookback', 10)
        self.trend_threshold = dyn_config.get('trend_threshold', 0.05)
        self.regime_weight = dyn_config.get('regime_weight', 0.3)
        self.momentum_decay = dyn_config.get('momentum_decay', 0.5)

        # v2 추가 파라미터
        self.drawdown_threshold = dyn_config.get('drawdown_threshold', 0.10)
        self.cash_ratio_max = dyn_config.get('cash_ratio_max', 0.5)
        self.reversal_lookback = dyn_config.get('reversal_lookback', 5)
        self.reversal_threshold = dyn_config.get('reversal_threshold', 0.03)
        self.vol_scaling = dyn_config.get('vol_scaling', True)
        self.fast_ma = dyn_config.get('fast_ma', 5)
        self.slow_ma = dyn_config.get('slow_ma', 20)
        self.rsi_period = dyn_config.get('rsi_period', 14)
        self.rsi_overbought = dyn_config.get('rsi_overbought', 70)
        self.rsi_oversold = dyn_config.get('rsi_oversold', 30)

        # Long/Short 허용 여부
        self.allow_short = config.get('portfolio', {}).get('allow_short', True)

        # 상태 추적
        self.current_regime = 'neutral'
        self.current_view_confidence = self.base_view_confidence
        self.confidence_history = []
        self.peak_value = None
        self.current_drawdown = 0.0
        self.defensive_mode = False
        self.reversal_warning = False

    def calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        if len(prices) < period + 1:
            return 50.0  # 중립값

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_moving_averages(self, returns):
        """이동평균 계산 및 크로스오버 감지"""
        if len(returns) < self.slow_ma:
            return 0.0, False, False

        prices = np.cumprod(1 + returns)

        fast_ma = np.mean(prices[-self.fast_ma:])
        slow_ma = np.mean(prices[-self.slow_ma:])

        # 골든 크로스 / 데드 크로스 감지
        if len(prices) >= self.slow_ma + 1:
            prev_fast = np.mean(prices[-self.fast_ma-1:-1])
            prev_slow = np.mean(prices[-self.slow_ma-1:-1])

            golden_cross = prev_fast <= prev_slow and fast_ma > slow_ma
            death_cross = prev_fast >= prev_slow and fast_ma < slow_ma
        else:
            golden_cross = False
            death_cross = False

        # MA 차이 비율
        ma_diff = (fast_ma - slow_ma) / slow_ma if slow_ma != 0 else 0

        return ma_diff, golden_cross, death_cross

    def detect_reversal_signal(self, historical_returns):
        """
        추세 반전 조기 감지

        - 급격한 방향 전환 감지
        - 고점/저점 대비 급변 감지
        """
        if len(historical_returns) < self.reversal_lookback + 5:
            return False, 'none'

        portfolio_returns = np.mean(historical_returns, axis=1)

        # 최근 수익률
        recent = portfolio_returns[-self.reversal_lookback:]
        prior = portfolio_returns[-self.reversal_lookback*2:-self.reversal_lookback]

        recent_return = np.prod(1 + recent) - 1
        prior_return = np.prod(1 + prior) - 1

        # 반전 감지: 이전 기간과 최근 기간의 방향이 반대
        reversal_detected = False
        reversal_type = 'none'

        if prior_return > self.reversal_threshold and recent_return < -self.reversal_threshold * 0.5:
            # 상승 후 하락 반전
            reversal_detected = True
            reversal_type = 'bearish_reversal'
        elif prior_return < -self.reversal_threshold and recent_return > self.reversal_threshold * 0.5:
            # 하락 후 상승 반전
            reversal_detected = True
            reversal_type = 'bullish_reversal'

        return reversal_detected, reversal_type

    def detect_market_regime(self, historical_returns):
        """
        시장 레짐 감지 (v2 - 선행 지표 포함)

        Args:
            historical_returns: 과거 수익률 (lookback x n_assets)

        Returns:
            regime: 'bullish', 'bearish', 'neutral', 'reversal_warning'
            trend_strength: 추세 강도 (0~1)
            extra_info: 추가 정보
        """
        extra_info = {}

        if len(historical_returns) < max(self.trend_lookback, self.slow_ma):
            return 'neutral', 0.0, extra_info

        # 포트폴리오 평균 수익률
        portfolio_returns = np.mean(historical_returns, axis=1)
        recent_returns = portfolio_returns[-self.trend_lookback:]

        # 누적 수익률
        cumulative_return = np.prod(1 + recent_returns) - 1
        trend_strength = min(abs(cumulative_return) / self.trend_threshold, 1.0)

        # 이동평균 분석
        ma_diff, golden_cross, death_cross = self.calculate_moving_averages(portfolio_returns)
        extra_info['ma_diff'] = ma_diff
        extra_info['golden_cross'] = golden_cross
        extra_info['death_cross'] = death_cross

        # RSI 분석
        prices = np.cumprod(1 + portfolio_returns)
        rsi = self.calculate_rsi(prices, self.rsi_period)
        extra_info['rsi'] = rsi

        # 반전 신호 감지
        reversal_detected, reversal_type = self.detect_reversal_signal(historical_returns)
        extra_info['reversal_detected'] = reversal_detected
        extra_info['reversal_type'] = reversal_type

        # 레짐 판단 (선행 지표 반영)
        regime = 'neutral'

        # 1. 반전 경고가 최우선
        if reversal_detected:
            if reversal_type == 'bearish_reversal':
                regime = 'reversal_warning_down'
                trend_strength = 0.8
            elif reversal_type == 'bullish_reversal':
                regime = 'reversal_warning_up'
                trend_strength = 0.8
        # 2. 데드 크로스 감지
        elif death_cross:
            regime = 'bearish'
            trend_strength = max(trend_strength, 0.6)
        # 3. RSI 과매수/과매도
        elif rsi > self.rsi_overbought and cumulative_return > 0:
            regime = 'overbought'  # 곧 하락 예상
            trend_strength = 0.7
        elif rsi < self.rsi_oversold and cumulative_return < 0:
            regime = 'oversold'  # 곧 반등 예상
            trend_strength = 0.7
        # 4. 기본 추세 판단
        elif cumulative_return > self.trend_threshold:
            regime = 'bullish'
        elif cumulative_return < -self.trend_threshold:
            regime = 'bearish'

        return regime, trend_strength, extra_info

    def update_drawdown_state(self, current_value):
        """드로우다운 상태 업데이트"""
        if self.peak_value is None:
            self.peak_value = current_value

        if current_value > self.peak_value:
            self.peak_value = current_value

        self.current_drawdown = (self.peak_value - current_value) / self.peak_value

        # 방어 모드 전환
        if self.current_drawdown > self.drawdown_threshold:
            self.defensive_mode = True
        elif self.current_drawdown < self.drawdown_threshold * 0.5:
            # 드로우다운이 절반 이하로 회복되면 방어 모드 해제
            self.defensive_mode = False

        return self.current_drawdown, self.defensive_mode

    def calculate_adaptive_confidence(self, historical_returns, stage1_signals):
        """
        Adaptive View Confidence 계산 (v2 - 선행 지표 포함)

        핵심 로직:
        1. 변동성 기반 조정: 고변동성 → 낮은 신뢰도
        2. 레짐 기반 조정: 하락장에서 모멘텀 신호 약화
        3. 신호 일관성 조정: 신호가 일관될수록 높은 신뢰도
        4. 선행 지표 반영: RSI, MA 크로스오버, 반전 신호

        Args:
            historical_returns: 과거 수익률 (lookback x n_assets)
            stage1_signals: Stage 1 IQL 신호 (n_assets,)

        Returns:
            adaptive_confidence: 조정된 view_confidence (0~1)
            adjustment_info: 조정 정보 딕셔너리
        """
        adjustment_info = {
            'base_confidence': self.base_view_confidence,
            'volatility_adjustment': 0.0,
            'regime_adjustment': 0.0,
            'signal_consistency_adjustment': 0.0,
            'leading_indicator_adjustment': 0.0,
            'final_confidence': self.base_view_confidence
        }

        if len(historical_returns) < self.vol_lookback:
            return self.base_view_confidence, adjustment_info

        # 1. 변동성 기반 조정
        recent_returns = historical_returns[-self.vol_lookback:]
        portfolio_vol = np.std(np.mean(recent_returns, axis=1))

        # 변동성에 따른 신뢰도 조정
        if portfolio_vol > self.vol_threshold_high:
            # 고변동성: 신뢰도 감소 (시장 균형에 더 의존)
            vol_adjustment = -0.2 * (portfolio_vol - self.vol_threshold_high) / self.vol_threshold_high
        elif portfolio_vol < self.vol_threshold_low:
            # 저변동성: 신뢰도 증가 (IQL 신호에 더 의존)
            vol_adjustment = 0.1 * (self.vol_threshold_low - portfolio_vol) / self.vol_threshold_low
        else:
            vol_adjustment = 0.0

        adjustment_info['volatility_adjustment'] = vol_adjustment

        # 2. 레짐 기반 조정 (v2 - 선행 지표 포함)
        regime, trend_strength, extra_info = self.detect_market_regime(historical_returns)
        self.current_regime = regime

        # 하락장에서 Long 신호의 신뢰도 감소
        # 상승장에서 Short 신호의 신뢰도 감소
        avg_signal = np.mean(stage1_signals)

        # 기본 레짐 조정
        if regime == 'bearish' and avg_signal > 0:
            # 하락장인데 Long 신호 → 신뢰도 대폭 감소
            regime_adjustment = -self.regime_weight * trend_strength * 1.5
        elif regime == 'bullish' and avg_signal < 0:
            # 상승장인데 Short 신호 → 신뢰도 감소
            regime_adjustment = -self.regime_weight * trend_strength * 0.5
        elif regime == 'bearish' and avg_signal < 0:
            # 하락장에서 Short 신호 → 신뢰도 증가
            regime_adjustment = self.regime_weight * trend_strength * 0.7
        elif regime == 'bullish' and avg_signal > 0:
            # 상승장에서 Long 신호 → 신뢰도 증가
            regime_adjustment = self.regime_weight * trend_strength * 0.3
        else:
            regime_adjustment = 0.0

        adjustment_info['regime_adjustment'] = regime_adjustment

        # 3. 선행 지표 기반 추가 조정 (v2 핵심 기능)
        leading_adjustment = 0.0

        # RSI 기반 조정
        rsi = extra_info.get('rsi', 50)
        if rsi > self.rsi_overbought and avg_signal > 0:
            # RSI 과매수인데 Long 신호 → 신뢰도 대폭 감소 (하락 임박)
            leading_adjustment -= 0.25
            self.reversal_warning = True
        elif rsi < self.rsi_oversold and avg_signal < 0:
            # RSI 과매도인데 Short 신호 → 신뢰도 감소 (반등 임박)
            leading_adjustment -= 0.15
            self.reversal_warning = True
        else:
            self.reversal_warning = False

        # 데드 크로스 감지 시 Long 신호 신뢰도 감소
        if extra_info.get('death_cross', False) and avg_signal > 0:
            leading_adjustment -= 0.2
            self.reversal_warning = True

        # 골든 크로스 감지 시 Short 신호 신뢰도 감소
        if extra_info.get('golden_cross', False) and avg_signal < 0:
            leading_adjustment -= 0.1

        # 반전 감지 시 강력한 조정
        if extra_info.get('reversal_detected', False):
            reversal_type = extra_info.get('reversal_type', 'none')
            if reversal_type == 'bearish_reversal' and avg_signal > 0:
                # 약세 반전인데 Long 신호 → 신뢰도 대폭 감소
                leading_adjustment -= 0.3
                self.reversal_warning = True
            elif reversal_type == 'bullish_reversal' and avg_signal < 0:
                # 강세 반전인데 Short 신호 → 신뢰도 감소
                leading_adjustment -= 0.15

        # 과매수/과매도 레짐 추가 처리
        if regime == 'overbought' and avg_signal > 0:
            leading_adjustment -= 0.2
        elif regime == 'oversold' and avg_signal < 0:
            leading_adjustment -= 0.1

        # 반전 경고 레짐
        if regime == 'reversal_warning_down' and avg_signal > 0:
            leading_adjustment -= 0.25
        elif regime == 'reversal_warning_up' and avg_signal < 0:
            leading_adjustment -= 0.15

        adjustment_info['leading_indicator_adjustment'] = leading_adjustment
        adjustment_info['rsi'] = rsi
        adjustment_info['death_cross'] = extra_info.get('death_cross', False)
        adjustment_info['golden_cross'] = extra_info.get('golden_cross', False)
        adjustment_info['reversal_detected'] = extra_info.get('reversal_detected', False)
        adjustment_info['reversal_type'] = extra_info.get('reversal_type', 'none')

        # 4. 신호 일관성 조정
        # 모든 코인의 신호가 같은 방향이면 신뢰도 증가
        signal_directions = np.sign(stage1_signals)
        signal_consistency = abs(np.mean(signal_directions))  # 0~1

        consistency_adjustment = 0.1 * (signal_consistency - 0.5)  # -0.05 ~ +0.05
        adjustment_info['signal_consistency_adjustment'] = consistency_adjustment

        # 최종 신뢰도 계산 (선행 지표 조정 포함)
        adaptive_confidence = (self.base_view_confidence + vol_adjustment +
                               regime_adjustment + consistency_adjustment + leading_adjustment)
        adaptive_confidence = np.clip(adaptive_confidence, self.confidence_min, self.confidence_max)

        adjustment_info['final_confidence'] = adaptive_confidence
        adjustment_info['regime'] = regime
        adjustment_info['trend_strength'] = trend_strength
        adjustment_info['volatility'] = portfolio_vol

        self.current_view_confidence = adaptive_confidence
        self.confidence_history.append(adaptive_confidence)

        return adaptive_confidence, adjustment_info

    def adjust_signals_by_regime(self, stage1_signals, regime, trend_strength, extra_info=None):
        """
        레짐에 따른 신호 조정 (v2 - 선행 지표 포함)

        하락장에서 모멘텀 기반 Long 신호를 약화시키고,
        반전 가능성을 고려하여 신호를 조정.
        선행 지표(RSI, MA 크로스오버)에 따라 조기 조정 적용.
        """
        adjusted_signals = stage1_signals.copy()
        extra_info = extra_info or {}

        # 기본 레짐 조정
        if regime == 'bearish':
            # 하락장: Long 신호 강하게 약화
            for i in range(len(adjusted_signals)):
                if adjusted_signals[i] > 0:
                    adjusted_signals[i] *= (1 - self.momentum_decay * trend_strength * 1.5)
        elif regime == 'bullish':
            # 상승장: Short 신호 약화
            for i in range(len(adjusted_signals)):
                if adjusted_signals[i] < 0:
                    adjusted_signals[i] *= (1 - self.momentum_decay * trend_strength * 0.3)

        # v2: 선행 지표 기반 추가 조정
        rsi = extra_info.get('rsi', 50)

        # RSI 과매수 시 Long 신호 약화 (하락 임박 가능성)
        if rsi > self.rsi_overbought:
            for i in range(len(adjusted_signals)):
                if adjusted_signals[i] > 0:
                    # RSI가 높을수록 더 많이 약화
                    rsi_factor = (rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
                    adjusted_signals[i] *= (1 - 0.5 * rsi_factor)

        # RSI 과매도 시 Short 신호 약화 (반등 임박 가능성)
        elif rsi < self.rsi_oversold:
            for i in range(len(adjusted_signals)):
                if adjusted_signals[i] < 0:
                    rsi_factor = (self.rsi_oversold - rsi) / self.rsi_oversold
                    adjusted_signals[i] *= (1 - 0.3 * rsi_factor)

        # 데드 크로스 시 Long 신호 강하게 약화
        if extra_info.get('death_cross', False):
            for i in range(len(adjusted_signals)):
                if adjusted_signals[i] > 0:
                    adjusted_signals[i] *= 0.3  # 70% 약화

        # 골든 크로스 시 Short 신호 약화
        if extra_info.get('golden_cross', False):
            for i in range(len(adjusted_signals)):
                if adjusted_signals[i] < 0:
                    adjusted_signals[i] *= 0.5  # 50% 약화

        # 반전 감지 시 강력한 신호 조정
        if extra_info.get('reversal_detected', False):
            reversal_type = extra_info.get('reversal_type', 'none')
            if reversal_type == 'bearish_reversal':
                # 약세 반전: Long 신호 대폭 약화, Short으로 전환 유도
                for i in range(len(adjusted_signals)):
                    if adjusted_signals[i] > 0:
                        adjusted_signals[i] *= 0.2  # 80% 약화
            elif reversal_type == 'bullish_reversal':
                # 강세 반전: Short 신호 약화
                for i in range(len(adjusted_signals)):
                    if adjusted_signals[i] < 0:
                        adjusted_signals[i] *= 0.4  # 60% 약화

        # 과매수/과매도 레짐 처리
        if regime == 'overbought':
            for i in range(len(adjusted_signals)):
                if adjusted_signals[i] > 0:
                    adjusted_signals[i] *= 0.4  # Long 신호 60% 약화
        elif regime == 'oversold':
            for i in range(len(adjusted_signals)):
                if adjusted_signals[i] < 0:
                    adjusted_signals[i] *= 0.5  # Short 신호 50% 약화

        # 반전 경고 레짐 처리
        if regime == 'reversal_warning_down':
            for i in range(len(adjusted_signals)):
                if adjusted_signals[i] > 0:
                    adjusted_signals[i] *= 0.25  # Long 신호 75% 약화
        elif regime == 'reversal_warning_up':
            for i in range(len(adjusted_signals)):
                if adjusted_signals[i] < 0:
                    adjusted_signals[i] *= 0.4  # Short 신호 60% 약화

        return adjusted_signals

    def calculate_equilibrium_returns(self, covariance, market_weights):
        """시장 균형 수익률 계산"""
        implied_returns = self.risk_aversion * covariance @ market_weights
        return implied_returns

    def create_views_from_signals(self, stage1_signals, historical_returns, view_confidence):
        """Stage 1 IQL 신호를 Black-Litterman 뷰로 변환 (동적 신뢰도 적용)"""
        n = len(stage1_signals)

        daily_vol = np.std(historical_returns, axis=0)

        P = np.eye(n)

        scale_factor = 0.02
        Q = stage1_signals * daily_vol * scale_factor

        # 동적 신뢰도 적용
        confidence_scale = 1.0 / (view_confidence + 1e-8) - 1.0
        omega_diag = confidence_scale * (daily_vol ** 2)
        Omega = np.diag(omega_diag)

        return P, Q, Omega

    def calculate_posterior_returns(self, equilibrium_returns, covariance, P, Q, Omega):
        """Black-Litterman 후행 기대수익률 계산"""
        tau_sigma = self.tau * covariance
        tau_sigma_inv = np.linalg.inv(tau_sigma + 1e-6 * np.eye(len(tau_sigma)))
        omega_inv = np.linalg.inv(Omega + 1e-6 * np.eye(len(Omega)))

        precision = tau_sigma_inv + P.T @ omega_inv @ P
        posterior_precision_inv = np.linalg.inv(precision + 1e-6 * np.eye(len(precision)))
        posterior_returns = posterior_precision_inv @ (tau_sigma_inv @ equilibrium_returns + P.T @ omega_inv @ Q)
        posterior_covariance = covariance + posterior_precision_inv

        return posterior_returns, posterior_covariance

    def optimize_weights(self, expected_returns, covariance, stage1_signals=None):
        """평균-분산 최적화로 최적 비중 계산"""
        n = len(expected_returns)

        if self.allow_short and stage1_signals is not None:
            daily_vol = np.sqrt(np.diag(covariance)) + 1e-8
            risk_adjusted_returns = np.abs(expected_returns) / daily_vol

            direction = np.sign(stage1_signals)
            raw_weights = direction * risk_adjusted_returns

            abs_sum = np.sum(np.abs(raw_weights))
            if abs_sum > 1e-6:
                optimal_weights = raw_weights / abs_sum
            else:
                optimal_weights = np.ones(n) / n

            return optimal_weights
        else:
            def objective(w):
                portfolio_return = w @ expected_returns
                portfolio_variance = w @ covariance @ w
                utility = portfolio_return - (self.risk_aversion / 2) * portfolio_variance
                return -utility

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bounds = [(0.0, 1.0) for _ in range(n)]
            w0 = np.ones(n) / n

            result = minimize(objective, w0, method='SLSQP', bounds=bounds,
                            constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-10})

            optimal_weights = result.x
            abs_sum = np.sum(np.abs(optimal_weights))
            if abs_sum > 1e-6:
                optimal_weights = optimal_weights / abs_sum
            else:
                optimal_weights = np.ones(n) / n

            return optimal_weights

    def get_optimal_weights(self, stage1_signals, historical_returns, current_portfolio_value=None):
        """
        Dynamic Black-Litterman 최적 비중 계산 (v2 - Drawdown Protection 포함)

        Args:
            stage1_signals: Stage 1 IQL 신호 (n,)
            historical_returns: 과거 수익률 데이터 (lookback x n)
            current_portfolio_value: 현재 포트폴리오 가치 (drawdown 계산용)

        Returns:
            weights: 최적 포트폴리오 비중 (n,)
            adjustment_info: 동적 조정 정보
        """
        n = self.n_assets

        if len(historical_returns) < 2:
            return np.ones(n) / n, {}

        # 0. Drawdown 상태 업데이트 (v2)
        if current_portfolio_value is not None:
            self.update_drawdown_state(current_portfolio_value)

        # 1. Adaptive View Confidence 계산
        adaptive_confidence, adjustment_info = self.calculate_adaptive_confidence(
            historical_returns, stage1_signals
        )

        # 2. 레짐에 따른 신호 조정 (v2 - extra_info 전달)
        regime = adjustment_info.get('regime', 'neutral')
        trend_strength = adjustment_info.get('trend_strength', 0.0)

        # extra_info를 구성해서 전달
        extra_info = {
            'rsi': adjustment_info.get('rsi', 50),
            'death_cross': adjustment_info.get('death_cross', False),
            'golden_cross': adjustment_info.get('golden_cross', False),
            'reversal_detected': adjustment_info.get('reversal_detected', False),
            'reversal_type': adjustment_info.get('reversal_type', 'none')
        }

        adjusted_signals = self.adjust_signals_by_regime(stage1_signals, regime, trend_strength, extra_info)

        # 3. 공분산 행렬 계산
        covariance = np.cov(historical_returns.T)
        if covariance.ndim == 0:
            covariance = np.array([[covariance]])
        covariance = covariance + 1e-6 * np.eye(n)

        # 4. 시장 균형 비중 및 균형 기대수익률
        market_weights = np.ones(n) / n
        equilibrium_returns = self.calculate_equilibrium_returns(covariance, market_weights)

        # 5. Stage 1 신호를 뷰로 변환 (동적 신뢰도 적용)
        P, Q, Omega = self.create_views_from_signals(adjusted_signals, historical_returns, adaptive_confidence)

        # 6. 후행 기대수익률 계산
        posterior_returns, posterior_covariance = self.calculate_posterior_returns(
            equilibrium_returns, covariance, P, Q, Omega
        )

        # 7. 최적 비중 계산
        optimal_weights = self.optimize_weights(posterior_returns, posterior_covariance, adjusted_signals)

        # 8. Drawdown Protection 적용 (v2 핵심 기능)
        if self.defensive_mode:
            # 방어 모드: 포지션 축소
            # 드로우다운이 클수록 더 많이 축소
            drawdown_ratio = min(self.current_drawdown / self.drawdown_threshold, 1.0)
            scale_factor = 1.0 - (self.cash_ratio_max * drawdown_ratio)

            # Long 포지션은 더 많이 축소, Short 포지션은 유지/증가
            for i in range(n):
                if optimal_weights[i] > 0:
                    optimal_weights[i] *= scale_factor
                # Short 포지션은 유지 (하락장 방어)

            adjustment_info['defensive_mode'] = True
            adjustment_info['position_scale'] = scale_factor
        else:
            adjustment_info['defensive_mode'] = False
            adjustment_info['position_scale'] = 1.0

        # 9. 반전 경고 시 추가 포지션 축소 (v2)
        if self.reversal_warning and not self.defensive_mode:
            # 반전 경고: Long 포지션 30% 추가 축소
            for i in range(n):
                if optimal_weights[i] > 0:
                    optimal_weights[i] *= 0.7
            adjustment_info['reversal_warning_applied'] = True
        else:
            adjustment_info['reversal_warning_applied'] = False

        # 비중 정규화
        abs_sum = np.sum(np.abs(optimal_weights))
        if abs_sum > 1e-6:
            optimal_weights = optimal_weights / abs_sum

        adjustment_info['drawdown'] = self.current_drawdown

        return optimal_weights, adjustment_info

# ============================================================
# Position Tracker (실현 손익 및 물타기 관리)
# ============================================================

class PositionTracker:
    """
    포지션 추적 클래스
    - 각 종목별 수량, 평균단가, 미실현/실현 손익 추적
    - 리밸런싱 시 실현 손익 계산
    - 물타기 로직 구현
    """

    def __init__(self, coins, initial_value):
        self.coins = coins
        self.n_assets = len(coins)
        self.initial_value = initial_value

        # 포지션 정보: {coin: {'quantity': 수량, 'avg_price': 평균단가, 'direction': 'long'/'short'}}
        self.positions = {coin: {'quantity': 0.0, 'avg_price': 0.0, 'direction': None} for coin in coins}

        # 현금
        self.cash = initial_value

        # 실현 손익 누적
        self.total_realized_pnl = 0.0

        # 상계 처리용 누적 이익 저장소
        self.accumulated_profit = 0.0

        # 기록
        self.realized_pnl_history = []
        self.averaging_down_history = []

    def get_position_pnl(self, coin, current_price):
        """특정 종목의 미실현 손익률 계산"""
        pos = self.positions[coin]
        if pos['quantity'] == 0 or pos['avg_price'] == 0:
            return 0.0

        if pos['direction'] == 'long':
            # 롱: (현재가 - 평균단가) / 평균단가
            return (current_price - pos['avg_price']) / pos['avg_price']
        else:
            # 숏: (평균단가 - 현재가) / 평균단가
            return (pos['avg_price'] - current_price) / pos['avg_price']

    def get_position_value(self, coin, current_price):
        """특정 종목의 현재 가치"""
        pos = self.positions[coin]
        if pos['quantity'] == 0:
            return 0.0

        if pos['direction'] == 'long':
            return pos['quantity'] * current_price
        else:
            # 숏: 진입 가치 + 미실현 손익
            entry_value = pos['quantity'] * pos['avg_price']
            unrealized_pnl = pos['quantity'] * (pos['avg_price'] - current_price)
            return entry_value + unrealized_pnl

    def get_total_portfolio_value(self, current_prices):
        """전체 포트폴리오 가치"""
        total = self.cash
        for coin in self.coins:
            total += self.get_position_value(coin, current_prices[coin])
        return total

    def initialize_positions(self, weights, current_prices, portfolio_value):
        """초기 포지션 설정"""
        self.cash = 0.0

        for i, coin in enumerate(self.coins):
            weight = weights[i]
            price = current_prices[coin]

            if abs(weight) < 1e-6:
                self.positions[coin] = {'quantity': 0.0, 'avg_price': 0.0, 'direction': None}
                continue

            position_value = abs(weight) * portfolio_value
            quantity = position_value / price

            if weight > 0:
                self.positions[coin] = {'quantity': quantity, 'avg_price': price, 'direction': 'long'}
            else:
                self.positions[coin] = {'quantity': quantity, 'avg_price': price, 'direction': 'short'}

    def estimate_pnl(self, current_prices):
        """
        리밸런싱 시 예상 실현 손익 계산 (실제 리밸런싱 수행 없이)

        Returns:
            estimated_pnl: 예상 실현 손익
        """
        estimated_pnl = 0.0

        for coin in self.coins:
            pos = self.positions[coin]
            price = current_prices[coin]

            if pos['quantity'] > 0 and pos['avg_price'] > 0:
                if pos['direction'] == 'long':
                    pnl = (price - pos['avg_price']) * pos['quantity']
                else:
                    pnl = (pos['avg_price'] - price) * pos['quantity']

                estimated_pnl += pnl

        return estimated_pnl

    def rebalance_and_calculate_pnl(self, new_weights, current_prices, portfolio_value):
        """
        리밸런싱 수행 및 실현 손익 계산

        Returns:
            realized_pnl: 이번 리밸런싱에서 발생한 실현 손익
        """
        realized_pnl = 0.0

        # 1. 기존 포지션 청산 및 실현 손익 계산
        for i, coin in enumerate(self.coins):
            pos = self.positions[coin]
            price = current_prices[coin]

            if pos['quantity'] > 0 and pos['avg_price'] > 0:
                if pos['direction'] == 'long':
                    # 롱 청산: (현재가 - 평균단가) × 수량
                    pnl = (price - pos['avg_price']) * pos['quantity']
                else:
                    # 숏 청산: (평균단가 - 현재가) × 수량
                    pnl = (pos['avg_price'] - price) * pos['quantity']

                realized_pnl += pnl

        self.total_realized_pnl += realized_pnl
        self.realized_pnl_history.append(realized_pnl)

        # 2. 새 포지션 진입
        self.cash = 0.0
        for i, coin in enumerate(self.coins):
            weight = new_weights[i]
            price = current_prices[coin]

            if abs(weight) < 1e-6:
                self.positions[coin] = {'quantity': 0.0, 'avg_price': 0.0, 'direction': None}
                continue

            position_value = abs(weight) * portfolio_value
            quantity = position_value / price

            if weight > 0:
                self.positions[coin] = {'quantity': quantity, 'avg_price': price, 'direction': 'long'}
            else:
                self.positions[coin] = {'quantity': quantity, 'avg_price': price, 'direction': 'short'}

        return realized_pnl

    def find_worst_performing_coin(self, current_prices):
        """가장 수익률이 낮은 종목 찾기"""
        worst_coin = None
        worst_pnl = float('inf')

        for coin in self.coins:
            pos = self.positions[coin]
            if pos['quantity'] > 0:
                pnl_rate = self.get_position_pnl(coin, current_prices[coin])
                if pnl_rate < worst_pnl:
                    worst_pnl = pnl_rate
                    worst_coin = coin

        return worst_coin, worst_pnl

    def process_pnl_with_netting(self, realized_pnl, current_prices):
        """
        상계 처리 방식의 물타기
        - 이익 발생 시: 누적 저장
        - 손실 발생 시: 누적 이익과 상계 처리
        - 상계 후 양수면: 가장 수익률이 낮은 종목에 물타기
        - 상계 후 음수면: 다음 이익 발생까지 대기

        Args:
            realized_pnl: 실현 손익
            current_prices: 현재 가격

        Returns:
            netting_info: 상계 처리 및 물타기 정보
        """
        netting_info = {
            'action': None,  # 'accumulated', 'netted_profit', 'netted_loss', 'averaging_executed'
            'realized_pnl': realized_pnl,
            'previous_accumulated': self.accumulated_profit,
            'new_accumulated': 0.0,
            'net_amount': 0.0,
            'averaging_executed': False,
            'coin': None,
            'direction': None,
            'amount': 0.0,
            'old_avg_price': 0.0,
            'new_avg_price': 0.0,
            'old_quantity': 0.0,
            'new_quantity': 0.0
        }

        if realized_pnl >= 0:
            # 이익 발생: 누적 저장
            self.accumulated_profit += realized_pnl
            netting_info['action'] = 'accumulated'
            netting_info['new_accumulated'] = self.accumulated_profit
            return netting_info
        else:
            # 손실 발생: 상계 처리
            net_amount = self.accumulated_profit + realized_pnl  # realized_pnl은 음수
            netting_info['net_amount'] = net_amount

            if net_amount > 0:
                # 상계 후 양수: 물타기 실행
                netting_info['action'] = 'netted_profit'

                # 가장 수익률이 낮은 종목 찾기
                worst_coin, worst_pnl = self.find_worst_performing_coin(current_prices)

                if worst_coin is not None:
                    pos = self.positions[worst_coin]
                    current_price = current_prices[worst_coin]

                    # 물타기 금액 = 상계 후 남은 금액
                    averaging_amount = net_amount

                    # 기존 정보 저장
                    netting_info['coin'] = worst_coin
                    netting_info['direction'] = pos['direction']
                    netting_info['amount'] = averaging_amount
                    netting_info['old_avg_price'] = pos['avg_price']
                    netting_info['old_quantity'] = pos['quantity']

                    # 추가 수량 계산
                    additional_quantity = averaging_amount / current_price

                    # 새 평균단가 계산
                    old_value = pos['quantity'] * pos['avg_price']
                    new_total_quantity = pos['quantity'] + additional_quantity
                    new_avg_price = (old_value + averaging_amount) / new_total_quantity

                    # 포지션 업데이트
                    pos['quantity'] = new_total_quantity
                    pos['avg_price'] = new_avg_price

                    netting_info['averaging_executed'] = True
                    netting_info['new_avg_price'] = new_avg_price
                    netting_info['new_quantity'] = new_total_quantity

                    self.averaging_down_history.append({
                        'coin': worst_coin,
                        'direction': pos['direction'],
                        'amount': averaging_amount,
                        'old_avg_price': netting_info['old_avg_price'],
                        'new_avg_price': new_avg_price
                    })

                # 누적 이익 초기화
                self.accumulated_profit = 0.0
                netting_info['new_accumulated'] = 0.0

            else:
                # 상계 후 음수: 물타기 없이 누적 (음수 상태로 저장)
                netting_info['action'] = 'netted_loss'
                self.accumulated_profit = net_amount  # 음수 저장
                netting_info['new_accumulated'] = self.accumulated_profit

            return netting_info

    def find_best_performing_coin(self, current_prices):
        """가장 수익률이 높은 종목 찾기"""
        best_coin = None
        best_pnl = float('-inf')

        for coin in self.coins:
            pos = self.positions[coin]
            if pos['quantity'] > 0:
                pnl_rate = self.get_position_pnl(coin, current_prices[coin])
                if pnl_rate > best_pnl:
                    best_pnl = pnl_rate
                    best_coin = coin

        return best_coin, best_pnl

    def reduce_best_performer_on_loss(self, realized_loss, current_prices):
        """
        손실 발생 시 가장 수익률이 좋은 종목의 가치를 손실만큼 감소
        - 실현 손익이 음수일 경우, 가장 수익률이 좋은 종목의 포지션 수량 감소

        Args:
            realized_loss: 실현 손실 (음수)
            current_prices: 현재 가격

        Returns:
            reduce_info: 감소 정보
        """
        reduce_info = {
            'executed': False,
            'coin': None,
            'direction': None,
            'reduced_amount': 0.0,
            'old_quantity': 0.0,
            'new_quantity': 0.0,
            'pnl_rate': 0.0
        }

        if realized_loss >= 0:
            return reduce_info

        # 가장 수익률이 좋은 종목 찾기
        best_coin, best_pnl = self.find_best_performing_coin(current_prices)

        if best_coin is None:
            return reduce_info

        pos = self.positions[best_coin]
        current_price = current_prices[best_coin]

        # 감소시킬 금액 = 손실 절대값
        reduce_amount = abs(realized_loss)

        # 현재 포지션 가치 계산
        current_value = self.get_position_value(best_coin, current_price)

        # 감소 금액이 현재 가치보다 크면 현재 가치까지만 감소
        if reduce_amount > current_value:
            reduce_amount = current_value

        # 감소시킬 수량 계산
        reduce_quantity = reduce_amount / current_price

        # 기존 정보 저장
        reduce_info['coin'] = best_coin
        reduce_info['direction'] = pos['direction']
        reduce_info['reduced_amount'] = reduce_amount
        reduce_info['old_quantity'] = pos['quantity']
        reduce_info['pnl_rate'] = best_pnl

        # 포지션 수량 감소
        new_quantity = pos['quantity'] - reduce_quantity
        if new_quantity < 0:
            new_quantity = 0

        pos['quantity'] = new_quantity
        reduce_info['new_quantity'] = new_quantity
        reduce_info['executed'] = True

        return reduce_info

    def inject_capital_on_loss(self, realized_loss, new_weights, current_prices):
        """
        손실 발생 시 추가 자금을 투입하여 최적 비중으로 재배분
        - 실현 손익이 음수일 경우, 손실액만큼 추가 자금 투입
        - 투입 자금을 현재 최적 비중에 따라 각 코인에 배분
        - 롱 포지션: 추가 매수
        - 숏 포지션: 추가 매도

        Args:
            realized_loss: 실현 손실 (음수)
            new_weights: 현재 최적 비중
            current_prices: 현재 가격

        Returns:
            inject_info: 자금 투입 정보
        """
        inject_info = {
            'executed': False,
            'injected_amount': abs(realized_loss),
            'distributions': []
        }

        if realized_loss >= 0:
            return inject_info

        inject_amount = abs(realized_loss)

        # 롱/숏 포지션 분리
        long_coins = []
        short_coins = []
        long_weight_sum = 0.0
        short_weight_sum = 0.0

        for i, coin in enumerate(self.coins):
            weight = new_weights[i]
            if weight > 1e-6:
                long_coins.append((coin, i, weight))
                long_weight_sum += weight
            elif weight < -1e-6:
                short_coins.append((coin, i, abs(weight)))
                short_weight_sum += abs(weight)

        # 투입 금액을 롱/숏 비율에 따라 분배
        total_weight = long_weight_sum + short_weight_sum
        if total_weight < 1e-6:
            return inject_info

        long_inject_share = (long_weight_sum / total_weight) * inject_amount
        short_inject_share = (short_weight_sum / total_weight) * inject_amount

        # 롱 포지션에 추가 매수
        if long_weight_sum > 1e-6 and len(long_coins) > 0:
            for coin, idx, weight in long_coins:
                # 해당 종목의 비중 비율에 따라 투입 금액 분배
                coin_inject = (weight / long_weight_sum) * long_inject_share
                price = current_prices[coin]
                additional_quantity = coin_inject / price

                pos = self.positions[coin]
                if pos['quantity'] > 0 and pos['direction'] == 'long':
                    old_value = pos['quantity'] * pos['avg_price']
                    new_total_quantity = pos['quantity'] + additional_quantity
                    new_avg_price = (old_value + coin_inject) / new_total_quantity

                    inject_info['distributions'].append({
                        'coin': coin,
                        'direction': 'long',
                        'amount': coin_inject,
                        'old_avg_price': pos['avg_price'],
                        'new_avg_price': new_avg_price,
                        'additional_quantity': additional_quantity
                    })

                    pos['quantity'] = new_total_quantity
                    pos['avg_price'] = new_avg_price

        # 숏 포지션에 추가 매도
        if short_weight_sum > 1e-6 and len(short_coins) > 0:
            for coin, idx, weight in short_coins:
                # 해당 종목의 비중 비율에 따라 투입 금액 분배
                coin_inject = (weight / short_weight_sum) * short_inject_share
                price = current_prices[coin]
                additional_quantity = coin_inject / price

                pos = self.positions[coin]
                if pos['quantity'] > 0 and pos['direction'] == 'short':
                    old_value = pos['quantity'] * pos['avg_price']
                    new_total_quantity = pos['quantity'] + additional_quantity
                    new_avg_price = (old_value + coin_inject) / new_total_quantity

                    inject_info['distributions'].append({
                        'coin': coin,
                        'direction': 'short',
                        'amount': coin_inject,
                        'old_avg_price': pos['avg_price'],
                        'new_avg_price': new_avg_price,
                        'additional_quantity': additional_quantity
                    })

                    pos['quantity'] = new_total_quantity
                    pos['avg_price'] = new_avg_price

        if len(inject_info['distributions']) > 0:
            inject_info['executed'] = True

        return inject_info


# ============================================================
# Portfolio Environment
# ============================================================

class PortfolioEnvironment:
    """포트폴리오 환경"""
    def __init__(self, all_data, coins, config):
        self.all_data = all_data
        self.coins = coins
        self.n_assets = len(coins)
        self.config = config
        self.window_size = config['environment']['window_size']
        self.fee_rate = config['portfolio'].get('fee_rate', 0.001)

        self._align_data()

    def _align_data(self):
        """모든 코인의 날짜를 정렬"""
        date_sets = [set(self.all_data[coin]['datetime']) for coin in self.coins]
        common_dates = sorted(set.intersection(*date_sets))

        self.dates = common_dates
        self.returns = {}
        self.prices = {}

        for coin in self.coins:
            df = self.all_data[coin]
            df_filtered = df[df['datetime'].isin(common_dates)].sort_values('datetime')
            self.returns[coin] = df_filtered['return'].values
            self.prices[coin] = df_filtered['close'].values

        print(f"  Aligned data: {len(common_dates)} common days")

    def get_historical_returns(self, idx, lookback=60):
        """과거 수익률 행렬 반환"""
        start_idx = max(0, idx - lookback)
        returns_matrix = []
        for coin in self.coins:
            returns_matrix.append(self.returns[coin][start_idx:idx])
        return np.array(returns_matrix).T

# ============================================================
# Portfolio Evaluation
# ============================================================

def evaluate_portfolio(env, stage1_agents, bl_optimizer, config, start_idx, end_idx):
    """포트폴리오 평가 - Dynamic Black-Litterman (v4 - 물타기 효과 실제 반영)"""
    window_size = config['environment']['window_size']
    learning_window = config['training'].get('learning_window', 30)
    initial_value = config['portfolio'].get('initial_value', 10000)
    fee_rate = config['portfolio'].get('fee_rate', 0.001)
    lookback = config.get('black_litterman', {}).get('lookback', 60)

    n_assets = len(env.coins)
    current_weights = np.ones(n_assets) / n_assets

    # Drawdown 추적 초기화
    bl_optimizer.peak_value = initial_value
    bl_optimizer.current_drawdown = 0.0
    bl_optimizer.defensive_mode = False

    # v4: 포지션 트래커 초기화 (실제 포트폴리오 가치 추적용)
    position_tracker = PositionTracker(env.coins, initial_value)
    is_first_rebalance = True

    results = {
        'dates': [],
        'portfolio_values': [],
        'portfolio_values_without_averaging': [],  # 물타기 없는 버전 비교용
        'weights': [],
        'daily_returns': [],
        'turnovers': [],
        'stage1_signals': [],
        'long_short_ratio': [],
        'view_confidence': [],
        'regime': [],
        'volatility': [],
        'drawdown': [],
        'defensive_mode': [],
        'reversal_warning': [],
        # v3: 실현손익 및 물타기 기록 추가
        'realized_pnl': [],
        'averaging_down_info': []
    }

    rebalance_count = 0
    last_rebalance_idx = start_idx
    defensive_mode_count = 0
    reversal_warning_count = 0
    averaging_down_count = 0
    total_realized_pnl = 0.0
    total_averaging_amount = 0.0  # 물타기에 사용된 총 금액

    # v4: 손익 통계
    profit_count = 0  # 이익 발생 횟수
    loss_count = 0    # 손실 발생 횟수
    total_profit = 0.0  # 총 이익
    total_loss = 0.0    # 총 손실

    # 비교용: 물타기 없는 포트폴리오 가치
    portfolio_value_no_avg = initial_value

    for idx in range(start_idx, end_idx):
        if idx < window_size + 20:
            continue

        # 현재 가격
        current_prices = {coin: env.prices[coin][idx] for coin in env.coins}

        # Stage 1 신호 생성
        stage1_signals = []
        for coin in env.coins:
            state = env.returns[coin][idx - window_size:idx]
            signal = stage1_agents[coin].select_action(state)
            stage1_signals.append(signal)
        stage1_signals = np.array(stage1_signals)

        # v4: PositionTracker 기반 현재 포트폴리오 가치 계산
        if not is_first_rebalance:
            portfolio_value = position_tracker.get_total_portfolio_value(current_prices)
        else:
            portfolio_value = initial_value

        # 리밸런싱 여부 결정 (v2: drawdown이 크면 더 자주 리밸런싱)
        force_rebalance = bl_optimizer.defensive_mode and (idx - last_rebalance_idx >= learning_window // 2)
        normal_rebalance = idx - last_rebalance_idx >= learning_window or idx == start_idx

        realized_pnl = 0.0
        averaging_info = {'executed': False}

        if force_rebalance or normal_rebalance:
            if not is_first_rebalance:
                # 리밸런싱 수행
                historical_returns = env.get_historical_returns(idx, lookback)
                new_weights, adjustment_info = bl_optimizer.get_optimal_weights(
                    stage1_signals, historical_returns, portfolio_value
                )

                turnover = np.sum(np.abs(new_weights - current_weights))
                transaction_cost = turnover * fee_rate * portfolio_value
                portfolio_value -= transaction_cost
                portfolio_value_no_avg -= turnover * fee_rate * portfolio_value_no_avg

                # 리밸런싱 시 실현 손익 계산
                realized_pnl = position_tracker.rebalance_and_calculate_pnl(
                    new_weights, current_prices, portfolio_value
                )
                total_realized_pnl += realized_pnl

                # 손익 통계 업데이트
                if realized_pnl >= 0:
                    profit_count += 1
                    total_profit += realized_pnl
                else:
                    loss_count += 1
                    total_loss += abs(realized_pnl)

                # 상계 처리 방식 물타기
                netting_info = position_tracker.process_pnl_with_netting(realized_pnl, current_prices)

                if netting_info['action'] == 'accumulated':
                    # 이익 누적
                    print(f"    [{env.dates[idx].strftime('%Y-%m-%d')}] 이익 누적: ${realized_pnl:.2f} | "
                          f"누적 이익: ${netting_info['new_accumulated']:.2f}")
                elif netting_info['action'] == 'netted_profit':
                    # 상계 후 양수 → 물타기 실행
                    if netting_info['averaging_executed']:
                        averaging_down_count += 1
                        total_averaging_amount += netting_info['amount']
                        print(f"    [{env.dates[idx].strftime('%Y-%m-%d')}] 상계 후 물타기: {netting_info['coin']} "
                              f"({netting_info['direction']}) | 손실: ${realized_pnl:.2f} | "
                              f"누적이익: ${netting_info['previous_accumulated']:.2f} | "
                              f"상계후: ${netting_info['net_amount']:.2f} | "
                              f"물타기금액: ${netting_info['amount']:.2f} | "
                              f"평균단가: ${netting_info['old_avg_price']:.2f} → ${netting_info['new_avg_price']:.2f}")
                    else:
                        print(f"    [{env.dates[idx].strftime('%Y-%m-%d')}] 상계 완료: 손실 ${realized_pnl:.2f} | "
                              f"누적이익: ${netting_info['previous_accumulated']:.2f} | "
                              f"상계후: ${netting_info['net_amount']:.2f}")
                elif netting_info['action'] == 'netted_loss':
                    # 상계 후 음수 → 대기
                    print(f"    [{env.dates[idx].strftime('%Y-%m-%d')}] 상계 후 적자: 손실 ${realized_pnl:.2f} | "
                          f"누적이익: ${netting_info['previous_accumulated']:.2f} | "
                          f"상계후: ${netting_info['net_amount']:.2f} (다음 이익까지 대기)")

                current_weights = new_weights
                last_rebalance_idx = idx
                rebalance_count += 1

                # 동적 조정 정보 기록
                results['view_confidence'].append(adjustment_info.get('final_confidence', 0.6))
                results['regime'].append(adjustment_info.get('regime', 'neutral'))
                results['volatility'].append(adjustment_info.get('volatility', 0.0))
                results['drawdown'].append(adjustment_info.get('drawdown', 0.0))
                results['defensive_mode'].append(adjustment_info.get('defensive_mode', False))
                results['reversal_warning'].append(adjustment_info.get('reversal_warning_applied', False))

                if adjustment_info.get('defensive_mode', False):
                    defensive_mode_count += 1
                if adjustment_info.get('reversal_warning_applied', False):
                    reversal_warning_count += 1
            else:
                # 첫 리밸런싱: 초기 포지션 설정
                historical_returns = env.get_historical_returns(idx, lookback)
                new_weights, adjustment_info = bl_optimizer.get_optimal_weights(
                    stage1_signals, historical_returns, portfolio_value
                )

                turnover = np.sum(np.abs(new_weights - current_weights))
                transaction_cost = turnover * fee_rate * portfolio_value
                portfolio_value -= transaction_cost
                portfolio_value_no_avg -= turnover * fee_rate * portfolio_value_no_avg

                position_tracker.initialize_positions(new_weights, current_prices, portfolio_value)
                is_first_rebalance = False

                current_weights = new_weights
                last_rebalance_idx = idx
                rebalance_count += 1

                # 동적 조정 정보 기록
                results['view_confidence'].append(adjustment_info.get('final_confidence', 0.6))
                results['regime'].append(adjustment_info.get('regime', 'neutral'))
                results['volatility'].append(adjustment_info.get('volatility', 0.0))
                results['drawdown'].append(adjustment_info.get('drawdown', 0.0))
                results['defensive_mode'].append(adjustment_info.get('defensive_mode', False))
                results['reversal_warning'].append(adjustment_info.get('reversal_warning_applied', False))

                if adjustment_info.get('defensive_mode', False):
                    defensive_mode_count += 1
            if adjustment_info.get('reversal_warning_applied', False):
                reversal_warning_count += 1
        else:
            turnover = 0.0
            if results['view_confidence']:
                results['view_confidence'].append(results['view_confidence'][-1])
                results['regime'].append(results['regime'][-1])
                results['volatility'].append(results['volatility'][-1])
                results['drawdown'].append(results['drawdown'][-1])
                results['defensive_mode'].append(results['defensive_mode'][-1])
                results['reversal_warning'].append(results['reversal_warning'][-1])
            else:
                results['view_confidence'].append(0.6)
                results['regime'].append('neutral')
                results['volatility'].append(0.0)
                results['drawdown'].append(0.0)
                results['defensive_mode'].append(False)
                results['reversal_warning'].append(False)

        # v4: 포트폴리오 가치는 PositionTracker에서 계산
        portfolio_value = position_tracker.get_total_portfolio_value(current_prices)

        # 비교용: 물타기 없는 버전의 수익률 적용
        daily_returns = np.array([env.returns[coin][idx] for coin in env.coins])
        portfolio_return_no_avg = np.sum(current_weights * daily_returns)
        portfolio_value_no_avg *= (1 + portfolio_return_no_avg)

        # 일별 수익률 (PositionTracker 기반)
        if len(results['portfolio_values']) > 0:
            prev_value = results['portfolio_values'][-1]
            portfolio_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0.0
        else:
            portfolio_return = 0.0

        # Long/Short 비율
        long_exposure = np.sum(current_weights[current_weights > 0])
        short_exposure = np.sum(np.abs(current_weights[current_weights < 0]))

        # 기록
        results['dates'].append(env.dates[idx])
        results['portfolio_values'].append(portfolio_value)
        results['portfolio_values_without_averaging'].append(portfolio_value_no_avg)
        results['weights'].append(current_weights.copy())
        results['daily_returns'].append(portfolio_return)
        results['turnovers'].append(turnover)
        results['stage1_signals'].append(stage1_signals.copy())
        results['long_short_ratio'].append((long_exposure, short_exposure))
        results['realized_pnl'].append(realized_pnl)
        results['averaging_down_info'].append(averaging_info)

    print(f"  Total rebalancing: {rebalance_count} times")

    avg_long = np.mean([ls[0] for ls in results['long_short_ratio']])
    avg_short = np.mean([ls[1] for ls in results['long_short_ratio']])
    print(f"  Average Long exposure: {avg_long:.1%}, Short exposure: {avg_short:.1%}")

    # 레짐 통계
    regime_counts = {}
    for r in results['regime']:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    print(f"  Regime distribution: {regime_counts}")

    avg_confidence = np.mean(results['view_confidence'])
    print(f"  Average View Confidence: {avg_confidence:.3f}")

    # v2 통계
    print(f"  Defensive mode activated: {defensive_mode_count} times")
    print(f"  Reversal warning applied: {reversal_warning_count} times")
    max_drawdown = max(results['drawdown']) if results['drawdown'] else 0
    print(f"  Max drawdown during trading: {max_drawdown:.1%}")

    # v4 통계: 실현손익 및 물타기 (효과 비교 포함)
    print(f"\n  [실현손익 & 물타기 통계]")
    print(f"  리밸런싱 이익 발생: {profit_count}회 (총 ${total_profit:.2f})")
    print(f"  리밸런싱 손실 발생: {loss_count}회 (총 ${total_loss:.2f})")
    print(f"  순 실현 손익: ${total_realized_pnl:.2f}")
    print(f"  물타기 실행: {averaging_down_count}회")
    print(f"  물타기 총 금액: ${total_averaging_amount:.2f}")

    # 물타기 효과 비교
    final_with_avg = results['portfolio_values'][-1]
    final_without_avg = results['portfolio_values_without_averaging'][-1]
    averaging_effect = final_with_avg - final_without_avg
    print(f"\n  [물타기 효과]")
    print(f"  물타기 포함 최종가치: ${final_with_avg:,.2f}")
    print(f"  물타기 미포함 최종가치: ${final_without_avg:,.2f}")
    print(f"  물타기 효과: ${averaging_effect:+,.2f} ({averaging_effect/initial_value*100:+.2f}%)")

    # 물타기 상세 정보
    if position_tracker.averaging_down_history:
        avg_coins = {}
        for info in position_tracker.averaging_down_history:
            coin = info['coin']
            avg_coins[coin] = avg_coins.get(coin, 0) + 1
        print(f"  Averaging down by coin: {avg_coins}")

    # results에 추가 정보 저장
    results['total_realized_pnl'] = total_realized_pnl
    results['averaging_down_count'] = averaging_down_count
    results['total_averaging_amount'] = total_averaging_amount
    results['averaging_effect'] = averaging_effect
    results['position_tracker'] = position_tracker
    return results

def evaluate_baselines(env, config, start_idx, end_idx):
    """베이스라인 전략 평가"""
    window_size = config['environment']['window_size']
    initial_value = config['portfolio'].get('initial_value', 10000)

    n_assets = len(env.coins)

    equal_weights = np.ones(n_assets) / n_assets
    equal_value = initial_value
    equal_values = []

    bh_values = {coin: [initial_value] for coin in env.coins}

    for idx in range(start_idx, end_idx):
        if idx < window_size + 20:
            continue

        daily_returns = np.array([env.returns[coin][idx] for coin in env.coins])

        equal_return = np.sum(equal_weights * daily_returns)
        equal_value *= (1 + equal_return)
        equal_values.append(equal_value)

        for i, coin in enumerate(env.coins):
            bh_values[coin].append(bh_values[coin][-1] * (1 + daily_returns[i]))

    return {
        'equal_weight': equal_values,
        'buy_hold': bh_values
    }

# ============================================================
# Visualization
# ============================================================

def plot_portfolio_results(results, baselines, env, config, save_path=None):
    """포트폴리오 결과 시각화 (Dynamic BL 버전)"""
    fig = plt.figure(figsize=(20, 20))

    dates = results['dates']
    portfolio_values = results['portfolio_values']
    initial_value = config['portfolio'].get('initial_value', 10000)

    # 1. 포트폴리오 가치 변화
    ax1 = fig.add_subplot(4, 2, (1, 2))

    ax1.plot(dates, portfolio_values, 'b-', linewidth=2.5, label='Dynamic B-L (Adaptive Confidence)')
    ax1.plot(dates, baselines['equal_weight'], 'g--', linewidth=2, alpha=0.7, label='Equal Weight')

    colors = ['#F7931A', '#627EEA', '#00FFA3', '#C2A633', '#00C1DE', '#888888']
    for i, coin in enumerate(env.coins):
        bh = baselines['buy_hold'][coin][1:]
        if len(bh) == len(dates):
            ax1.plot(dates, bh, '--', linewidth=1, alpha=0.5, color=colors[i], label=f'{coin} B&H')

    ax1.axhline(y=initial_value, color='gray', linestyle=':', alpha=0.5)

    final_value = portfolio_values[-1]
    final_return = (final_value / initial_value - 1) * 100
    ax1.set_title(f'Dynamic Black-Litterman 포트폴리오 (Adaptive Confidence)\n최종: ${final_value:,.0f} ({final_return:+.1f}%)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. 누적 수익률 비교
    ax2 = fig.add_subplot(4, 2, 3)

    iql_returns = [(v / initial_value - 1) * 100 for v in portfolio_values]
    equal_returns = [(v / initial_value - 1) * 100 for v in baselines['equal_weight']]

    ax2.plot(dates, iql_returns, 'b-', linewidth=2, label='Dynamic B-L')
    ax2.plot(dates, equal_returns, 'g--', linewidth=2, alpha=0.7, label='Equal Weight')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.fill_between(dates, 0, iql_returns, alpha=0.3,
                     where=[r > 0 for r in iql_returns], color='green')
    ax2.fill_between(dates, 0, iql_returns, alpha=0.3,
                     where=[r <= 0 for r in iql_returns], color='red')

    ax2.set_title('누적 수익률 비교', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative Return (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Adaptive View Confidence 변화
    ax3 = fig.add_subplot(4, 2, 4)

    view_conf = results['view_confidence']
    ax3.plot(dates, view_conf, 'purple', linewidth=1.5, label='View Confidence')
    ax3.axhline(y=config['black_litterman'].get('view_confidence', 0.6),
                color='gray', linestyle='--', alpha=0.7, label='Base Confidence')
    ax3.fill_between(dates, config['dynamic_bl'].get('confidence_min', 0.2),
                     config['dynamic_bl'].get('confidence_max', 0.8),
                     alpha=0.1, color='purple', label='Confidence Range')

    # 레짐 표시
    regime_colors = {'bullish': 'green', 'bearish': 'red', 'neutral': 'gray'}
    for i, (date, regime) in enumerate(zip(dates, results['regime'])):
        if i % 10 == 0:  # 10일마다 표시
            ax3.axvline(x=date, color=regime_colors.get(regime, 'gray'), alpha=0.1, linewidth=2)

    ax3.set_title('Adaptive View Confidence (시장 상황에 따른 동적 조정)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('View Confidence')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # 4. 비중 변화
    ax4 = fig.add_subplot(4, 2, 5)

    weights_array = np.array(results['weights'])
    x = range(len(dates))

    colors_coins = ['#F7931A', '#C2A633', '#627EEA', '#00C1DE', '#00FFA3', '#888888']
    for i, coin in enumerate(env.coins):
        ax4.plot(x, weights_array[:, i], label=coin, color=colors_coins[i], alpha=0.7, linewidth=1)

    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_title('포트폴리오 비중 변화', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Trading Days')
    ax4.set_ylabel('Weight (+ Long, - Short)')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.set_ylim(-0.6, 0.6)
    ax4.grid(True, alpha=0.3)

    # 5. 시장 변동성 추이
    ax5 = fig.add_subplot(4, 2, 6)

    volatility = results['volatility']
    ax5.plot(dates, [v * 100 for v in volatility], 'orange', linewidth=1.5)
    ax5.axhline(y=config['dynamic_bl'].get('vol_threshold_high', 0.04) * 100,
                color='red', linestyle='--', alpha=0.7, label='High Vol Threshold')
    ax5.axhline(y=config['dynamic_bl'].get('vol_threshold_low', 0.015) * 100,
                color='green', linestyle='--', alpha=0.7, label='Low Vol Threshold')

    ax5.set_title('시장 변동성 추이 (Adaptive Confidence 결정 요인)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Daily Volatility (%)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 일별 수익률
    ax6 = fig.add_subplot(4, 2, 7)
    daily_returns_pct = [r * 100 for r in results['daily_returns']]
    colors = ['green' if r > 0 else 'red' for r in daily_returns_pct]
    ax6.bar(range(len(daily_returns_pct)), daily_returns_pct, color=colors, alpha=0.7, width=1.0)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax6.set_title('일별 포트폴리오 수익률', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Trading Days')
    ax6.set_ylabel('Daily Return (%)')
    ax6.grid(True, alpha=0.3)

    # 7. Drawdown
    ax7 = fig.add_subplot(4, 2, 8)
    cum_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(cum_values)
    drawdown = (peak - cum_values) / peak * 100

    ax7.fill_between(dates, 0, -drawdown, color='red', alpha=0.5)
    ax7.plot(dates, -drawdown, 'r-', linewidth=1)
    ax7.set_title(f'Drawdown (최대 낙폭: {np.max(drawdown):.1f}%)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Date')
    ax7.set_ylabel('Drawdown (%)')
    ax7.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nResults saved to: {save_path}")

    plt.show()

    print_performance_stats(results, baselines, env, config)

def print_performance_stats(results, baselines, env, config):
    """성과 통계 출력"""
    initial_value = config['portfolio'].get('initial_value', 10000)
    portfolio_values = results['portfolio_values']
    daily_returns = np.array(results['daily_returns'])

    print("\n" + "="*70)
    print("    Dynamic Black-Litterman 포트폴리오 성과 분석")
    print("    (Adaptive View Confidence)")
    print("="*70)

    final_return = (portfolio_values[-1] / initial_value - 1) * 100
    equal_return = (baselines['equal_weight'][-1] / initial_value - 1) * 100

    print(f"\n[수익률]")
    print(f"  Dynamic B-L:   {final_return:+.2f}%")
    print(f"  Equal Weight:  {equal_return:+.2f}%")
    print(f"  초과 수익:     {final_return - equal_return:+.2f}%")

    print(f"\n[개별 코인 Buy & Hold]")
    for coin in env.coins:
        bh_return = (baselines['buy_hold'][coin][-1] / initial_value - 1) * 100
        print(f"  {coin}: {bh_return:+.2f}%")

    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)

    cum_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(cum_values)
    drawdown = (peak - cum_values) / peak
    max_dd = np.max(drawdown) * 100

    print(f"\n[리스크 지표]")
    print(f"  연율화 샤프비율: {sharpe:.3f}")
    print(f"  최대 낙폭 (MDD): {max_dd:.2f}%")
    print(f"  일별 변동성:     {np.std(daily_returns) * 100:.3f}%")

    wins = np.sum(daily_returns > 0)
    total = len(daily_returns)
    print(f"\n[승률]")
    print(f"  승리: {wins}일 / 전체: {total}일 ({wins/total*100:.1f}%)")

    total_turnover = np.sum(results['turnovers'])
    total_cost = total_turnover * config['portfolio'].get('fee_rate', 0.001) * initial_value
    print(f"\n[거래]")
    print(f"  총 Turnover: {total_turnover:.2f}")
    print(f"  추정 거래비용: ${total_cost:.2f}")

    if 'long_short_ratio' in results:
        avg_long = np.mean([ls[0] for ls in results['long_short_ratio']])
        avg_short = np.mean([ls[1] for ls in results['long_short_ratio']])
        print(f"\n[Long/Short 비율]")
        print(f"  평균 Long 비중: {avg_long:.1%}")
        print(f"  평균 Short 비중: {avg_short:.1%}")
        print(f"  Net 방향: {'Long 우위' if avg_long > avg_short else 'Short 우위'}")

    # Adaptive Confidence 통계
    print(f"\n[Adaptive View Confidence 통계]")
    conf = results['view_confidence']
    print(f"  평균: {np.mean(conf):.3f}")
    print(f"  최소: {np.min(conf):.3f}")
    print(f"  최대: {np.max(conf):.3f}")
    print(f"  표준편차: {np.std(conf):.3f}")

    # 레짐 분포
    regime_counts = {}
    for r in results['regime']:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    print(f"\n[시장 레짐 분포]")
    total_days = len(results['regime'])
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count}일 ({count/total_days*100:.1f}%)")

    print("="*70)

# ============================================================
# Main
# ============================================================

def main():
    global device

    folder_path = os.path.dirname(os.path.abspath(__file__))

    # Config 로드
    bl_config_path = os.path.join(folder_path, 'bl_portfolio_config.yaml')
    portfolio_config_path = os.path.join(folder_path, 'portfolio_config.yaml')
    near_config_path = os.path.join(folder_path, 'near_iql_config.yaml')

    if os.path.exists(bl_config_path):
        config = load_config(bl_config_path)
        print(f"Config loaded from: bl_portfolio_config.yaml")
    elif os.path.exists(portfolio_config_path):
        config = load_config(portfolio_config_path)
        print(f"Config loaded from: portfolio_config.yaml")
    elif os.path.exists(near_config_path):
        config = load_config(near_config_path)
        print(f"Config loaded from: near_iql_config.yaml")
    else:
        config = get_default_config()
        print("Using default configuration")

    # 포트폴리오 설정 기본값
    if 'portfolio' not in config:
        config['portfolio'] = {}
    config['portfolio'].setdefault('initial_value', 10000)
    config['portfolio'].setdefault('fee_rate', 0.001)
    config['portfolio'].setdefault('allow_short', True)
    config['portfolio'].setdefault('max_leverage', 1.0)

    # Black-Litterman 설정 기본값
    if 'black_litterman' not in config:
        config['black_litterman'] = {}
    config['black_litterman'].setdefault('tau', 0.05)
    config['black_litterman'].setdefault('risk_aversion', 2.5)
    config['black_litterman'].setdefault('view_confidence', 0.6)
    config['black_litterman'].setdefault('lookback', 60)

    # Dynamic BL 설정 기본값
    if 'dynamic_bl' not in config:
        config['dynamic_bl'] = {}
    config['dynamic_bl'].setdefault('vol_lookback', 20)
    config['dynamic_bl'].setdefault('vol_threshold_high', 0.04)
    config['dynamic_bl'].setdefault('vol_threshold_low', 0.015)
    config['dynamic_bl'].setdefault('confidence_min', 0.2)
    config['dynamic_bl'].setdefault('confidence_max', 0.8)
    config['dynamic_bl'].setdefault('trend_lookback', 10)
    config['dynamic_bl'].setdefault('trend_threshold', 0.05)
    config['dynamic_bl'].setdefault('regime_weight', 0.3)
    config['dynamic_bl'].setdefault('momentum_decay', 0.5)

    setup_seed(config.get('seed', 42))
    device = setup_device(config.get('gpu', {}).get('use_cuda', True))

    coins = ['BTC', 'DOGE', 'ETH', 'NEAR', 'SOL', 'WLD']

    print("\n" + "="*70)
    print("    Two-Stage Portfolio: IQL + Dynamic Black-Litterman")
    print("    (Adaptive View Confidence)")
    print("="*70)
    print(f"Coins: {', '.join(coins)}")
    print(f"Train period: ~ {config['data']['train_end_date']}")
    print(f"Test period: {config['data']['test_start_date']} ~")
    print(f"Rebalancing Window: {config['training'].get('learning_window', 30)} days")
    print(f"Strategy: {'Long/Short' if config['portfolio']['allow_short'] else 'Long Only'}")
    print(f"\n[Black-Litterman Parameters]")
    print(f"  Tau: {config['black_litterman']['tau']}")
    print(f"  Risk Aversion: {config['black_litterman']['risk_aversion']}")
    print(f"  Base View Confidence: {config['black_litterman']['view_confidence']}")
    print(f"\n[Dynamic BL Parameters (Adaptive Confidence)]")
    print(f"  Volatility Lookback: {config['dynamic_bl']['vol_lookback']} days")
    print(f"  High Vol Threshold: {config['dynamic_bl']['vol_threshold_high']*100:.1f}%")
    print(f"  Low Vol Threshold: {config['dynamic_bl']['vol_threshold_low']*100:.1f}%")
    print(f"  Confidence Range: [{config['dynamic_bl']['confidence_min']}, {config['dynamic_bl']['confidence_max']}]")
    print(f"  Trend Lookback: {config['dynamic_bl']['trend_lookback']} days")
    print(f"  Regime Weight: {config['dynamic_bl']['regime_weight']}")
    print(f"  Momentum Decay: {config['dynamic_bl']['momentum_decay']}")
    print("="*70)

    # 데이터 로드
    print("\n[1] Loading data...")
    all_data = load_all_coin_data(folder_path, coins)

    # 포트폴리오 환경 생성
    print("\n[2] Creating portfolio environment...")
    env = PortfolioEnvironment(all_data, coins, config)

    # 날짜 인덱스 찾기
    train_end = pd.to_datetime(config['data']['train_end_date'])
    test_start = pd.to_datetime(config['data']['test_start_date'])

    train_end_idx = None
    test_start_idx = None
    for i, date in enumerate(env.dates):
        if date <= train_end:
            train_end_idx = i
        if date >= test_start and test_start_idx is None:
            test_start_idx = i

    print(f"  Train end index: {train_end_idx}")
    print(f"  Test start index: {test_start_idx}")

    # ==================== Stage 1 ====================
    print("\n[3] Stage 1: Training individual coin IQL agents...")
    stage1_agents = {}
    window_size = config['environment']['window_size']

    for coin in coins:
        print(f"\n  Training {coin} agent...")
        setup_seed(config.get('seed', 42))

        agent = Stage1IQLAgent(state_dim=window_size, action_dim=1, config=config['iql'])

        dataset = create_stage1_dataset(all_data[coin], config, train_end_idx)
        if len(dataset['states']) > 0:
            agent = train_stage1_agent(agent, dataset, config)
            print(f"    Trained on {len(dataset['states'])} samples")

        stage1_agents[coin] = agent

    # ==================== Stage 2: Dynamic Black-Litterman ====================
    print("\n[4] Stage 2: Initializing Dynamic Black-Litterman optimizer...")
    bl_optimizer = DynamicBlackLittermanOptimizer(n_assets=len(coins), config=config)
    print("  Dynamic Black-Litterman optimizer ready (Adaptive Confidence enabled)")

    # ==================== Evaluation ====================
    print("\n[5] Evaluating portfolio on test period...")
    results = evaluate_portfolio(env, stage1_agents, bl_optimizer, config,
                                  test_start_idx, len(env.dates))

    print("\n[6] Evaluating baselines...")
    baselines = evaluate_baselines(env, config, test_start_idx, len(env.dates))

    # ==================== Visualization ====================
    print("\n[7] Visualizing results...")
    save_path = os.path.join(folder_path, 'two_stage_DYBL_portfolio_results.png')
    plot_portfolio_results(results, baselines, env, config, save_path)

if __name__ == "__main__":
    main()
