"""
SDELP-DDPG Portfolio Environment
논문 Section 3.1-3.2: MDP 정의, 보상 함수, 거래비용 모델
"""

import numpy as np
import pandas as pd
import torch
import warnings

# curl_cffi SSL 인증서 검증 우회 (yfinance가 curl_cffi 백엔드 사용)
try:
    import curl_cffi.requests
    _original_init = curl_cffi.requests.Session.__init__
    def _patched_init(self, *args, **kwargs):
        kwargs['verify'] = False
        _original_init(self, *args, **kwargs)
    curl_cffi.requests.Session.__init__ = _patched_init
except ImportError:
    pass
warnings.filterwarnings("ignore")

import yfinance as yf
from config import *


class PortfolioEnv:
    """
    포트폴리오 관리 MDP 환경

    State:  가격 변화율 텐서 (window × assets) + 현재 포트폴리오 가중치
    Action: 목표 포트폴리오 가중치 (softmax 정규화, 현금 포함)
    Reward: 로그 수익률 - β × 상대 엔트로피 (KL divergence)
    """

    def __init__(self, tickers=TICKERS, start=TRAIN_START, end=TRAIN_END,
                 window=WINDOW_SIZE, tc_rate=TRANSACTION_COST, beta=BETA):
        self.tickers = tickers
        self.window = window
        self.tc_rate = tc_rate
        self.beta = beta
        self.num_assets = len(tickers)
        self.total_assets = self.num_assets + 1  # +1 for cash

        # 데이터 로드
        self.prices, self.returns, self.dates = self._load_data(tickers, start, end)
        self.num_steps = len(self.prices) - window

        # 상태/행동 차원
        self.state_dim = self.num_assets * window + self.total_assets
        self.action_dim = self.total_assets

        self.reset()

    def _load_data(self, tickers, start, end):
        """Yahoo Finance에서 가격 데이터 로드"""
        df = yf.download(tickers, start=start, end=end, auto_adjust=True)

        # 종가 추출
        if len(tickers) == 1:
            close = df["Close"].to_frame()
        else:
            close = df["Close"]

        # 결측치 처리 (논문: 인접 값 평균으로 대체)
        close = close.ffill().bfill()

        # NaN이 남아있는 열 제거
        valid_cols = close.dropna(axis=1, how="any").columns
        if len(valid_cols) < len(tickers):
            dropped = set(tickers) - set(valid_cols)
            print(f"  ⚠ 데이터 부족으로 제외된 종목: {dropped}")
            close = close[valid_cols]
            self.tickers = list(valid_cols)
            self.num_assets = len(self.tickers)
            self.total_assets = self.num_assets + 1

        prices = close.values  # (T, num_assets)

        if prices.shape[0] < self.window + 2:
            raise ValueError(
                f"데이터가 부족합니다. {prices.shape[0]}일 (최소 {self.window + 2}일 필요). "
                f"기간을 확인해주세요: {start} ~ {end}"
            )

        # 상대 가격 변화율: v_t = p_t / p_{t-1}
        returns = prices[1:] / prices[:-1]  # (T-1, num_assets)
        dates = close.index[1:]  # 날짜 인덱스 (returns와 동일 길이)
        prices = prices[1:]  # 정렬

        print(f"  [OK] Data loaded: {prices.shape[0]} days x {prices.shape[1]} assets")

        return prices, returns, dates

    def reset(self):
        """환경 초기화"""
        self.step_idx = 0
        # 초기 포트폴리오: 100% 현금
        self.weights = np.zeros(self.total_assets)
        self.weights[0] = 1.0  # index 0 = cash
        self.portfolio_value = 1.0
        self.portfolio_values = [1.0]
        self.weight_history = [self.weights.copy()]

        return self._get_state()

    def _get_state(self):
        """
        State 구성:
        - 가격 변화율 윈도우: (window, num_assets) → flatten
        - 현재 포트폴리오 가중치: (total_assets,)
        """
        start = self.step_idx
        end = start + self.window
        price_window = self.returns[start:end]  # (window, num_assets)

        state = np.concatenate([
            price_window.flatten(),
            self.weights
        ])
        return state.astype(np.float32)

    def _transaction_cost(self, w_old, w_new):
        """
        거래비용 모델 (논문 Eq. 3)
        μ_t = 1 - c_s × Σ|w_new - w_old|
        """
        turnover = np.sum(np.abs(w_new - w_old))
        mu = 1.0 - self.tc_rate * turnover
        return max(mu, 0.0)

    def _relative_entropy(self, w_new, w_old):
        """
        상대 엔트로피 (KL Divergence) - 논문 Eq. 6
        D_KL(w_new || w_old) = Σ w_new × ln(w_new / w_old)

        포트폴리오 급변 억제 역할
        """
        # 수치 안정성을 위한 클리핑
        w_new_c = np.clip(w_new, 1e-8, 1.0)
        w_old_c = np.clip(w_old, 1e-8, 1.0)
        return np.sum(w_new_c * np.log(w_new_c / w_old_c))

    def step(self, action):
        """
        환경 스텝 실행

        Args:
            action: 목표 포트폴리오 가중치 (total_assets,) — softmax 정규화 완료 상태

        Returns:
            next_state, reward, done, info
        """
        # 현재 가격 변화율
        t = self.step_idx + self.window
        if t >= len(self.returns):
            return self._get_state(), 0.0, True, {}

        v_t = self.returns[t]  # (num_assets,)

        # 전체 수익률 벡터 (현금 = 1.0)
        v_full = np.concatenate([[1.0], v_t])  # (total_assets,)

        # 거래비용
        w_old = self.weights
        w_new = action
        mu_t = self._transaction_cost(w_old, w_new)

        # 포트폴리오 수익률 (논문 Eq. 4-5)
        portfolio_return = mu_t * np.dot(v_full, w_new)

        # 로그 수익률 (논문 Eq. 5)
        log_return = np.log(max(portfolio_return, 1e-8))

        # 상대 엔트로피 페널티 (논문 Eq. 6)
        kl_penalty = self._relative_entropy(w_new, w_old)

        # 최종 보상 (논문 Eq. 6)
        # R_t = r_t - β × D_KL(w_new || w_old)
        reward = log_return - self.beta * kl_penalty

        # 포트폴리오 업데이트
        self.portfolio_value *= portfolio_return
        self.portfolio_values.append(self.portfolio_value)

        # 가중치 업데이트 (거래 후 시장 변동 반영)
        self.weights = (w_new * v_full) / max(np.dot(v_full, w_new), 1e-8)
        self.weight_history.append(self.weights.copy())

        # 다음 스텝
        self.step_idx += 1
        done = (self.step_idx + self.window) >= len(self.returns)
        next_state = self._get_state() if not done else self._get_state()

        info = {
            "portfolio_value": self.portfolio_value,
            "log_return": log_return,
            "kl_penalty": kl_penalty,
            "turnover": np.sum(np.abs(w_new - w_old)),
            "transaction_cost_factor": mu_t,
        }

        return next_state, reward, done, info

    def get_buy_and_hold_values(self):
        """Buy & Hold 벤치마크 (동일 비중)"""
        equal_weight = np.ones(self.num_assets) / self.num_assets
        values = [1.0]
        for t in range(self.window, len(self.returns)):
            v_t = self.returns[t]
            ret = np.dot(v_t, equal_weight)
            values.append(values[-1] * ret)
        return values[:len(self.portfolio_values)]
