"""
SDELP-DDPG Configuration
config.yaml에서 실험 설정을 읽어옵니다.
"""

import os
import yaml

# ── config.yaml 로드 ─────────────────────────────────
_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(_config_path, "r", encoding="utf-8") as f:
    _yaml = yaml.safe_load(f)

# ── 데이터 설정 (YAML에서 읽기) ──────────────────────
TICKERS = _yaml["tickers"]
TRAIN_START = _yaml["train_start"]
TRAIN_END = _yaml["train_end"]
TEST_START = _yaml["test_start"]
TEST_END = _yaml["test_end"]
NUM_EPISODES = _yaml["num_episodes"]

# ── 포트폴리오 환경 ──────────────────────────────────
WINDOW_SIZE = 50            # 관측 윈도우 (일)
NUM_ASSETS = len(TICKERS)
TOTAL_ASSETS = NUM_ASSETS + 1
TRANSACTION_COST = 0.0025   # 거래 비용 비율 (논문: 0.25%)
BETA = 0.05                 # 리스크 페널티 가중치
GAMMA = 0.99                # 할인율

# ── SDE Actor 네트워크 ───────────────────────────────
SDE_STEPS = 5               # Euler-Maruyama 이산화 스텝 수 (논문: m)
LEVY_ALPHA = 1.4            # alpha-stable 분포 파라미터
LEVY_BETA_PARAM = 0.0       # 대칭 분포 (beta = 0)
DRIFT_HIDDEN = 64           # Drift Net 은닉층 크기
DIFFUSION_HIDDEN = 64       # Diffusion Net 은닉층 크기
ATTENTION_HEADS = 4

# ── Critic 네트워크 ──────────────────────────────────
CRITIC_HIDDEN = 128

# ── DDPG 학습 ────────────────────────────────────────
BUFFER_SIZE = 50000         # 리플레이 버퍼 용량
BATCH_SIZE = 64             # 미니배치 크기
ACTOR_LR = 1e-4             # Actor 학습률
CRITIC_LR = 3e-4            # Critic 학습률
TAU = 0.001                 # 소프트 업데이트 계수
OU_THETA = 0.15
OU_SIGMA = 0.2

# ── 시각화 & 저장 ────────────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOT_INTERVAL = 10

# ── 디바이스 ─────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
