# SDELP-DDPG: Portfolio Management with Lévy Process

> 강화학습 학회 Session 3 | AI-Assisted Paper Implementation

## 📁 프로젝트 구조

```
sdelp-ddpg/
├── config.py                    # 하이퍼파라미터 설정
├── environment.py               # 포트폴리오 MDP 환경
├── networks.py                  # SDE Actor + Critic 네트워크
├── agent.py                     # DDPG 에이전트 (Algorithm 1)
├── train.py                     # 학습 루프 + 시각화
├── compare_levy_gaussian.py     # Lévy vs Gaussian 비교 시각화
├── run_all.py                   # 전체 파이프라인 원클릭 실행
└── outputs/                     # 결과 저장 폴더
    ├── training_results.png     # 학습 결과 4-panel 그래프
    ├── losses.png               # Actor/Critic Loss 그래프
    ├── levy_vs_gaussian.png     # 노이즈 분포 비교
    ├── action_distribution.png  # 행동 분포 비교
    └── best_model.pt            # 최고 성과 모델
```

## 🚀 실행 방법

### 전체 실행 (권장)
```bash
cd sdelp-ddpg
python run_all.py
```

### 개별 실행
```bash
# Lévy vs Gaussian 시각화만
python compare_levy_gaussian.py

# 학습만
python train.py
```

## 📦 필요 패키지

```bash
pip install torch numpy matplotlib yfinance scipy pandas
```

## 🏗️ 아키텍처

```
State (가격 변화율 + 포트폴리오 가중치)
  │
  ▼
┌─────────────── SDE Actor ───────────────┐
│  Init-Net → a₀                          │
│  for k = 1..K:                          │
│    a_{k+1} = a_k + u(a_k,s)Δt + σΔL_α  │
│           Drift     Diffusion   Lévy    │
│  Attention(trajectory) → best action    │
│  Softmax → portfolio weights            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
            Action (w_cash, w_BTC, w_ETH)
                   │
     ┌─────────────┴──────────────┐
     ▼                            ▼
  Environment                  Critic
  r = log_ret - β·KL        Q(s, a) → value
     │                            │
     └─────── Replay Buffer ──────┘
                   │
              DDPG Update
```

## 📊 핵심 수식

| 수식 | 설명 | 파일 |
|------|------|------|
| `R = r_t - β·D_KL(w_new‖w_old)` | 보상 함수 | environment.py |
| `a_{k+1} = a_k + u·Δt + σ·ΔL_α` | Euler-Maruyama | networks.py |
| `∇θJ ≈ -E[Q(s, π(s))]` | Policy Gradient | agent.py |
| `θ' ← τθ + (1-τ)θ'` | Soft Update | agent.py |

## 📖 논문

**SDELP-DDPG**: Stochastic Differential Equations with Lévy Processes-Driven
Deep Deterministic Policy Gradient for Portfolio Management

*Expert Systems With Applications (2025)*
