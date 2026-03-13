"""
SDELP-DDPG 전체 파이프라인 실행
세션 전 리허설 또는 세션 중 Plan B로 사용

실행: python run_all.py
"""

import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("╔" + "═" * 58 + "╗")
    print("║  SDELP-DDPG: Full Pipeline Execution                     ║")
    print("║  강화학습 학회 Session 3                                   ║")
    print("╚" + "═" * 58 + "╝")

    # ── Step 1: Lévy vs Gaussian 비교 시각화 ──
    print("\n" + "=" * 60)
    print("  STEP 1: Lévy vs Gaussian 비교 시각화")
    print("=" * 60)
    from compare_levy_gaussian import compare_noise_distributions, compare_action_generation
    compare_noise_distributions()
    compare_action_generation()

    # ── Step 2: SDELP-DDPG 학습 ──
    print("\n" + "=" * 60)
    print("  STEP 2: SDELP-DDPG 학습 실행")
    print("=" * 60)
    from train import train
    agent, env, rewards, values = train()

    # ── 최종 요약 ──
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  ✅ ALL DONE — 생성된 파일 목록                           ║")
    print("╠" + "═" * 58 + "╣")
    output_dir = "outputs"
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"║  📊 {output_dir}/{f:<35} ({size//1024:>4} KB) ║")
    print("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()
