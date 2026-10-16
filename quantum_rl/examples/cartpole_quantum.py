from __future__ import annotations

import argparse

import numpy as np

from quantum_rl import CartPoleEnv, HybridAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid quantum agent on CartPole")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    print("=" * 60)
    print("  Quantum Reinforcement Learning — CartPole")
    print("=" * 60)

    env = CartPoleEnv(max_steps=200, seed=args.seed)
    agent = HybridAgent(n_qubits=4, n_layers=2, n_actions=2, obs_dim=4,
                        input_scale=env.obs_scale, seed=args.seed)
    print(f"\n{agent}\n")

    before = agent.evaluate(CartPoleEnv(max_steps=200, seed=123), n_episodes=10)
    result = agent.train(env, episodes=args.episodes, lr_quantum=0.1,
                         lr_classical=0.1, print_every=10)
    after = agent.evaluate(CartPoleEnv(max_steps=200, seed=123), n_episodes=10)

    print("\n  Greedy policy, 10 evaluation episodes:")
    print(f"    before training: {before['mean_reward']:6.1f} ± {before['std_reward']:.1f} steps")
    print(f"    after training : {after['mean_reward']:6.1f} ± {after['std_reward']:.1f} steps")

    window = max(1, len(result.episode_rewards) // 6)
    print("\n  Training curve (moving average):")
    for k in range(0, len(result.episode_rewards), window):
        avg = np.mean(result.episode_rewards[k:k + window])
        last = min(k + window, len(result.episode_rewards))
        print(f"    ep {k + 1:3d}-{last:3d}  {avg:6.1f}  {'▇' * int(avg / 5)}")

    demo_env = CartPoleEnv(max_steps=200, seed=7)
    obs = demo_env.reset()
    done = False
    while not done:
        obs, _, done, _ = demo_env.step(int(np.argmax(agent.get_action_probs(obs))))
    print("\n  Final frame of a greedy rollout:")
    print("  " + demo_env.render_ascii().replace("\n", "\n  "))
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
