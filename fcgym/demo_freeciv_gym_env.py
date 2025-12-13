"""
Demo: Freeciv Gymnasium environment backed by fcgym.

Usage (from repo root):
  python freeciv/fcgym/demo_freeciv_gym_env.py --steps 50 --seed 123

What it shows:
- `env.reset()` returns (obs, info) where obs is a Dict of fixed-shape arrays.
- Actions are chosen by sampling indices where `info["action_mask"] == 1`.
- `info["legal_actions"][i]` encodes one action as:
    [action_type, actor_slot, target, sub_target]

Notes:
- `fcgym` uses global Freeciv state, so run only 1 env per process.
  For parallelism, use subprocess-based vector envs (e.g. AsyncVectorEnv).
"""

from __future__ import annotations

import argparse

import numpy as np

from freeciv_gym_env import FcActionType, FreecivGymEnv


def _format_action(env: FreecivGymEnv, action_index: int, info: dict) -> str:
    row = info["legal_actions"][action_index]
    action_type_id = int(row[0])
    try:
        action_type_name = FcActionType(action_type_id).name
    except ValueError:
        action_type_name = str(action_type_id)

    decoded = env._decode_action(action_index)
    return (
        f"idx={action_index} legal_row={row.tolist()} "
        f"type={action_type_name} decoded={decoded}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo for FreecivGymEnv (fcgym).")
    parser.add_argument("--ruleset", default="civ2civ3")
    parser.add_argument("--map-width", type=int, default=40)
    parser.add_argument("--map-height", type=int, default=40)
    parser.add_argument("--num-ai-players", type=int, default=2)
    parser.add_argument("--ai-skill-level", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fog-of-war", dest="fog_of_war", action="store_true", default=True)
    parser.add_argument("--no-fog-of-war", dest="fog_of_war", action="store_false")
    parser.add_argument("--max-legal-actions", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    env = FreecivGymEnv(
        ruleset=args.ruleset,
        map_width=args.map_width,
        map_height=args.map_height,
        num_ai_players=args.num_ai_players,
        ai_skill_level=args.ai_skill_level,
        fog_of_war=args.fog_of_war,
        max_legal_actions=args.max_legal_actions,
        render_mode=None,
    )

    try:
        obs, info = env.reset(seed=args.seed)

        print("Observation keys:", list(obs.keys()))
        for key, value in obs.items():
            print(f"  {key}: shape={value.shape} dtype={value.dtype}")

        print("\nAction info:")
        print("  action_space:", env.action_space)
        print("  num_legal_actions:", info["num_legal_actions"])
        print("  action_mask shape:", info["action_mask"].shape, "dtype:", info["action_mask"].dtype)
        print("  legal_actions shape:", info["legal_actions"].shape, "dtype:", info["legal_actions"].dtype)

        num_units = int(obs["unit_mask"].sum())
        num_cities = int(obs["city_mask"].sum())
        print(f"\nControlled entities: units={num_units}, cities={num_cities}")

        for t in range(args.steps):
            valid = np.flatnonzero(info["action_mask"] > 0)
            if valid.size == 0:
                print("No valid actions; stopping.")
                break

            action_index = int(rng.choice(valid))
            print(f"\nStep {t}: turn={info.get('turn')} " + _format_action(env, action_index, info))

            obs, reward, terminated, truncated, info = env.step(action_index)
            print(f"  reward={reward:.4f} terminated={terminated} truncated={truncated} num_legal={info['num_legal_actions']}")

            if terminated or truncated:
                print("Episode ended.")
                break

        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
