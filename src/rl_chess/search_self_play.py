from __future__ import annotations

import random

import chess

from rl_chess.env import ChessEnv
from rl_chess.mcts import MCTS
from rl_chess.replay import SearchTrainingExample
from rl_chess.self_play import assign_episode_returns
from rl_chess.replay import Transition


def collect_search_episode(
    env: ChessEnv,
    mcts: MCTS,
    max_plies: int = 200,
    seed: int | None = None,
) -> list[SearchTrainingExample]:
    """Collect one MCTS-guided self-play game as policy/value examples."""

    if max_plies <= 0:
        raise ValueError("max_plies must be positive")

    rng = random.Random(seed)
    obs = env.reset()
    examples: list[SearchTrainingExample] = []
    transitions: list[Transition] = []
    final_reward = 0.0

    for _ply in range(max_plies):
        board_before = env.board.copy(stack=False)
        if board_before.is_game_over(claim_draw=True):
            break

        policy_target = mcts.search_policy(board_before, rng=rng)
        if not policy_target:
            break
        move_uci = max(policy_target, key=lambda uci: (policy_target[uci], uci))
        move = chess.Move.from_uci(move_uci)

        examples.append(
            SearchTrainingExample(
                state_ascii=obs.board_ascii,
                legal_moves=obs.legal_moves,
                policy_target=policy_target,
                value_target=None,
            )
        )

        next_obs, reward, done, info = env.step(move)
        transitions.append(
            Transition(
                state_ascii=obs.board_ascii,
                action_uci=move.uci(),
                player=board_before.turn,
                reward=reward if done else 0.0,
                done=done,
                next_state_ascii=next_obs.board_ascii,
                result=info["result"],
            )
        )
        obs = next_obs

        if done:
            final_reward = reward
            break

    assigned = assign_episode_returns(transitions, final_reward)
    return [
        SearchTrainingExample(
            state_ascii=example.state_ascii,
            legal_moves=example.legal_moves,
            policy_target=example.policy_target,
            value_target=transition.return_,
        )
        for example, transition in zip(examples, assigned)
    ]
