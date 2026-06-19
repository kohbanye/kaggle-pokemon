"""Smoke test for the cg simulator (Linux x86-64 only).

Loads the compiled engine, prints a couple of cards from `all_card_data()`, and
plays one fully-random battle to confirm the .so works end to end. Run inside the
Docker image (see README): ``python scripts/sim_smoke.py``.
"""

import random
import sys
from pathlib import Path

# The engine package lives under data/sample_submission/cg (gitignored download).
CG_PARENT = Path(__file__).resolve().parent.parent / "data" / "sample_submission"
sys.path.insert(0, str(CG_PARENT))

from cg.api import all_card_data, to_observation_class  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402


def random_agent(obs_dict: dict, deck: list[int]) -> list[int]:
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        return deck
    return random.sample(range(len(obs.select.option)), obs.select.maxCount)


def main() -> None:
    cards = all_card_data()
    print(f"all_card_data(): {len(cards)} cards. e.g. {cards[0].name!r}, {cards[1].name!r}")  # noqa: E501

    deck_path = CG_PARENT / "deck.csv"
    deck = [int(x) for x in deck_path.read_text().split() if x.strip()]
    assert len(deck) == 60, f"deck must be 60 cards, got {len(deck)}"

    obs, start = battle_start(deck, list(deck))
    if obs is None:
        raise SystemExit(f"battle failed to start: errorType={start.errorType}")

    steps = 0
    while obs["current"] is None or obs["current"].get("result", -1) == -1:
        select = random_agent(obs, deck)
        obs = battle_select(select)
        steps += 1
        if steps > 5000:
            raise SystemExit("battle did not terminate in 5000 steps")

    result = obs["current"]["result"]
    print(f"battle finished in {steps} selections. winner = player {result}")
    battle_finish()
    print("OK: simulator works.")


if __name__ == "__main__":
    main()
