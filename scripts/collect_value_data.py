"""Increment-1 value-model DATA COLLECTION (Linux x86-64 native, imports ``cg``).

Plays N games of the bespoke ArchaludonAgent against a MIX of opponents
(the bespoke AlakazamAgent + ``greedy_plus`` on a handful of heldout2 ladder
decks), slot-swapped, and logs -- for every decision taken on Archaludon's turn
-- a compact fixed-length STATE feature vector encoded from the raw observation
dict (from Archaludon's point of view). At game end EVERY Archaludon-side sample
of that game is labelled with the game outcome ``z`` (1 = Archaludon won, 0 =
lost). Saved to a ``.npz`` for ``train_value_net.py``.

This increment ONLY measures whether a learned V(state)->P(win) can predict the
outcome better than trivial baselines; nothing here is wired into an agent.

Run:  uv run python scripts/collect_value_data.py --games 400
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from scripts.run_eval import load_engine_data  # noqa: E402
from src.agents import build_agent  # noqa: E402
from src.agents.alakazam_pilot import AlakazamAgent  # noqa: E402
from src.agents.archaludon_pilot import ArchaludonAgent  # noqa: E402
from src.agents.base import is_legal, legal_fallback  # noqa: E402

DECK_REQUEST = {"select": None, "logs": [], "current": None}
MAX_SELECTIONS = 5000
ARCHALUDON_EX_ID = 190

ARCH_DECK = ROOT / "decklists" / "archaludon_netdeck.csv"
ALA_DECK = ROOT / "decklists" / "alakazam_netdeck.csv"
HELDOUT_DECKS = [
    ROOT / "decklists" / "heldout2" / name
    for name in (
        "lad00_P_evo_1pz_e790.csv",
        "lad03_G_evo_1pz_e741.csv",
        "lad04_F_evo_mid_e839.csv",
        "lad07_M_evo_mid_e711.csv",
        "lad09_P_basic_big_e589.csv",
    )
]

# --- compact self-contained state encoder (raw obs dict -> fixed vector) ------
# Deterministic; any missing/malformed field degrades to 0. Feature order is
# documented by FEATURE_NAMES so the training side can interpret columns.
FEATURE_NAMES = [
    "my_prize/6", "my_deck/60", "my_hand/10", "my_bench/5", "my_active_present",
    "my_active_hpfrac", "my_active_hp/400", "my_active_energy/6",
    "my_bench_energy/12", "my_ex_count/4", "my_mega_count/2", "my_has_archaludon",
    "opp_prize/6", "opp_deck/60", "opp_hand/10", "opp_bench/5", "opp_active_present",
    "opp_active_hpfrac", "opp_active_hp/400", "opp_active_energy/6",
    "opp_bench_energy/12", "opp_ex_count/4", "opp_mega_count/2",
    "turn/20", "prize_diff/6", "my_total_energy/12", "opp_total_energy/12",
]
FEAT_DIM = len(FEATURE_NAMES)


def _pk_list(player: dict, key: str) -> list:
    spot = player.get(key) or []
    return [pk for pk in spot if pk is not None]


def _player_feats(
    player: dict, ex_ids: set[int], mega_ids: set[int],
) -> tuple[list[float], int]:
    """Return (11 or 12 per-player features, total energy in play)."""
    active_list = _pk_list(player, "active")
    active = active_list[0] if active_list else None
    if active is not None:
        max_hp = active.get("maxHp") or 0
        hp = active.get("hp", 0) or 0
        hp_frac = hp / max_hp if max_hp > 0 else 0.0
        a_present, a_hpfrac, a_hp = 1.0, hp_frac, hp / 400.0
        a_energy = len(active.get("energies") or [])
    else:
        a_present = a_hpfrac = a_hp = 0.0
        a_energy = 0

    bench = _pk_list(player, "bench")
    bench_energy = sum(len(pk.get("energies") or []) for pk in bench)
    all_pk = ([active] if active is not None else []) + bench
    ex_count = sum(1 for pk in all_pk if pk.get("id") in ex_ids)
    mega_count = sum(1 for pk in all_pk if pk.get("id") in mega_ids)
    total_energy = a_energy + bench_energy

    feats = [
        len(player.get("prize") or []) / 6.0,
        player.get("deckCount", 0) / 60.0,
        player.get("handCount", 0) / 10.0,
        len(bench) / 5.0,
        a_present,
        a_hpfrac,
        a_hp,
        a_energy / 6.0,
        bench_energy / 12.0,
        ex_count / 4.0,
        mega_count / 2.0,
    ]
    return feats, total_energy


def encode_state(
    cur: dict, us: int, ex_ids: set[int], mega_ids: set[int],
) -> np.ndarray:
    players = cur.get("players") or []
    if len(players) < 2:
        return np.zeros(FEAT_DIM, dtype=np.float32)
    me, opp = players[us], players[1 - us]
    me_feats, me_energy = _player_feats(me, ex_ids, mega_ids)
    # my_has_archaludon
    my_pk = _pk_list(me, "active") + _pk_list(me, "bench")
    has_arch = float(any(pk.get("id") == ARCHALUDON_EX_ID for pk in my_pk))
    me_feats.append(has_arch)

    opp_feats, opp_energy = _player_feats(opp, ex_ids, mega_ids)

    my_prize = len(me.get("prize") or [])
    opp_prize = len(opp.get("prize") or [])
    glob = [
        cur.get("turn", 0) / 20.0,
        (opp_prize - my_prize) / 6.0,  # +ve = Archaludon ahead on prizes
        me_energy / 12.0,
        opp_energy / 12.0,
    ]
    return np.asarray(me_feats + opp_feats + glob, dtype=np.float32)


# --- one game -----------------------------------------------------------------
def play_and_record(
    arch: ArchaludonAgent,
    opp,
    *,
    arch_is_p0: bool,
    seed: int,
    ex_ids: set[int],
    mega_ids: set[int],
) -> tuple[float, list[tuple[np.ndarray, int, int, int]]] | None:
    """Play one game; return (z, [(feat, turn, my_prize, opp_prize), ...]) or None
    if aborted / drawn / no samples."""
    arch.reset(seed)
    opp.reset(seed)
    agents = (arch, opp) if arch_is_p0 else (opp, arch)
    arch_slot = 0 if arch_is_p0 else 1
    d0 = agents[0](DECK_REQUEST)
    d1 = agents[1](DECK_REQUEST)
    obs, _start = battle_start(d0, d1)
    if obs is None:
        return None

    records: list[tuple[np.ndarray, int, int, int]] = []
    selections = 0
    winner = -1
    while True:
        cur = obs["current"]
        if cur is not None and cur.get("result", -1) != -1:
            winner = cur["result"]
            break
        yidx = 0 if cur is None else int(cur.get("yourIndex", 0))
        select = obs["select"]
        try:
            choice = agents[yidx](obs)
        except Exception:  # noqa: BLE001 - never crash a match
            choice = None
        if not is_legal(choice, select):
            choice = legal_fallback(select)

        if cur is not None and yidx == arch_slot:
            players = cur.get("players") or []
            if len(players) >= 2:
                feat = encode_state(cur, arch_slot, ex_ids, mega_ids)
                my_p = len(players[arch_slot].get("prize") or [])
                opp_p = len(players[1 - arch_slot].get("prize") or [])
                records.append((feat, int(cur.get("turn", 0)), my_p, opp_p))

        obs = battle_select(choice)
        selections += 1
        if selections >= MAX_SELECTIONS:
            winner = -1
            break

    battle_finish()
    if winner == -1 or not records:
        return None
    z = 1.0 if winner == arch_slot else 0.0
    return z, records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "value_ds.npz")
    args = ap.parse_args()

    engine = load_engine_data()
    cards = engine["cards"]
    ex_ids = {cid for cid, c in cards.items() if c.get("ex")}
    mega_ids = {cid for cid, c in cards.items() if c.get("mega")}
    print(f"engine: {len(ex_ids)} ex ids, {len(mega_ids)} mega ids")

    arch_deck = [int(x) for x in ARCH_DECK.read_text().split() if x.strip()]
    ala_deck = [int(x) for x in ALA_DECK.read_text().split() if x.strip()]

    arch = ArchaludonAgent(arch_deck, engine)

    # Opponent pool: bespoke Alakazam + greedy_plus on several heldout2 decks.
    ala = AlakazamAgent(ala_deck, engine)
    opponents: list[tuple[str, object]] = [("alakazam", ala)]
    for p in HELDOUT_DECKS:
        deck = [int(x) for x in p.read_text().split() if x.strip()]
        opponents.append((f"gp:{p.stem}", build_agent("greedy_plus", deck, engine)))
    print(f"opponents ({len(opponents)}): {[n for n, _ in opponents]}")

    feats_all: list[np.ndarray] = []
    turns_all: list[int] = []
    myp_all: list[int] = []
    oppp_all: list[int] = []
    labels_all: list[float] = []
    game_ids_all: list[int] = []

    gid = 0
    wins = 0
    kept_games = 0
    t0 = time.perf_counter()
    for g in range(args.games):
        _opp_name, opp_agent = opponents[g % len(opponents)]
        arch_is_p0 = (g % 2 == 0)
        out = play_and_record(
            arch, opp_agent, arch_is_p0=arch_is_p0, seed=args.seed + g,
            ex_ids=ex_ids, mega_ids=mega_ids,
        )
        if out is None:
            continue
        z, records = out
        wins += int(z == 1.0)
        kept_games += 1
        for feat, turn, my_p, opp_p in records:
            feats_all.append(feat)
            turns_all.append(turn)
            myp_all.append(my_p)
            oppp_all.append(opp_p)
            labels_all.append(z)
            game_ids_all.append(gid)
        gid += 1
        if (g + 1) % 50 == 0:
            rate = wins / kept_games if kept_games else 0.0
            print(f"  [{g + 1}/{args.games}] kept={kept_games} "
                  f"arch_winrate={rate:.3f} samples={len(feats_all)} "
                  f"({time.perf_counter() - t0:.0f}s)")

    features = np.asarray(feats_all, dtype=np.float32)
    labels = np.asarray(labels_all, dtype=np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=features,
        labels=labels,
        turns=np.asarray(turns_all, dtype=np.int32),
        my_prize=np.asarray(myp_all, dtype=np.int32),
        opp_prize=np.asarray(oppp_all, dtype=np.int32),
        game_ids=np.asarray(game_ids_all, dtype=np.int32),
        feature_names=np.asarray(FEATURE_NAMES),
    )

    pos = int(labels.sum())
    n = len(labels)
    print("\n== collection summary ==")
    print(f"games played={args.games} kept(decisive)={kept_games} "
          f"arch winrate={wins / kept_games:.3f}")
    print(f"samples={n}  positives(win)={pos} ({pos / n:.3f})  "
          f"negatives(loss)={n - pos} ({1 - pos / n:.3f})")
    print(f"feature dim={features.shape[1]}  wall={time.perf_counter() - t0:.0f}s")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
