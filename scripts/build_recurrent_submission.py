"""Stage a *recurrent net* Kaggle submission: numpy RecurrentNetAgent + its deck.

The canonical paper net (recurrent play LSTM + factored deck head) as a numpy-only,
self-contained bundle. Unlike ``build_net_submission.py`` (the older memoryless net
on a pinned meta deck), this:

- bundles :class:`RecurrentPolicyValueNet` and serves it with the stateful
  :class:`RecurrentNetAgent` (the play LSTM carries across decisions, reset at each
  game start when ``obs.select is None``);
- ships the **net's own greedy deck**, built offline from the same checkpoint
  (greedy decode is deterministic, so the bundled deck is exactly what the net would
  build at init) -- pass ``--deck`` to pin a meta deck instead.

numpy-only at runtime: ``src.deck`` / ``src.cards`` (pandas) are avoided by
precomputing the pool ids offline and vendoring stub ``deck_factored`` /
``deck_sample`` modules (only their import is needed for serving, never their deck
math). ``cg`` is bundled and the engine feature dict is rebuilt from it at startup.

  uv run python scripts/build_recurrent_submission.py \
      --ckpt data/paperosfp/main/paper_final.npz
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from src.deck import (  # noqa: E402
    build_pool,
    card_kind,
    legality_errors,
    load_deck_csv,
)
from src.net.cb import build_deck  # noqa: E402
from src.net.features import CardFeatures, load_engine_json  # noqa: E402
from src.net.recurrent_model import RecurrentPolicyValueNet  # noqa: E402

CG_SRC = ROOT / "data" / "sample_submission" / "cg"
ENGINE_JSON = ROOT / "data" / "bc" / "engine.json"
OUT = ROOT / "build" / "recurrent_submission"

# The closed numpy-only inference chain to vendor verbatim.
VENDOR = [
    "src/agents/base.py",
    "src/agents/recurrent_agent.py",
    "src/net/model.py",
    "src/net/recurrent_model.py",
    "src/net/embedding.py",
    "src/net/features.py",
    "src/net/encode.py",
    "src/net/nn.py",
]
PKG_INITS = ["src/__init__.py", "src/agents/__init__.py", "src/net/__init__.py"]

# Stubs for the two modules that pull src.deck (pandas). Serving never calls their
# deck math (the bundle plays a fixed deck), only imports them, so a constant +
# no-ops suffice. N_CATEGORIES must match src.net.deck_factored (the cat head width).
DECK_FACTORED_STUB = '''"""Stub: factored-deck math unused (fixed deck)."""

N_CATEGORIES = 3
CAT_POKEMON, CAT_TRAINER, CAT_ENERGY = 0, 1, 2


def category_of_rows(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201, ARG001
    raise NotImplementedError


def factored_pick(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201, ARG001
    raise NotImplementedError


def factored_logp(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201, ARG001
    raise NotImplementedError
'''

DECK_SAMPLE_STUB = '''"""Stub: deck sampling unused (fixed deck)."""


def sample_deck_with_logp(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201, ARG001
    raise NotImplementedError
'''

MAIN_PY = '''"""Self-contained Kaggle submission: numpy recurrent net + its own deck.

Loads the trained .npz, rebuilds the engine feature dict from the bundled cg, and
plays a fixed (pre-built) deck with the recurrent play policy. Stateful: the play
LSTM is reset at each game start (obs.select is None). Never crashes -- any failure
returns a legal fallback (and the initial deck selection returns the bundled ids).

kaggle_environments exec-s this source, so ``__file__`` may be undefined -- resolve
files relative to CWD first, then the well-known agent dir.
"""

import json
import os
import sys

_AGENT_DIR = "/kaggle_simulations/agent"
for p in (os.getcwd(), _AGENT_DIR):
    if p and p not in sys.path:
        sys.path.insert(0, p)


def _path(name):
    return name if os.path.exists(name) else os.path.join(_AGENT_DIR, name)


def _read_deck():
    with open(_path("deck.csv")) as f:
        return [int(x) for x in f.read().split() if x.strip()]


def _load_engine():
    from cg.api import all_attack, all_card_data

    attacks = {
        a.attackId: {"dmg": int(a.damage), "cost": [int(e) for e in a.energies]}
        for a in all_attack()
    }
    cards = {
        c.cardId: {
            "hp": int(c.hp), "retreat": int(c.retreatCost), "type": int(c.energyType),
            "weak": None if c.weakness is None else int(c.weakness),
            "ex": bool(c.ex), "mega": bool(c.megaEx), "basic": bool(c.basic),
            "ctype": int(c.cardType), "attacks": list(c.attacks),
        }
        for c in all_card_data()
    }
    return {"attacks": attacks, "cards": cards}


class _PoolStub:
    """Minimal pool: CardEmbeddingIndex only calls .ids()."""

    def __init__(self, ids):
        self._ids = list(ids)

    def ids(self):
        return self._ids


DECK = _read_deck()


def _build_agent():
    from src.agents.recurrent_agent import RecurrentNetAgent
    from src.net.recurrent_model import RecurrentPolicyValueNet

    engine = _load_engine()
    net = RecurrentPolicyValueNet.load(_path("weights.npz"))
    with open(_path("pool_ids.json")) as f:
        pool_ids = json.load(f)
    return RecurrentNetAgent(
        DECK, engine, net=net, cb_pool=_PoolStub(pool_ids),
        build_deck_from_net=False, temperature=0.0,
    )


try:
    _AGENT = _build_agent()
except Exception:  # never crash at import; fall back to deck-only behaviour
    _AGENT = None


def _legal_fallback(select):
    return list(range(int(select.get("maxCount", 0))))


def agent(obs_dict):
    """Kaggle contract: obs dict -> option indices (deck ids at init)."""
    try:
        if obs_dict.get("select") is None:
            if _AGENT is not None:
                _AGENT.reset(0)  # new game -> reset the play LSTM hidden state
            return list(DECK)
        if _AGENT is not None:
            return _AGENT(obs_dict)
        return _legal_fallback(obs_dict.get("select") or {})
    except Exception:
        try:
            return _legal_fallback(obs_dict.get("select") or {})
        except Exception:
            return [0]
'''


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage the recurrent net submission")
    ap.add_argument(
        "--ckpt", type=Path, default=ROOT / "data/paperosfp/main/paper_final.npz",
    )
    ap.add_argument(
        "--deck", type=Path, default=None,
        help="pin this deck instead of the net's own greedy deck",
    )
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    pool = build_pool()
    pool_ids = sorted(pool.ids())
    net = RecurrentPolicyValueNet.load(args.ckpt)
    n_pool = net.params["cb_embed"].shape[0] - 1
    if n_pool != len(pool_ids):
        msg = f"pool/embed mismatch: net n_pool={n_pool} pool={len(pool_ids)}"
        raise SystemExit(msg)

    # The deck: the net's own greedy build (default) or a pinned meta deck.
    if args.deck is not None:
        deck = load_deck_csv(args.deck)
        deck_src = args.deck.name
    else:
        feats = CardFeatures(load_engine_json(ENGINE_JSON))
        deck = build_deck(net, pool, feats)  # greedy, deterministic
        deck_src = "net greedy"
    errors = legality_errors(deck, pool)
    if errors:
        raise SystemExit(f"deck ({deck_src}) is illegal: {errors}")
    if not CG_SRC.exists():
        raise SystemExit(f"engine not found at {CG_SRC} (run scripts/download_data.sh)")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    for rel in VENDOR:
        dst = args.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dst)
    for rel in PKG_INITS:
        (args.out / rel).write_text("")  # empty package init (skip REGISTRY etc.)
    (args.out / "src/net/deck_factored.py").write_text(DECK_FACTORED_STUB)
    (args.out / "src/net/deck_sample.py").write_text(DECK_SAMPLE_STUB)

    (args.out / "main.py").write_text(MAIN_PY)
    (args.out / "deck.csv").write_text(" ".join(str(c) for c in deck) + "\n")
    (args.out / "pool_ids.json").write_text(json.dumps(pool_ids))
    shutil.copy(args.ckpt, args.out / "weights.npz")
    shutil.copytree(
        CG_SRC, args.out / "cg", ignore=shutil.ignore_patterns("__pycache__"),
    )

    comp = Counter(card_kind(pool, c) for c in deck)
    print(f"staged {args.out}")
    print(f"  ckpt={args.ckpt.name}  deck={deck_src} ({len(deck)} cards, legal)")
    print(f"  deck comp: {dict(comp)}  distinct={len(set(deck))}")
    print(f"  pool_ids={len(pool_ids)}  n_pool={n_pool} (match)")
    print("next: package + submit:")
    print(f"  tar -czf build/recurrent_submission.tar.gz -C {args.out} .")
    print("  kaggle competitions submit -c pokemon-tcg-ai-battle "
          "-f build/recurrent_submission.tar.gz -m 'recurrent paper net + own deck'")


if __name__ == "__main__":
    main()
