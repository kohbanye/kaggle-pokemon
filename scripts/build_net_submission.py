"""Stage a *net* Kaggle submission bundle: numpy NetAgent + weights + fixed deck.

Unlike ``build_submission.py`` (self-contained greedy), this bundles the trained
policy net for inference. The net plays a **fixed** deck (its CB deck head is the
known-broken part -- it loses ~100% vs greedy -- so we pin a strong deck and use
only the play policy). The bundle is numpy-only at runtime: ``src.deck`` /
``src.cards`` (pandas) are avoided by precomputing the pool ids offline and
vendoring a stub ``cb.py`` so ``net_agent``'s ``build_deck`` import never pulls
pandas. ``cg`` is bundled and the engine feature dict is rebuilt from it at start.

  uv run python scripts/build_net_submission.py \
      --ckpt data/paperosfp/main/paperiter_163.npz --deck decklists/psychic_aggro.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "sample_submission"))

from src.deck import build_pool, legality_errors, load_deck_csv  # noqa: E402
from src.net.model import PolicyValueNet  # noqa: E402

CG_SRC = ROOT / "data" / "sample_submission" / "cg"
OUT = ROOT / "build" / "net_submission"

# src files to vendor (the closed numpy-only inference chain). cb.py is replaced
# by a stub below so the build_deck import does not drag in src.deck -> pandas.
VENDOR = [
    "src/agents/base.py",
    "src/agents/net_agent.py",
    "src/net/model.py",
    "src/net/embedding.py",
    "src/net/features.py",
    "src/net/encode.py",
    "src/net/nn.py",
]
PKG_INITS = ["src/__init__.py", "src/agents/__init__.py", "src/net/__init__.py"]

CB_STUB = '''"""Stub: deck building is disabled in the submission bundle (fixed deck).

The real cb.py imports src.deck (pandas). The bundle pins a fixed deck and never
calls build_deck, so this stub just satisfies net_agent's import.
"""


def build_deck(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201, ARG001
    raise NotImplementedError("deck building disabled in submission bundle")
'''

MAIN_PY = '''"""Self-contained Kaggle net submission: numpy PolicyValueNet, fixed deck.

Loads the trained .npz, rebuilds the engine feature dict from the bundled cg, and
plays a fixed deck with the net's play policy. Never crashes: any failure returns
a legal fallback (and the initial deck selection returns the bundled 60 ids).

NOTE: kaggle_environments loads the agent by exec-ing this source, so ``__file__``
may be undefined -- do NOT reference it. Resolve files like the official reference:
relative (CWD == agent dir) first, then ``/kaggle_simulations/agent``.
"""

import json
import os
import sys

_AGENT_DIR = "/kaggle_simulations/agent"
# Make the vendored ``src`` package importable in both layouts (CWD == agent dir,
# or files only under the well-known agent dir). No ``__file__`` dependency.
for p in (os.getcwd(), _AGENT_DIR):
    if p and p not in sys.path:
        sys.path.insert(0, p)


def _path(name):
    if os.path.exists(name):
        return name
    return os.path.join(_AGENT_DIR, name)


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
    from src.agents.net_agent import NetAgent
    from src.net.model import PolicyValueNet

    engine = _load_engine()
    net = PolicyValueNet.load(_path("weights.npz"))
    with open(_path("pool_ids.json")) as f:
        pool_ids = json.load(f)
    return NetAgent(
        DECK, engine=engine, net=net,
        cb_pool=_PoolStub(pool_ids), build_deck_from_net=False,
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
    ap = argparse.ArgumentParser(description="Stage the net submission bundle")
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "data/paperosfp/main/paperiter_163.npz")
    ap.add_argument("--deck", type=Path, default=ROOT / "decklists/psychic_aggro.csv")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    deck = load_deck_csv(args.deck)
    errors = legality_errors(deck, build_pool())
    if errors:
        raise SystemExit(f"deck {args.deck} is illegal: {errors}")
    if not CG_SRC.exists():
        raise SystemExit(f"engine not found at {CG_SRC}")

    # pool ids + net/pool size sanity (the embedding table must match the pool).
    pool_ids = sorted(build_pool().ids())
    net = PolicyValueNet.load(args.ckpt)
    n_pool = net.params["cb_embed"].shape[0] - 1
    if n_pool != len(pool_ids):
        raise SystemExit(
            f"pool/embed mismatch: net n_pool={n_pool} pool={len(pool_ids)}")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    # vendored src tree
    for rel in VENDOR:
        dst = args.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, dst)
    for rel in PKG_INITS:
        (args.out / rel).write_text("")  # empty package init (skip REGISTRY etc.)
    (args.out / "src/net/cb.py").write_text(CB_STUB)

    # artifacts
    (args.out / "main.py").write_text(MAIN_PY)
    (args.out / "deck.csv").write_text(" ".join(str(c) for c in deck) + "\n")
    (args.out / "pool_ids.json").write_text(json.dumps(pool_ids))
    shutil.copy(args.ckpt, args.out / "weights.npz")
    shutil.copytree(CG_SRC, args.out / "cg",
                    ignore=shutil.ignore_patterns("__pycache__"))

    print(f"staged {args.out}")
    print(f"  ckpt={args.ckpt.name}  deck={args.deck.name} ({len(deck)} cards, legal)")
    print(f"  pool_ids={len(pool_ids)}  n_pool={n_pool} (match)")
    print("next: validate, then package + submit:")
    print(f"  tar -czf build/net_submission.tar.gz -C {args.out} .")
    print("  kaggle competitions submit -c pokemon-tcg-ai-battle "
          "-f build/net_submission.tar.gz -m 'net iter163 + psychic_aggro'")


if __name__ == "__main__":
    main()
