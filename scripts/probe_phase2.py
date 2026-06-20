"""Phase-2 grounding probe (Docker/Linux only): dump the real shapes the
heuristic must read.

  1. CardData for every distinct deck card id (+ basic energy) and the Attack
     records they reference -> confirms hp / weakness / energyType / ex / attacks
     / damage / cost field names and value encodings.
  2. A greedy-vs-greedy game, logging the first few *interesting* MAIN selections
     (>=2 option types, or any ATTACH/EVOLVE/PLAY/ATTACK) with the full option
     dicts, plus the opponent's in-play Pokemon (to verify we can see their hp /
     energies for KO-risk).

Run: docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
         python scripts/probe_phase2.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CG_PARENT = ROOT / "data" / "sample_submission"
sys.path.insert(0, str(CG_PARENT))

from cg.api import all_attack, all_card_data  # noqa: E402
from cg.game import battle_finish, battle_select, battle_start  # noqa: E402

from src.agents import build_agent  # noqa: E402
from src.agents.base import is_legal, legal_fallback  # noqa: E402


def read_deck() -> list[int]:
    text = (CG_PARENT / "deck.csv").read_text()
    return [int(x) for x in text.split() if x.strip()]


def dump_card_data(deck: list[int]) -> None:
    cards = {c.cardId: c for c in all_card_data()}
    attacks = {a.attackId: a for a in all_attack()}
    deck_ids = sorted(set(deck))
    print(f"== deck distinct card ids ({len(deck_ids)}): {deck_ids} ==")
    for cid in deck_ids:
        c = cards.get(cid)
        if c is None:
            print(f"  id {cid}: <not found>")
            continue
        print(f"  id={cid} name={c.name!r} type={int(c.cardType)} "
              f"hp={c.hp} retreat={c.retreatCost} energyType={c.energyType} "
              f"weak={c.weakness} resist={c.resistance} basic={c.basic} "
              f"s1={c.stage1} s2={c.stage2} ex={c.ex} mega={c.megaEx} "
              f"evolvesFrom={c.evolvesFrom!r} attacks={c.attacks}")
        for aid in c.attacks:
            a = attacks.get(aid)
            if a:
                print(f"      attack {aid}: {a.name!r} dmg={a.damage} "
                      f"energies={a.energies}")


def summarize_pokemon(p: dict | None) -> str:
    if p is None:
        return "None(facedown)"
    return (f"id={p.get('id')} hp={p.get('hp')}/{p.get('maxHp')} "
            f"energies={p.get('energies')} "
            f"nTools={len(p.get('tools') or [])}")


def dump_game(deck: list[int]) -> None:
    agent = build_agent("greedy", deck, {a.attackId: a.damage for a in all_attack()})
    deck_req = {"select": None, "logs": [], "current": None}
    agent(deck_req)

    obs, start = battle_start(deck, list(deck))
    if obs is None:
        print(f"battle failed: {start.errorType}")
        return

    shown = 0
    type_ctr: Counter = Counter()
    steps = 0
    while True:
        cur = obs["current"]
        if cur is not None and cur.get("result", -1) != -1:
            print(f"== game over: result={cur['result']} turn={cur.get('turn')} ==")
            break
        sel = obs["select"]
        if sel is not None:
            opt_types = [int(o["type"]) for o in sel["option"]]
            type_ctr.update(opt_types)
            distinct = set(opt_types)
            interesting = sel.get("type") == 0 and (
                len(distinct) >= 2 or distinct & {7, 8, 9, 13, 12}
            )
            if interesting and shown < 8:
                shown += 1
                yidx = int(cur.get("yourIndex", 0))
                me = cur["players"][yidx]
                opp = cur["players"][1 - yidx]
                print(f"\n--- MAIN selection #{shown} "
                      f"(turn={cur.get('turn')} yourIndex={yidx} "
                      f"min={sel['minCount']} max={sel['maxCount']}) ---")
                act = (me.get("active") or [None])
                print(f"  MY active: {summarize_pokemon(act[0] if act else None)}")
                bench = [summarize_pokemon(b) for b in me.get("bench", [])]
                print(f"  MY bench: {bench}")
                print(f"  MY prizes_left={len(me.get('prize', []))} "
                      f"hand={me.get('handCount')} deck={me.get('deckCount')}")
                oact = (opp.get("active") or [None])
                print(f"  OPP active: {summarize_pokemon(oact[0] if oact else None)}")
                print(f"  OPP prizes_left={len(opp.get('prize', []))}")
                for i, o in enumerate(sel["option"]):
                    print(f"    opt[{i}] {json.dumps(o)}")

        choice = agent(obs)
        if not is_legal(choice, sel):
            choice = legal_fallback(sel)
        obs = battle_select(choice)
        steps += 1
        if steps > 5000:
            print("did not terminate")
            break
    battle_finish()
    print(f"\n== option-type histogram over the game: {dict(type_ctr)} ==")


def main() -> None:
    deck = read_deck()
    dump_card_data(deck)
    print()
    dump_game(deck)


if __name__ == "__main__":
    main()
