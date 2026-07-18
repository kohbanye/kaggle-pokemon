"""A/B: Archaludon Judge flex build vs baseline Archaludon across a fixed opponent
set. Prints per-opponent win rate (Wilson 95% CI) for both subjects + delta CI."""
import math
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "data/sample_submission")
from scripts.run_eval import _make_agent, load_engine_data, play_game, resolve_deck

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200

SUBJECTS = {
    "baseline": ("archaludon", "archaludon_netdeck"),
    "judge": ("archaludon_judge", "archaludon_judge"),
}
# (label, opponent agent name, opponent deck name)
OPPONENTS = [
    ("alakazam", "alakazam", "alakazam_netdeck"),
    ("lad00_P_evo_1pz", "greedy_plus", "heldout2/lad00_P_evo_1pz_e790"),
    ("lad03_G_evo_1pz", "greedy_plus", "heldout2/lad03_G_evo_1pz_e741"),
    ("lad04_F_evo_mid", "greedy_plus", "heldout2/lad04_F_evo_mid_e839"),
    ("lad07_M_evo_mid", "greedy_plus", "heldout2/lad07_M_evo_mid_e711"),
    ("lad11_L_evo_1pz", "greedy_plus", "heldout2/lad11_L_evo_1pz_e745"),
]


def wilson(w: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = w / n
    z = 1.96
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


def run(  # noqa: PLR0913
    subj_name: str,
    deck_name: str,
    opp_name: str,
    opp_deck: str,
    n: int,
    base_seed: int,
) -> tuple[int, int]:
    eng = ENG
    sub = _make_agent(subj_name, resolve_deck(deck_name), eng, None, None)
    opp = _make_agent(opp_name, resolve_deck(opp_deck), eng, None, None)
    w = dec = 0
    for g in range(n):
        a_p0 = g % 2 == 0
        r = play_game(sub, opp, a_is_player0=a_p0, seed=base_seed + g) if a_p0 \
            else play_game(opp, sub, a_is_player0=False, seed=base_seed + g)
        if r.a_won or r.b_won:
            dec += 1
            if r.a_won:
                w += 1
    return w, dec


ENG = load_engine_data()
t0 = time.perf_counter()
results = {}  # opp_label -> {subj: (w, dec)}
for opp_label, opp_name, opp_deck in OPPONENTS:
    results[opp_label] = {}
    for subj, (sname, dname) in SUBJECTS.items():
        w, dec = run(sname, dname, opp_name, opp_deck, N, base_seed=1000)
        results[opp_label][subj] = (w, dec)

print(f"\n== Judge A/B, {N} games/matchup, slot-swapped, base_seed=1000 ==")
print(f"(elapsed {time.perf_counter() - t0:.1f}s)\n")
hdr = (f"{'opponent':<18}{'baseline WR [95% CI]':<30}"
       f"{'judge WR [95% CI]':<30}{'delta [95% CI]':<24}")
print(hdr)
print("-" * len(hdr))
agg = {"baseline": [0, 0], "judge": [0, 0]}
for opp_label, _, _ in OPPONENTS:
    bw, bd = results[opp_label]["baseline"]
    jw, jd = results[opp_label]["judge"]
    bp, blo, bhi = wilson(bw, bd)
    jp, jlo, jhi = wilson(jw, jd)
    delta = jp - bp
    se = math.sqrt(bp * (1 - bp) / max(bd, 1) + jp * (1 - jp) / max(jd, 1))
    dlo, dhi = delta - 1.96 * se, delta + 1.96 * se
    sep = "*" if (dlo > 0 or dhi < 0) else " "
    print(f"{opp_label:<18}"
          f"{f'{bp:.3f} [{blo:.3f},{bhi:.3f}]':<30}"
          f"{f'{jp:.3f} [{jlo:.3f},{jhi:.3f}]':<30}"
          f"{f'{delta:+.3f} [{dlo:+.3f},{dhi:+.3f}]{sep}':<24}")
    agg["baseline"][0] += bw
    agg["baseline"][1] += bd
    agg["judge"][0] += jw
    agg["judge"][1] += jd

bp, blo, bhi = wilson(*agg["baseline"])
jp, jlo, jhi = wilson(*agg["judge"])
delta = jp - bp
se = math.sqrt(bp * (1 - bp) / agg["baseline"][1] + jp * (1 - jp) / agg["judge"][1])
dlo, dhi = delta - 1.96 * se, delta + 1.96 * se
print("-" * len(hdr))
sep = "*" if (dlo > 0 or dhi < 0) else " "
print(f"{'AGGREGATE':<18}"
      f"{f'{bp:.3f} [{blo:.3f},{bhi:.3f}]':<30}"
      f"{f'{jp:.3f} [{jlo:.3f},{jhi:.3f}]':<30}"
      f"{f'{delta:+.3f} [{dlo:+.3f},{dhi:+.3f}]{sep}':<24}")
print("\n(* = delta CI excludes 0)")
