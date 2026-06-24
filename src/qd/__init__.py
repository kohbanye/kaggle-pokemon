"""Quality-Diversity deck search (MAP-Elites) -- the deck side of "QD decks + RL play".

The recurrent net's CB head collapses the deck distribution to one archetype (the
eval showed sampled-deck diversity halving over training). This package replaces
*learning* the deck with *searching* it: a MAP-Elites archive keeps the best deck
per archetype niche (so diversity is structural, not lost to a gradient), while the
RL net keeps learning to *play*. Only engine legality is enforced -- no hand-coded
"best-practice" deck constraints; fitness (win rate) weeds out unplayable decks.
"""

from src.qd.archive import Elite, MapElitesArchive
from src.qd.deck_qd import (
    behaviour_descriptor,
    deck_stats,
    mutate,
    random_legal_deck,
)

__all__ = [
    "Elite",
    "MapElitesArchive",
    "behaviour_descriptor",
    "deck_stats",
    "mutate",
    "random_legal_deck",
]
