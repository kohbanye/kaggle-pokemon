"""Phase 3 -- state/action representation + the policy/value/CB net skeleton.

The plan converges on an OSFP self-play policy+value net with a deck-construction
(CB) head. This package is the *wiring* for that net (Phase 3 is explicitly
"plumbing, not strength"):

- :mod:`src.net.features` -- a fixed per-card numeric feature table, built from
  the engine card/attack stats the runner injects (no learned embedding table to
  start; the "card embedding" is a learned MLP *projection* of these features).
- :mod:`src.net.encode`   -- turn an observation into a fixed-length state vector
  and each presented option into an option-feature vector.
- :mod:`src.net.model`    -- :class:`PolicyValueNet`, a pure-numpy net with a
  shared trunk and three heads: value, policy (scores the presented options), and
  CB (scores candidate cards for deck construction). Weights save/load as ``.npz``
  so the submission needs only numpy (plan SS D: minimal inference deps).
- :mod:`src.net.cb`       -- sequential, legal-masked 60-card deck generation.
- :mod:`src.net.train`    -- a numpy SGD step used for the Phase-3 learning-wiring
  sanity (and reused as the Phase-4 BC trainer).

The forward path is pure numpy because numpy is the one declared runtime
dependency (torch is not), which keeps the agent a light, self-contained bundle.
"""

from __future__ import annotations

from src.net.features import CARD_FEAT_DIM, CardFeatures
from src.net.model import PolicyValueNet

__all__ = ["CARD_FEAT_DIM", "CardFeatures", "PolicyValueNet"]
