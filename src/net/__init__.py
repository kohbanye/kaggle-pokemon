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

Training lives apart, in torch + Lightning, so it never burdens the submission:

- :mod:`src.net.torch_model` -- the same net in torch, with an exact weight bridge
  to/from the numpy parameter dict (``to_numpy_net`` / ``from_numpy_net``).
- :mod:`src.net.lit`         -- a Lightning module for behaviour-cloning warm-start
  (Phase 4) and beyond.

The *serving* forward (model.py) is pure numpy because numpy is the one declared
runtime dependency; torch/Lightning are dev-only training deps. Train in torch,
export the weights to ``.npz``, and serve them from numpy -- a parity test keeps
the two forwards identical. This keeps the agent a light, self-contained bundle.
"""

from __future__ import annotations

from src.net.features import CARD_FEAT_DIM, CardFeatures
from src.net.model import PolicyValueNet

__all__ = ["CARD_FEAT_DIM", "CardFeatures", "PolicyValueNet"]
