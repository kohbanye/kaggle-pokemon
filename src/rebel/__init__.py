"""Approximate ReBeL play arm (design: docs/research/rebel-approximate-design.md).

Gate B (the estimator kill-test) lives here: a game interface (`game.py`), exact CFR +
external-sampling MCCFR + exact exploitability (`cfr.py`), validated on Kuhn poker
before any determinized-engine or belief code is written.
"""
