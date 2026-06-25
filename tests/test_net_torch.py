"""Tests for the base net's torch<->numpy weight bridge.

The submission serves the pure-numpy forward; training happens in torch. These
tests pin that the torch and numpy forwards are numerically identical after a
weight copy (so the exported npz the agent loads behaves exactly like what was
trained), for the base :class:`PolicyValueNet` the recurrent net extends.
"""

import numpy as np
import torch

from src.net.encode import OPTION_DIM, SLOT_MAX, STATE_DIM, STATE_EMBED_SLOTS
from src.net.features import CARD_FEAT_DIM
from src.net.model import NetConfig, PolicyValueNet
from src.net.torch_model import TorchPolicyValueNet, from_numpy_net

# --- torch <-> numpy forward parity ---------------------------------------

def test_torch_numpy_forward_parity() -> None:
    torch.manual_seed(0)
    # n_cards > 0 so the shared card embedding is non-trivial in both forwards.
    cfg = NetConfig(n_cards=8)
    tnet = TorchPolicyValueNet(cfg).double()  # float64 to match the numpy net exactly
    npnet = tnet.to_numpy_net()

    rng = np.random.default_rng(0)
    x = rng.standard_normal(STATE_DIM)  # fixed state features
    options = rng.standard_normal((6, OPTION_DIM))  # fixed option features
    # Embedding rows index cb_embed's n_cards + 1 rows (the last is UNK).
    rows = rng.integers(0, cfg.n_cards + 1, size=(STATE_EMBED_SLOTS, SLOT_MAX))
    mask = rng.random((STATE_EMBED_SLOTS, SLOT_MAX)) > 0.4
    mask[1, :] = False  # an empty slot must contribute a zero block (parity edge)
    option_rows = rng.integers(0, cfg.n_cards + 1, size=6)
    # card_logits consumes [lstm_hidden ⊕ fixed features ⊕ card embedding].
    cb_in = cfg.lstm_hidden + CARD_FEAT_DIM + cfg.embed_dim
    cards = rng.standard_normal((9, cb_in))

    def t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a).unsqueeze(0)

    with torch.no_grad():
        t_value = tnet.value(t(x), t(rows), t(mask)).item()
        t_policy = (
            tnet.policy_logits(t(x), t(rows), t(mask), t(options), t(option_rows))
            .squeeze(0)
            .numpy()
        )
        t_cards = tnet.card_logits(torch.as_tensor(cards)).numpy()

    assert abs(t_value - npnet.value(x, rows, mask)) < 1e-9
    assert np.allclose(
        t_policy, npnet.policy_logits(x, rows, mask, options, option_rows), atol=1e-9,
    )
    assert np.allclose(t_cards, npnet.card_logits(cards), atol=1e-9)


def test_lstm_cell_parity() -> None:
    # H != input != 4H so any transpose or gate-order slip shape-fails or differs.
    cfg = NetConfig(lstm_hidden=7, embed_dim=5, n_cards=4)
    tnet = TorchPolicyValueNet(cfg).double()
    npnet = tnet.to_numpy_net()
    rng = np.random.default_rng(3)
    x = rng.standard_normal(cfg.embed_dim)
    h = rng.standard_normal(cfg.lstm_hidden)
    c = rng.standard_normal(cfg.lstm_hidden)

    with torch.no_grad():
        th, tc = tnet.cb_lstm(
            torch.as_tensor(x).unsqueeze(0),
            (torch.as_tensor(h).unsqueeze(0), torch.as_tensor(c).unsqueeze(0)),
        )
    nh, nc = npnet.lstm_step(x, h, c)
    assert np.allclose(th.squeeze(0).numpy(), nh, atol=1e-9)
    assert np.allclose(tc.squeeze(0).numpy(), nc, atol=1e-9)


def test_lstm_weights_roundtrip_not_transposed() -> None:
    cfg = NetConfig(lstm_hidden=7, embed_dim=5, n_cards=4)  # H != embed_dim
    src = PolicyValueNet.random(np.random.default_rng(4), cfg)
    assert src.params["lstm_w_ih"].shape == (28, 5)  # (4H, in), NOT (5, 28)
    assert src.params["lstm_w_hh"].shape == (28, 7)  # (4H, H)
    back = from_numpy_net(src).double().to_numpy_net()
    for key in ("lstm_w_ih", "lstm_w_hh", "lstm_b_ih", "lstm_b_hh", "cb_start"):
        assert back.params[key].shape == src.params[key].shape
        assert np.allclose(back.params[key], src.params[key], atol=1e-9)


def test_round_trip_numpy_to_torch_to_numpy() -> None:
    src = PolicyValueNet.random(np.random.default_rng(1))
    back = from_numpy_net(src).double().to_numpy_net()
    for k, v in src.params.items():
        assert np.allclose(back.params[k], v, atol=1e-6)


def test_cb_embed_roundtrip_not_transposed() -> None:
    # The card embedding is a row table, bridged WITHOUT transpose. n_cards != embed_dim
    # so a stray .T would shape-mismatch (6,3)->(3,6) and fail loudly here.
    cfg = NetConfig(n_cards=5, embed_dim=3)
    src = PolicyValueNet.random(np.random.default_rng(2), cfg)
    assert src.params["cb_embed"].shape == (6, 3)  # n_cards + 1 (UNK) rows
    back = from_numpy_net(src).double().to_numpy_net()
    assert back.params["cb_embed"].shape == (6, 3)
    assert np.allclose(back.params["cb_embed"], src.params["cb_embed"], atol=1e-9)
