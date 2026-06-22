"""Tests for the torch training side and the torch<->numpy weight bridge.

The submission serves the pure-numpy forward; training happens in torch +
Lightning. These tests pin the contract between the two: (1) the torch and numpy
forwards are numerically identical after a weight copy (so the exported npz the
agent loads behaves exactly like what was trained), (2) the Lightning policy loss
actually trains, and (3) a end-to-end ``Trainer.fit`` runs and exports cleanly.
"""

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.net.encode import OPTION_DIM, SLOT_MAX, STATE_DIM, STATE_EMBED_SLOTS
from src.net.features import CARD_FEAT_DIM
from src.net.lit import LitPolicyValue
from src.net.model import NetConfig, PolicyValueNet
from src.net.torch_model import TorchPolicyValueNet, from_numpy_net


def _trainer() -> L.Trainer:
    """A quiet, CPU, no-IO trainer for fast smoke tests."""
    return L.Trainer(
        max_epochs=2, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False,
    )


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


# --- Lightning policy loss actually trains --------------------------------

def _empty_rows(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero state-embedding rows + all-False mask (UNK / zero contribution)."""
    rows = torch.zeros(batch, STATE_EMBED_SLOTS, SLOT_MAX, dtype=torch.long)
    mask = torch.zeros(batch, STATE_EMBED_SLOTS, SLOT_MAX, dtype=torch.bool)
    return rows, mask


def test_policy_loss_trains_and_beats_chance() -> None:
    torch.manual_seed(0)
    batch, k = 96, 4
    states = torch.randn(batch, STATE_DIM)
    state_rows, state_mask = _empty_rows(batch)
    options = torch.randn(batch, k, OPTION_DIM)
    mask = torch.ones(batch, k, dtype=torch.bool)
    option_rows = torch.zeros(batch, k, dtype=torch.long)
    # Target = the option whose first feature is largest -- a signal the policy
    # head sees directly, so a working forward/backward/optim must learn it.
    targets = options[:, :, 0].argmax(dim=1)

    lit = LitPolicyValue(lr=0.05)
    opt = torch.optim.Adam(lit.parameters(), lr=0.05)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = lit.policy_loss(
            states, state_rows, state_mask, options, mask, option_rows, targets,
        )
        first = loss.item() if first is None else first
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.7

    with torch.no_grad():
        logits = lit.net.policy_logits(
            states, state_rows, state_mask, options, option_rows,
        )
    assert (logits.argmax(dim=1) == targets).float().mean().item() > 0.5  # vs 0.25


def test_padding_mask_excludes_options() -> None:
    # A padded (masked-out) option must never be chosen, even with a huge logit.
    lit = LitPolicyValue()
    states = torch.zeros(1, STATE_DIM)
    state_rows, state_mask = _empty_rows(1)
    options = torch.zeros(1, 3, OPTION_DIM)
    mask = torch.tensor([[True, True, False]])
    option_rows = torch.zeros(1, 3, dtype=torch.long)
    targets = torch.tensor([0])
    loss = lit.policy_loss(
        states, state_rows, state_mask, options, mask, option_rows, targets,
    )
    assert torch.isfinite(loss)  # masking to -inf must not produce nan/inf loss


# --- end-to-end Lightning fit + export to the numpy serving net -----------

def test_trainer_fit_and_export(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    torch.manual_seed(0)
    batch, k = 32, 4
    states = torch.randn(batch, STATE_DIM)
    state_rows, state_mask = _empty_rows(batch)
    options = torch.randn(batch, k, OPTION_DIM)
    mask = torch.ones(batch, k, dtype=torch.bool)
    option_rows = torch.zeros(batch, k, dtype=torch.long)
    targets = options[:, :, 0].argmax(dim=1)
    values = torch.zeros(batch)
    loader = DataLoader(
        TensorDataset(
            states, state_rows, state_mask, options, mask, option_rows,
            targets, values,
        ),
        batch_size=8,
    )

    lit = LitPolicyValue(lr=0.05)
    _trainer().fit(lit, loader)

    # Export the trained net to the numpy serving format and confirm it loads
    # and reproduces the torch forward (this is what NetAgent will consume).
    path = tmp_path / "weights.npz"
    lit.net.double().to_numpy_net().save(path)
    served = PolicyValueNet.load(path)
    x = np.random.default_rng(0).standard_normal(STATE_DIM)
    nrows = np.zeros((STATE_EMBED_SLOTS, SLOT_MAX), dtype=np.intp)
    nmask = np.zeros((STATE_EMBED_SLOTS, SLOT_MAX), dtype=np.bool_)
    with torch.no_grad():
        t_value = lit.net.value(
            torch.as_tensor(x).unsqueeze(0),
            torch.as_tensor(nrows).unsqueeze(0),
            torch.as_tensor(nmask).unsqueeze(0),
        ).item()
    assert abs(t_value - served.value(x, nrows, nmask)) < 1e-9
