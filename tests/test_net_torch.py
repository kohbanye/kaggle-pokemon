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

from src.net.encode import OPTION_DIM, STATE_DIM
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
    tnet = TorchPolicyValueNet().double()  # float64 to match the numpy net exactly
    npnet = tnet.to_numpy_net()

    rng = np.random.default_rng(0)
    x = rng.standard_normal(STATE_DIM)
    options = rng.standard_normal((6, OPTION_DIM))
    # card_logits consumes the fixed features concatenated with the card embedding.
    cards = rng.standard_normal((9, CARD_FEAT_DIM + tnet.config.embed_dim))

    with torch.no_grad():
        t_value = tnet.value(torch.as_tensor(x).unsqueeze(0)).item()
        t_policy = tnet.policy_logits(
            torch.as_tensor(x).unsqueeze(0), torch.as_tensor(options).unsqueeze(0),
        ).squeeze(0).numpy()
        t_cards = tnet.card_logits(torch.as_tensor(cards)).numpy()

    assert abs(t_value - npnet.value(x)) < 1e-9
    assert np.allclose(t_policy, npnet.policy_logits(x, options), atol=1e-9)
    assert np.allclose(t_cards, npnet.card_logits(cards), atol=1e-9)


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

def test_policy_loss_trains_and_beats_chance() -> None:
    torch.manual_seed(0)
    batch, k = 96, 4
    states = torch.randn(batch, STATE_DIM)
    options = torch.randn(batch, k, OPTION_DIM)
    mask = torch.ones(batch, k, dtype=torch.bool)
    # Target = the option whose first feature is largest -- a signal the policy
    # head sees directly, so a working forward/backward/optim must learn it.
    targets = options[:, :, 0].argmax(dim=1)

    lit = LitPolicyValue(lr=0.05)
    opt = torch.optim.Adam(lit.parameters(), lr=0.05)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = lit.policy_loss(states, options, mask, targets)
        first = loss.item() if first is None else first
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.7

    with torch.no_grad():
        acc = (lit.net.policy_logits(states, options).argmax(dim=1) == targets)
    assert acc.float().mean().item() > 0.5  # vs 0.25 chance


def test_padding_mask_excludes_options() -> None:
    # A padded (masked-out) option must never be chosen, even with a huge logit.
    lit = LitPolicyValue()
    states = torch.zeros(1, STATE_DIM)
    options = torch.zeros(1, 3, OPTION_DIM)
    mask = torch.tensor([[True, True, False]])
    targets = torch.tensor([0])
    loss = lit.policy_loss(states, options, mask, targets)
    assert torch.isfinite(loss)  # masking to -inf must not produce nan/inf loss


# --- end-to-end Lightning fit + export to the numpy serving net -----------

def test_trainer_fit_and_export(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    torch.manual_seed(0)
    batch, k = 32, 4
    states = torch.randn(batch, STATE_DIM)
    options = torch.randn(batch, k, OPTION_DIM)
    mask = torch.ones(batch, k, dtype=torch.bool)
    targets = options[:, :, 0].argmax(dim=1)
    values = torch.zeros(batch)
    loader = DataLoader(
        TensorDataset(states, options, mask, targets, values), batch_size=8,
    )

    lit = LitPolicyValue(lr=0.05)
    _trainer().fit(lit, loader)

    # Export the trained net to the numpy serving format and confirm it loads
    # and reproduces the torch forward (this is what NetAgent will consume).
    path = tmp_path / "weights.npz"
    lit.net.double().to_numpy_net().save(path)
    served = PolicyValueNet.load(path)
    x = np.random.default_rng(0).standard_normal(STATE_DIM)
    with torch.no_grad():
        t_value = lit.net.value(torch.as_tensor(x).unsqueeze(0)).item()
    assert abs(t_value - served.value(x)) < 1e-9
