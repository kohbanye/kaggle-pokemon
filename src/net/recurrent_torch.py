"""Torch mirror of :class:`~src.net.recurrent_model.RecurrentPolicyValueNet`.

Same architecture in torch (for training) plus the exact numpy weight bridge, so
we train here and serve the numpy net. The addition over
:class:`~src.net.torch_model.TorchPolicyValueNet` is the **play LSTM**: the value
and policy heads read its hidden state, and :meth:`play_sequence` runs it over a
whole battle trajectory (the learner's recurrent forward). A parity test
(``tests/test_recurrent_parity.py``) pins the per-step torch sequence forward to
the numpy stateful :meth:`~RecurrentPolicyValueNet.step`.
"""

from __future__ import annotations

import torch
from torch import nn

from src.net.recurrent_model import RecurrentNetConfig, RecurrentPolicyValueNet
from src.net.torch_model import TorchPolicyValueNet


class TorchRecurrentNet(TorchPolicyValueNet):
    """Recurrent policy/value net in torch (play LSTM + sequence forward)."""

    def __init__(self, config: RecurrentNetConfig | None = None) -> None:
        cfg = config or RecurrentNetConfig()
        super().__init__(cfg)
        self.config = cfg
        ph = cfg.play_lstm_hidden
        # Re-size the heads to read the play-LSTM hidden, and add the play LSTM
        # (input = trunk output of width ``hidden``). Created after super().__init__
        # so they overwrite the base (trunk-width) heads.
        self.value_head = nn.Linear(ph, 1)
        self.policy1 = nn.Linear(ph + cfg.option_dim + cfg.embed_dim, cfg.policy_hidden)
        self.play_lstm = nn.LSTMCell(cfg.hidden, ph)

    # --- heads off the play-LSTM hidden -------------------------------------

    def value_from_h(self, h: torch.Tensor) -> torch.Tensor:
        """Value in ``[-1, 1]`` from the play-LSTM hidden: ``(B, ph) -> (B,)``."""
        return torch.tanh(self.value_head(h)).squeeze(-1)

    def policy_logits_from_h(
        self,
        h: torch.Tensor,
        options: torch.Tensor,
        option_rows: torch.Tensor,
    ) -> torch.Tensor:
        """Option logits from the hidden state: ``(B,ph),(B,K,opt),(B,K) -> (B,K)``."""
        k = options.shape[1]
        h_rep = h.unsqueeze(1).expand(-1, k, -1)
        opt_emb = self.cb_embed[option_rows]
        joint = torch.cat([h_rep, options, opt_emb], dim=-1)
        return self.policy2(torch.relu(self.policy1(joint))).squeeze(-1)

    # --- sequence forward (the learner's recurrent pass) --------------------

    def play_sequence(
        self,
        states: torch.Tensor,
        state_rows: torch.Tensor,
        state_mask: torch.Tensor,
        options: torch.Tensor,
        option_rows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the play LSTM over a battle trajectory batch.

        Shapes: ``states (B,T,state_dim)``, ``state_rows/mask (B,T,S,SLOT_MAX)``,
        ``options (B,T,K,option_dim)``, ``option_rows (B,T,K)``. Returns
        ``(logits (B,T,K), values (B,T))`` -- the per-step policy logits and value.
        Padded steps are computed too; the loss masks them out by ``valid``.
        """
        bsz, t_len = states.shape[0], states.shape[1]
        ph = self.config.play_lstm_hidden
        h = torch.zeros(bsz, ph, device=states.device, dtype=states.dtype)
        c = torch.zeros(bsz, ph, device=states.device, dtype=states.dtype)
        logits_t: list[torch.Tensor] = []
        values_t: list[torch.Tensor] = []
        for t in range(t_len):
            aug = self.augment_state(states[:, t], state_rows[:, t], state_mask[:, t])
            e = self.trunk(aug)
            h, c = self.play_lstm(e, (h, c))
            values_t.append(self.value_from_h(h))
            logits_t.append(
                self.policy_logits_from_h(h, options[:, t], option_rows[:, t]),
            )
        return torch.stack(logits_t, dim=1), torch.stack(values_t, dim=1)

    # --- numpy bridge -------------------------------------------------------

    def _matrix_keys(self) -> list[tuple[nn.Parameter, str]]:
        """Base raw tensors plus the play-LSTM's four (torch-native layout)."""
        return [
            *super()._matrix_keys(),
            (self.play_lstm.weight_ih, "play_lstm_w_ih"),
            (self.play_lstm.weight_hh, "play_lstm_w_hh"),
            (self.play_lstm.bias_ih, "play_lstm_b_ih"),
            (self.play_lstm.bias_hh, "play_lstm_b_hh"),
        ]

    def to_numpy_net(self) -> RecurrentPolicyValueNet:
        """A numpy :class:`RecurrentPolicyValueNet` with this net's weights."""
        return RecurrentPolicyValueNet(self.config, self.to_numpy_params())


def from_numpy_recurrent(net: RecurrentPolicyValueNet) -> TorchRecurrentNet:
    """Build a torch recurrent net initialised from a numpy recurrent net.

    The CB (deck-build) head is unchanged from Phase 5c, so the learner reuses
    :func:`src.net.lit.cb_sequence_logits` directly on this net for the deck arm.
    """
    torch_net = TorchRecurrentNet(net.config)
    torch_net.load_numpy_params(net.params)
    return torch_net
