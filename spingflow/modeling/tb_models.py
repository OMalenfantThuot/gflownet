from spingflow.modeling.base import BaseFlowModel
from spingflow.modeling.flow_models import setup_mlp, ConvolutionsModule
from typing import Optional
import torch


class BaseTBFlowModel(BaseFlowModel):
    def __init__(
        self, N: int, internal_net: torch.nn.Module, logZ: Optional[float] = 1.0
    ):
        super().__init__(N=N, internal_net=internal_net)
        self.logZ = torch.nn.Parameter(torch.ones(1) * logZ)

    def make_forward_choice(self, state):
        PF, _, _ = self.get_logits(state)
        action_choice = torch.distributions.categorical.Categorical(logits=PF).sample()
        new_state = self.create_new_state_from_choice(state, action_choice)
        return new_state

    def get_logits(self, state):
        unavailable_actions = torch.cat(
            [state[:, : self.N**2] + state[:, self.N**2 :]] * 2, dim=-1
        )
        logits = self.internal_net(state)
        PF = (
            logits[..., : 2 * self.N**2] * (1 - unavailable_actions)
            + unavailable_actions * -100
        )
        PB = logits[..., 2 * self.N**2 :] * state + (1 - state) * -100
        return PF, PB, unavailable_actions

    def get_current_logZ(self):
        return self.logZ


class MlpTBFlowModel(BaseTBFlowModel):
    def __init__(
        self,
        N: int,
        n_layers: int = 2,
        n_hidden: int = 256,
        logZ: Optional[float] = 1.0,
    ):
        N = int(N)
        internal_net = setup_mlp(
            input_dim=2 * N**2,
            output_dim=2 * 2 * N**2,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )
        super().__init__(N=N, internal_net=internal_net, logZ=logZ)


class ConvTBFlowModel(BaseTBFlowModel):
    def __init__(
        self,
        N,
        conv_n_layers=3,
        mlp_n_layers=2,
        mlp_n_hidden=256,
        conv_batch_norm=False,
        mlp_batch_norm=False,
        logZ: Optional[float] = 1.0,
    ):
        N = int(N)
        convolutions_module = ConvolutionsModule(
            N=N, conv_n_layers=conv_n_layers, batch_norm=conv_batch_norm
        )
        mlp_module = setup_mlp(
            input_dim=convolutions_module.output_dim,
            output_dim=2 * 2 * N**2,
            n_layers=mlp_n_layers,
            n_hidden=mlp_n_hidden,
            batch_norm=mlp_batch_norm,
        )
        internal_net = torch.nn.Sequential(convolutions_module, mlp_module)
        super().__init__(N=N, internal_net=internal_net, logZ=logZ)
