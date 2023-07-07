from spingflow.modeling.base import BaseFlowModel
from spingflow.modeling.flow_models import setup_mlp, ConvolutionsModule
from typing import Optional
import torch


class BaseDBFlowModel(BaseFlowModel):
    def __init__(
        self,
        N: int,
        internal_net: torch.nn.Module,
        state_flow_net: torch.nn.Module,
    ):
        super().__init__(N=N, internal_net=internal_net)
        self.state_flow_net = state_flow_net

    def make_forward_choice(self, state):
        PF, _ = self.get_logits(state)
        action_choice = torch.distributions.categorical.Categorical(probs=PF).sample()
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

    def get_state_flow(self, state):
        return self.state_flow_net(state)

    def get_current_logZ(self):
        empty_state = self.create_input_batch(batch_size=1)
        logZ = self.get_state_flow(empty_state)
        return logZ


class MlpDBFlowModel(BaseDBFlowModel):
    def __init__(
        self,
        N: int,
        n_layers: int = 2,
        n_hidden: int = 256,
    ):
        N = int(N)
        internal_net = setup_mlp(
            input_dim=2 * N**2,
            output_dim=2 * 2 * N**2,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )
        state_flow_net = setup_mlp(
            input_dim=2 * N**2,
            output_dim=1,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )
        super().__init__(N=N, internal_net=internal_net, state_flow_net=state_flow_net)


class ConvDBFlowModel(BaseDBFlowModel):
    def __init__(
        self,
        N,
        conv_n_layers=3,
        mlp_n_layers=2,
        mlp_n_hidden=256,
        conv_batch_norm=False,
        mlp_batch_norm=False,
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

        state_flow_net = setup_mlp(
            input_dim=2 * N**2,
            output_dim=1,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )
        super().__init__(N=N, internal_net=internal_net, state_flow_net=state_flow_net)
