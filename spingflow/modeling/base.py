from spingflow.modeling.energy import IsingEnergyModel
import torch


class BaseFlowModel(torch.nn.Module):
    r"""
    Base class to use as inheritance for policy specific
    base flow model classes. These classes must implement
    the make_forward_choice and get_current_logZ methods.
    """

    def __init__(self, N: int, internal_net: torch.nn.Module):
        super().__init__()
        self.N = int(N)
        self.internal_net = internal_net

    def forward(self, state):
        for _ in range(self.N**2):
            state = self.make_forward_choice(state)
        return state

    def create_new_state_from_choice(self, state, choice):
        base = torch.zeros_like(state)
        base[torch.arange(base.shape[0], dtype=torch.int64), choice] = 1
        new_state = state + base
        return new_state

    def make_forward_choice(self, state):
        raise NotImplementedError()

    def get_current_logZ(self):
        raise NotImplementedError()


class IsingFullGFlowModel(torch.nn.Module):
    def __init__(self, flow_model: BaseFlowModel, reward_model: IsingEnergyModel):
        super().__init__()
        self.flow_model = flow_model
        self.reward_model = reward_model
        assert (
            self.reward_model.N == self.flow_model.N
        ), "The reward and flow models are not of matching dimensions."

    @property
    def N(self):
        return self.flow_model.N

    def forward(self, initial_state: torch.Tensor):
        final_state = self.flow_model(initial_state)
        reward = self.reward_model(final_state)
        return final_state, reward

    def create_input_batch(self, batch_size):
        return torch.zeros(
            (batch_size, 2 * (self.flow_model.N**2)),
            dtype=torch.float32,
            requires_grad=True,
        )

    def get_current_logZ(self):
        return self.flow_model.get_current_logZ()
