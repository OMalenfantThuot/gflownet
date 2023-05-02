import torch
import numpy as np
from spingflow.modeling.energy import IsingEnergyModel


class BaseFlowModel(torch.nn.Module):
    def __init__(self, N:int, internal_net:torch.nn.Module):
        super().__init__()
        self.N = int(N)
        self.internal_net = internal_net
        self.logZ = torch.nn.Parameter(torch.ones(1) * self.N)
        
    def forward(self, state):
        for _ in range(self.N**2):
            state = self.make_forward_choice(state)
        return state
    
    def make_forward_choice(self, state):
        PF, _ = self.get_logits(state)
        action_choice = torch.distributions.categorical.Categorical(probs=PF).sample()
        new_state = self.create_new_state_from_choice(state, action_choice)
        return new_state
        
    def get_logits(self, state):
        unavailable_actions = torch.cat([state[:,:self.N**2] + state[:,self.N**2:]]*2, dim=-1)
        logits = self.internal_net(state)
        PF = logits[..., :2*self.N**2] * (1 - unavailable_actions) + unavailable_actions * -100
        PB = logits[..., 2*self.N**2:] * state + (1 - state) * -100
        return PF, PB, unavailable_actions
        
    def create_new_state_from_choice(self, state, choice):
        base = torch.zeros_like(state)
        base[torch.arange(base.shape[0], dtype=torch.int64), choice] = 1
        new_state = state + base
        return new_state

class IsingFullGFlowModel(torch.nn.Module):
    def __init__(self, flow_model:BaseFlowModel, reward_model:IsingEnergyModel):
        super().__init__()
        self.flow_model = flow_model
        self.reward_model = reward_model
        assert self.reward_model.N == self.flow_model.N, "The reward and flow models are not of matching dimensions."
        
    @property
    def N(self):
        return self.flow_model.N

    def forward(self, initial_state:torch.Tensor):
        final_state = self.flow_model(initial_state)
        reward = self.reward_model(final_state)
        return final_state, reward
    
    def create_input_batch(self, batch_size):
        return torch.zeros(
            (batch_size, 2 * (self.flow_model.N**2)), dtype=torch.float32, requires_grad=True
        )