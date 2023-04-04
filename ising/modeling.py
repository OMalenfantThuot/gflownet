import torch
import numpy as np


class IsingEnergyModel(torch.nn.Module):
    def __init__(self, J, device="cpu"):
        super().__init__()
        self.J = J.to(device)
        self.flatten = torch.nn.Flatten().to(device)

    def _batched_states_to_spins(self, states):
        N = int(np.sqrt(states.shape[1] / 2))
        spin_values = states[..., : N**2] + -1 * states[..., N**2 :]
        return spin_values

    def forward(self, states):
        spin_values = self._batched_states_to_spins(states)
        energies = (
            -self.flatten(spin_values) @ self.J * self.flatten(spin_values)
        ).sum(dim=-1)
        return energies

    def get_reward(self, states):
        energies = self(states)
        reward = torch.exp(-1 * energies)
        return reward

    
class IsingSimpleFlowModel(torch.nn.Module):
    def __init__(self, N, n_hidden=256):
        super().__init__()
        
        self.N = N
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * self.N**2, n_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(n_hidden, 2 * self.N**2))
        
    def forward(self, state):
        mask = torch.cat([state[:,:self.N**2] + state[:,self.N**2:]]*2, dim=-1)
        probs = self.net(state).exp() * (1 - mask)
        return probs
    
    def make_choice(self, state):
        probs = self(state)
        choice = torch.distributions.categorical.Categorical(probs=probs).sample()
        new_state = state.clone()
        new_state[torch.arange(new_state.shape[0], dtype=torch.int64), choice] = 1
        return new_state