import torch
import numpy as np


class IsingEnergyModel(torch.nn.Module):
    def __init__(self, J):
        super().__init__()
        self.J = J
        self.flatten = torch.nn.Flatten()

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
        reward = torch.exp(-1 * energies).sum()
        return reward
