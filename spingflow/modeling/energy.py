from spingflow.spins import create_J_matrix
from typing import Union
import torch


class IsingEnergyModel(torch.nn.Module):
    def __init__(self, N: int, J: Union[int, torch.Tensor]):
        super().__init__()
        self.N = int(N)
        self.J = J

    @property
    def J(self):
        return self._J

    @J.setter
    def J(self, J):
        if type(J) is float:
            _J = create_J_matrix(self.N, sigma=J)
            self.register_buffer("_J", _J)
        elif type(J) is torch.Tensor:
            _J = J
            self.register_buffer("_J", _J)
        else:
            raise RuntimeError(
                "The J input should be a float or the matrix as a tensor."
            )

    def _states_to_spins(self, states: torch.Tensor):
        spin_values = states[..., : self.N**2] + -1 * states[..., self.N**2 :]
        return spin_values

    def forward(self, states):
        spin_values = self._states_to_spins(states)
        energies = (-1 * spin_values @ self.J * spin_values).sum(dim=-1)
        return energies

    def get_reward(self, states, T):
        energies = self(states)
        reward = torch.exp(-1 * energies / T)
        return reward

    def get_logreward(self, states, T):
        energies = self(states)
        logreward = (-1 * energies / T).clip(-20)
        return logreward

    def get_energy(self, states):
        return self(states)
