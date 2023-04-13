import torch


class IsingEnergyModel(torch.nn.Module):
    def __init__(self, N: int, J: torch.Tensor, device: str = "cpu"):
        super().__init__()
        self.N = int(N)
        self.J = J.to(device)

    def _states_to_spins(self, states: torch.Tensor):
        spin_values = states[..., : self.N**2] + -1 * states[..., self.N**2 :]
        return spin_values

    def forward(self, states):
        spin_values = self._states_to_spins(states)
        energies = (-1 * spin_values @ self.J * spin_values).sum(dim=-1)
        return energies

    def get_reward(self, states, T):
        energies = self(states)
        reward = torch.exp(-1 / T * energies)
        return reward

    def get_energy(self, states):
        return self(states)
