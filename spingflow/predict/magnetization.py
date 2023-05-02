import torch


class IsingMagnetizationModel(torch.nn.Module):
    def __init__(self, N: int):
        super().__init__()
        self.N = int(N)
        
    def _states_to_spins(self, states: torch.Tensor):
        spin_values = states[..., : self.N**2] + -1 * states[..., self.N**2 :]
        return spin_values
        
    def forward(self, states):
        spin_values = self._states_to_spins(states)
        magnetizations = (spin_values).mean(dim=-1, keepdim=True).abs()
        return magnetizations
