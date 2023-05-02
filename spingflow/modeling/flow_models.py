from spingflow.modeling import BaseFlowModel
import torch


class SimpleIsingFlowModel(BaseFlowModel):
    def __init__(self, N, n_layers=2, n_hidden=256):
        N = int(N)
        internal_net = torch.nn.Sequential(
            torch.nn.Linear(2 * N**2, n_hidden), torch.nn.LeakyReLU()
        )
        if n_layers > 2:
            for _ in range(n_layers - 2):
                internal_net.append(torch.nn.Linear(n_hidden, n_hidden))
                internal_net.append(torch.nn.LeakyReLU())
        internal_net.append(torch.nn.Linear(n_hidden, 2 * 2 * N**2))
        super().__init__(N=N, internal_net=internal_net)
