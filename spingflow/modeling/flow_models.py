from spingflow.modeling import BaseFlowModel
import torch


class SimpleIsingTBFlowModel(BaseFlowModel):
    def __init__(self, N, n_hidden=256):
        N = int(N)
        internal_net = torch.nn.Sequential(
            torch.nn.Linear(2 * N**2, n_hidden),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(n_hidden, 2 * 2 * N**2)
        )
        super().__init__(N=N, internal_net=internal_net)