import torch
import numpy as np
from spingflow.modeling import IsingFullGFlowModel
from spingflow.predict.magnetization import IsingMagnetizationModel
from torch.distributions.categorical import Categorical


class SpinGFlowPredictor:
    def __init__(
        self,
        model: IsingFullGFlowModel,
        nsamples: int,
        batch_size: int,
        device: torch.device = torch.device("cpu")
    ):
        self.model = model
        self.nsamples = nsamples
        self.batch_size = batch_size
        self.device = device
        
    def predict(self, property):
        self.model.eval()
        
        if property == "magn":
            magn_model = IsingMagnetizationModel(N=self.model.N)
            magn_model.eval()
        else:
            raise NotImplementedError()

        n_traj = 0
        predictions = []
        while n_traj < self.nsamples:
            n_traj += self.batch_size
            
            batch = self.model.create_input_batch(batch_size=self.batch_size)
            states = self.choose_trajectories(batch)
            
            if property == "magn":
                predictions.append(magn_model(states))

        predictions = torch.cat(predictions, dim=0)
        prediction = torch.mean(predictions, dim=0)
        return prediction
            
            
    def choose_trajectories(self, state):
        PF, _, _ = self.model.flow_model.get_logits(state)
        
        for step in range(1, self.model.N**2 + 1):
            categorical = Categorical(logits=PF)
            choice = categorical.sample()
            state = self.model.flow_model.create_new_state_from_choice(
                state, choice
            )
            PF, PB, _ = self.model.flow_model.get_logits(state)
        return state