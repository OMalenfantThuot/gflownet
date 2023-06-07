from spingflow.modeling import IsingFullGFlowModel
from spingflow.training.policies import BasePolicy
from torch.distributions.categorical import Categorical
import torch


class TrajectoryBalancePolicy(BasePolicy):
    def __init__(self, model: IsingFullGFlowModel):
        super().__init__(policy="tb", model=model)

    def training_trajectory_and_metrics(self, state: torch.Tensor, temperature: float):
        # Evaluate initial state (S_0)
        PF, PB, _ = self.model.flow_model.get_logits(state)
        traj_PF, traj_PB = 0, 0

        # Loop on following decisions
        for step in range(1, self.model.N**2 + 1):
            # Use forward probabilities and make choice
            categorical = Categorical(logits=PF)
            choice = categorical.sample()
            new_state = self.model.flow_model.create_new_state_from_choice(
                state, choice
            )
            traj_PF += categorical.log_prob(choice)

            # Check if we are at terminal state
            if step == self.model.N**2:
                # Calculate reward
                logreward = self.model.reward_model.get_logreward(
                    new_state, temperature
                )

            # Get forward and backward probabilities for new state
            PF, PB, _ = self.model.flow_model.get_logits(new_state)
            traj_PB += Categorical(logits=PB).log_prob(choice)

            # Reset for next loop
            state = new_state

        # Trajectory balance loss function
        loss = (self.model.flow_model.logZ + traj_PF - traj_PB - logreward) ** 2

        return state, loss.mean()
