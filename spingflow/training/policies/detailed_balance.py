from spingflow.modeling import IsingFullGFlowModel
from spingflow.training.policies import BasePolicy
from torch.distributions.categorical import Categorical
import torch


class DetailedBalancePolicy(BasePolicy):
    def __init__(self, model: IsingFullGFlowModel):
        super().__init__(policy="db", model=model)

    def training_trajectory_and_metrics(
        self, state: torch.Tensor, temperature: float, epsilon: float = 0
    ):
        # Evaluate initial state (S_0)
        PF, PB, _ = self.model.flow_model.get_logits(state)
        log_flow = self.model.flow_model.get_state_flow(state)
        traj_mismatch = 0

        # Loop on following decisions
        for step in range(1, self.model.N**2 + 1):
            # Make choices
            choice, logprob = self.choose_actions_from_PF(PF, epsilon)
            new_state = self.model.flow_model.create_new_state_from_choice(
                state, choice
            )

            # Get forward and backward probabilities for new state
            newPF, newPB, _ = self.model.flow_model.get_logits(new_state)
            newlog_flow = self.model.flow_model.get_state_flow(new_state)

            # Check if we are at terminal state
            if step == self.model.N**2:
                # Calculate reward
                logreward = self.model.get_logreward(new_state, temperature)
                action_mismatch = (
                    log_flow
                    + logprob
                    - logreward
                    - Categorical(logits=newPB).log_prob(choice)
                )
            else:
                action_mismatch = (
                    log_flow
                    + logprob
                    - newlog_flow
                    - Categorical(logits=newPB).log_prob(choice)
                )
            traj_mismatch += action_mismatch**2

            # Reset for next loop
            state = new_state
            PF, PB, log_flow = newPF, newPB, newlog_flow

        # Trajectory balance loss function
        loss = traj_mismatch  # + (logreward - logflow) ** 2

        return state, loss.mean()
