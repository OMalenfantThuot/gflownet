from spingflow.modeling.tb_models import BaseTBFlowModel
from spingflow.training.policies.utils import get_policy
from torch.distributions.categorical import Categorical
import torch


class BasePolicy:
    def __init__(self, policy, model):
        self.check_model_is_consistent_with_policy(policy, model)
        self.policy = policy
        self.model = model

    def training_trajectory_and_metrics(
        self, batch: torch.Tensor, temperature: float, epsilon: float = 0
    ):
        r"""
        Method to implement for subclasses
        """
        return NotImplementedError()

    @staticmethod
    def check_model_is_consistent_with_policy(policy, model):
        if policy == "tb":
            assert isinstance(model.flow_model, BaseTBFlowModel)
        else:
            raise NotImplementedError()

    @staticmethod
    def choose_actions_from_PF(PF, epsilon):
        categorical = Categorical(logits=PF)
        choices = categorical.sample()

        if epsilon == 0:
            pass
        elif epsilon > 0:
            random_actions = torch.rand(state.shape[0], device=self.device) < epsilon
            random_choices = torch.randint(
                low=0, high=PF.shape[1], size=(torch.sum(random_actions).item(),)
            )
            choices[random_actions] = random_choices
        else:
            raise RuntimeError(f"Epsilon value is {epsilon}, but should be positive.")

        logprob = categorical.log_prob(choices)
        return choices, logprob
