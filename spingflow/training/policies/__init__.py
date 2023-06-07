from spingflow.modeling.tb_models import BaseTBFlowModel
from spingflow.training.policies.utils import get_policy
import torch


class BasePolicy:
    def __init__(self, policy, model):
        self.check_model_is_consistent_with_policy(policy, model)
        self.policy = policy
        self.model = model

    def training_trajectory_and_metrics(self, batch: torch.Tensor):
        return NotImplementedError()

    @staticmethod
    def check_model_is_consistent_with_policy(policy, model):
        if policy == "tb":
            assert isinstance(model.flow_model, BaseTBFlowModel)
        else:
            raise NotImplementedError()
