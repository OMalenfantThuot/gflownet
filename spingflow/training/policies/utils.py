from spingflow.modeling import IsingFullGFlowModel


def get_policy(policy: str, model: IsingFullGFlowModel):
    if policy == "tb":
        from spingflow.training.policies.trajectory_balance import (
            TrajectoryBalancePolicy,
        )

        return TrajectoryBalancePolicy(model)
    else:
        raise NotImplementedError()
