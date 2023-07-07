from spingflow.modeling import IsingFullGFlowModel


def get_policy(policy: str, model: IsingFullGFlowModel):
    if policy == "tb":
        from spingflow.training.policies.trajectory_balance import (
            TrajectoryBalancePolicy,
        )

        return TrajectoryBalancePolicy(model)
    elif policy == "db":
        from spingflow.training.policies.detailed_balance import DetailedBalancePolicy

        return DetailedBalancePolicy(model)
    else:
        raise NotImplementedError()
