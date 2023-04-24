from spingflow.modeling.base import BaseFlowModel, IsingFullGFlowModel
from spingflow.modeling.energy import IsingEnergyModel


def setup_model_from_args(args):
    flow_model = setup_flow_model_from_args(args)
    reward_model = setup_reward_model_from_args(args)
    return IsingFullGFlowModel(flow_model=flow_model, reward_model=reward_model)


def setup_flow_model_from_args(args):
    if args.model_type == "simple":
        from spingflow.modeling.flow_models import SimpleIsingFlowModel

        return SimpleIsingFlowModel(N=args.N, n_layers=args.n_layers, n_hidden=args.n_hidden)
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented.")


def setup_reward_model_from_args(args):
    return IsingEnergyModel(N=args.N, J=args.J, device=args.device)


