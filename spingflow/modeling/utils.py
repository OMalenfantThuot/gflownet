from spingflow.modeling.base import BaseFlowModel, IsingFullGFlowModel


def setup_model_from_args(args):
    if args.model_type is "simple":
        from spingflow.modeling.flow_models import SimpleIsingFlowModel

        return SimpleIsingFlowModel(N=args.N, n_hidden=256, n_layers=2)
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented.")
