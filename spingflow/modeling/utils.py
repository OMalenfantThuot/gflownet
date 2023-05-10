from spingflow.modeling.base import BaseFlowModel, IsingFullGFlowModel
from spingflow.modeling.energy import IsingEnergyModel


def setup_model_from_args(args):
    flow_model = setup_flow_model_from_args(args)
    reward_model = setup_reward_model_from_args(args)
    return IsingFullGFlowModel(flow_model=flow_model, reward_model=reward_model)


def setup_flow_model_from_args(args):
    if args.model_type == "simple":
        from spingflow.modeling.flow_models import SimpleIsingFlowModel

        return SimpleIsingFlowModel(
            N=args.N, n_layers=args.n_layers, n_hidden=args.n_hidden
        )
    elif args.model_type == "conv":
        from spingflow.modeling.flow_models import ConvIsingFlowModel

        conv_n_layers = args.conv_n_layers if args.conv_n_layers else args.n_layers
        return ConvIsingFlowModel(
            N=args.N,
            conv_n_layers=conv_n_layers,
            mlp_n_layers=args.n_layers,
            mlp_n_hidden=args.n_hidden,
        )
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented.")


def setup_reward_model_from_args(args):
    return IsingEnergyModel(N=args.N, J=args.J, device=args.device)


def add_modeling_arguments_to_parser(parser):
    parser.add_argument("--N", type=int, help="Size (side) of the spin grid")
    parser.add_argument(
        "--J", type=float, default=1, help="Ising interaction parameter."
    )
    parser.add_argument(
        "--model_type", choices=["simple", "conv"], help="Name of the model type to use"
    )
    parser.add_argument(
        "--n_layers", type=int, default=2, help="Number of layers in the model"
    )
    parser.add_argument(
        "--n_hidden", type=int, default=256, help="Number of hidden neurons"
    )
    parser.add_argument(
        "--conv_n_layers", type=int, default=2, help="Number of convolutional layers"
    )
    return parser
