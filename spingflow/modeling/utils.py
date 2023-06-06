from spingflow.modeling.tb_models import MlpTBFlowModel, ConvTBFlowModel
from spingflow.modeling.energy import IsingEnergyModel
from spingflow.modeling import IsingFullGFlowModel
import torch


def setup_model_from_args(args):
    flow_model = setup_flow_model_from_args(args)
    reward_model = setup_reward_model_from_args(args)
    return IsingFullGFlowModel(flow_model=flow_model, reward_model=reward_model)


def setup_flow_model_from_args(args):
    if args.policy == "tb":
        if args.model_type == "mlp":
            from spingflow.modeling.tb_models import MlpTBFlowModel

            return MlpTBFlowModel(
                N=args.N, n_layers=args.n_layers, n_hidden=args.n_hidden, logZ=args.initial_logZ
            )
        elif args.model_type == "conv":
            from spingflow.modeling.tb_models import ConvTBFlowModel

            conv_n_layers = args.conv_n_layers if args.conv_n_layers else args.n_layers
            return ConvTBFlowModel(
                N=args.N,
                conv_n_layers=conv_n_layers,
                mlp_n_layers=args.n_layers,
                mlp_n_hidden=args.n_hidden,
                conv_batch_norm=args.conv_norm,
                mlp_batch_norm=args.mlp_norm,
                logZ=args.initial_logZ,
            )
        else:
            raise NotImplementedError(
                f"Model type {args.model_type} is not implemented for the policy {args.policy}."
            )
    else:
        raise NotImplementedError(f"Policy {args.policy} is not implemented.")


def setup_reward_model_from_args(args):
    return IsingEnergyModel(N=args.N, J=args.J)


def add_modeling_arguments_to_parser(parser):
    parser.add_argument("--N", type=int, help="Size (side) of the spin grid")
    parser.add_argument(
        "--J", type=float, default=1, help="Ising interaction parameter."
    )
    parser.add_argument(
        "--model_type", choices=["mlp", "conv"], help="Name of the model type to use"
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
    parser.add_argument(
        "--conv_norm",
        action="store_true",
        help="Whether to use batch norm in the convolutional layers",
    )
    parser.add_argument(
        "--mlp_norm",
        action="store_true",
        help="Whether to use batch norm in the mlp layers",
    )
    return parser
