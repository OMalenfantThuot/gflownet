from spingflow.modeling.utils import add_modeling_arguments_to_parser
import argparse


def create_train_parser():
    # Initialize parser
    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Add modeling related arguments
    parser = add_modeling_arguments_to_parser(parser)

    # Add training related arguments
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature to train the model on",
    )
    parser.add_argument(
        "--max_traj",
        type=lambda x: int(float(x)),
        help="Maximum number of trajectories",
    )
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument(
        "--val_interval", type=int, help="Number of batches in between validation steps"
    )
    parser.add_argument("--val_batch_size", type=int, help="Validation batch size")
    parser.add_argument("--lr", type=float, help="Initial learning rate")
    parser.add_argument(
        "--logZ_lr_factor",
        type=float,
        default=100,
        help="Multiplicative factor to get the logZ learning rate",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="Weight decay parameter"
    )
    parser.add_argument("--patience", type=int, help="Scheduler patience")
    parser.add_argument("--factor", type=float, help="Scheduler reduction factor")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Training device")
    parser.add_argument(
        "--log_dir",
        default="../../logs",
        help="Top level directory for Tensorboard logs",
    )
    parser.add_argument("--run_name", default=None, help="Name of the run")
    return parser


class HparamsDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value


def create_hparams_dict_from_args(args):
    hparams_dict = {}
    valid_hparams = (
        "N",
        "J",
        "temperature",
        "policy",
        "max_traj",
        "batch_size",
        "lr",
        "logZ_lr_factor",
        "weight_decay",
        "patience",
        "factor",
        "model_type",
        "n_layers",
        "n_hidden",
        "conv_n_layers",
        "conv_norm",
        "mlp_norm",
        "val_interval",
        "val_batch_size",
        "initial_logZ",
    )
    for key, value in vars(args).items():
        if key in valid_hparams:
            hparams_dict[key] = value
    return HparamsDict(hparams_dict)
