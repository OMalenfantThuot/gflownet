import torch
import argparse
from spingflow.modeling.utils import add_modeling_arguments_to_parser


def create_train_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser = add_modeling_arguments_to_parser(parser)

    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature to train the model on",
    )
    parser.add_argument("--max_traj", type=int, help="Maximum number of trajectories")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument(
        "--val_interval", type=int, help="Number of batches in between validation steps"
    )
    parser.add_argument("--val_batch_size", type=int, help="Validation batch size")
    parser.add_argument("--lr", type=float, help="Initial learning rate")
    parser.add_argument("--patience", type=int, help="Scheduler patience")
    parser.add_argument("--factor", type=float, help="Scheduler reduction factor")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Training device")
    return parser
