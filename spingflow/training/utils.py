import torch
import argparse


def create_train_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--N", type=int, help="Size (side) of the spin grid")
    parser.add_argument("--J", type=float, help="Ising interaction parameter.")
    parser.add_argument(
        "--epsilon", type=float, help="Epsilon parameter of the loss function."
    )
    parser.add_argument(
        "--model_type", choices=["simple"], help="Name of the model type to use"
    )
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--n_hidden", type=int, default=256, help="Number of hidden neurons")
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
