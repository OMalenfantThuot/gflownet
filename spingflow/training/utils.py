import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from spingflow.modeling.utils import add_modeling_arguments_to_parser


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
        help="Multiplicative factor to get the ",
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


def create_summary_writer(args):
    if args.run_name:
        base_log_dir = os.path.join(args.log_dir, args.run_name)
    else:
        base_log_dir = os.path.join(args.log_dir, f"N{args.N}_T{args.temperature}")

    log_dir_counter = 0
    while True:
        test_log_dir = base_log_dir + f"_{log_dir_counter:04}"
        if os.path.exists(test_log_dir):
            log_dir_counter += 1
        else:
            break

    writer = SummaryWriter(log_dir=test_log_dir)
    return writer
