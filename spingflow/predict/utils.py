import torch
import argparse
from spingflow.modeling.utils import add_modeling_arguments_to_parser


def create_inference_parser():
    # Initiate parser
    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Add modeling related arguments
    parser = add_modeling_arguments_to_parser(parser)

    # Add inference related arguments
    parser.add_argument("--savepath", help="Path to the saved state_dict")
    parser.add_argument("--nsamples", type=int, help="Number of structures to sample")
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for the prediction, only useful to reduce max memory usage",
    )
    parser.add_argument(
        "--property",
        choices=["magn"],
        help="Property to predict over the sampled structures",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], help="Inference device", default="cpu"
    )
    return parser
