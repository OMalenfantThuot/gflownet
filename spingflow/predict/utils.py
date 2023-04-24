import torch
import argparse


def create_inference_parser():
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
    parser.add_argument("--savepath", help="Path to the saved state_dict")
    parser.add_argument("--nsamples", type=int, help="Number of structures to sample")
    parser.add_argument("--batch_size", type=int, help="Batch size for the prediction, only useful to reduce max memory usage")
    parser.add_argument(
        "--property",
        choices=["magn"],
        help="Property to predict over the sampled structures",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], help="Inference device", default="cpu"
    )
    return parser