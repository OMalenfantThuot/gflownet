import torch
import numpy as np
import argparse
from spingflow.spins import StatesIterator, create_J_matrix


def create_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--N", type=int, help="Size (side) of the spin grid")
    parser.add_argument("--J", type=float, help="Ising interaction parameter.")
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature to train the model on",
    )
    parser.add_argument(
        "--property",
        choices=["magn"],
        help="Property to predict over the sampled structures",
    )
    return parser


def main(args):
    if args.N > 5:
        print("N is high, this will be very long.")
    states_iterator = StatesIterator(N=args.N)
    J = create_J_matrix(args.N, sigma=args.J)
    Z, property = 0, 0

    for state in states_iterator:
        energy = state.get_energy(J)
        relative_prob = torch.exp(-1 * energy / args.temperature).item()
        Z += relative_prob

        if args.property == "magn":
            mag = state.get_magnetization().abs().item()
            property += mag * relative_prob
        else:
            raise NotImplementedError()
    property = property / Z
    print(property)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
