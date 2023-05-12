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
        choices=["magn", "Z", "logZ"],
        help="Property to predict over the sampled structures",
    )
    return parser


def main(args):
    if args.N > 5:
        print("N is high, this will be very long.")
    states_iterator = StatesIterator(N=args.N)
    J = create_J_matrix(args.N, sigma=args.J)
    Z, property = np.longdouble(0), np.longdouble(0)

    for state in states_iterator:
        energy = state.get_energy(J)
        log_relative_prob = np.longdouble((-1 * energy / args.temperature).item())
        relative_prob = np.exp(log_relative_prob)
        Z += relative_prob

        if args.property == "magn":
            mag = state.get_magnetization().abs().item()
            property += mag * relative_prob
        elif args.property in ["Z", "logZ"]:
            pass
        else:
            raise NotImplementedError()
    if args.property not in ["Z", "logZ"]:
        property = property / Z
        print(property)
    elif args.property == "logZ":
        print(np.log(Z))
    elif args.property == "Z":
        print(Z)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
