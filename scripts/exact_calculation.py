from spingflow.spins import StatesIterator, create_J_matrix, SpinConfiguration
import argparse
import numpy as np
import torch


def create_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--N", type=int, help="Size (side) of the spin grid")
    parser.add_argument("--J", type=float, help="Ising interaction parameter.")
    parser.add_argument(
        "--mode",
        choices=["exact", "metropolis"],
        help="Do either the exact full calculations or sample states through Metropolis.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for the calculation",
    )
    parser.add_argument(
        "--property",
        choices=["magn", "Z", "logZ"],
        help="Property to predict over the sampled structures",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=0,
        help="Number of states sampled to average the prediction on",
    )
    parser.add_argument(
        "--thermalize_steps",
        default=300,
        type=int,
        help="Number of thermalization steps in Metropolis",
    )
    return parser


def main(args):
    if args.mode == "exact":
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

    elif args.mode == "metropolis":
        if args.n_samples <= 0:
            raise RuntimeError(f"Number of samples is {args.n_samples}")
        if args.property in ["Z", "logZ"]:
            raise NotImplementedError()

        J = create_J_matrix(args.N, sigma=args.J)
        values = []
        for i in range(args.n_samples):
            spin = SpinConfiguration.create_random(N=args.N)
            spin.thermalize(J=J, T=args.temperature, nstep=args.thermalize_steps)

            if args.property == "magn":
                values.append(spin.get_magnetization().abs().item())
            else:
                raise NotImplementedError()

        print(float(np.mean(values)))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
