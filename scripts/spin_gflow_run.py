import torch
from spingflow.modeling import setup_model_from_args
from spingflow.predict.utils import create_inference_parser
from spingflow.predict import SpinGFlowPredictor


def main(args):
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA was asked, but is not available"
        device = torch.device(args.device)
    elif args.device == "cpu":
        device = torch.device(args.device)

    model = setup_model_from_args(args)
    model.load_state_dict(torch.load(args.savepath, map_location=device))

    if args.property == "logZ":
        print(model.flow_model.logZ.item())

    else:
        predictor = SpinGFlowPredictor(
            model=model,
            nsamples=args.nsamples,
            batch_size=args.batch_size,
            device=device,
        )
        prediction = predictor.predict(args.property).item()
        print(prediction)


if __name__ == "__main__":
    parser = create_inference_parser()
    args = parser.parse_args()
    main(args)
