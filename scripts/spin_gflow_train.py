from spingflow.training.utils import create_train_parser
from spingflow.modeling import setup_model_from_args
from spingflow.training.trainer import SpinGFlowTrainer
import torch


def main(args):
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA was asked, but is not available"
        device = torch.device(args.device)
    elif args.device == "cpu":
        device = torch.device(args.device)

    model = setup_model_from_args(args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.factor, patience=args.patience
    )

    trainer = SpinGflowTrainer(
        model=model,
        temperature=args.temperature,
        max_traj=args.max_traj,
        batch_size=args.batch_size,
        val_interval=args.val_interval,
        val_batch_size=args.val_batch_size,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    parser = create_train_parser()
    args = parser.parse_args()
    main(args)
