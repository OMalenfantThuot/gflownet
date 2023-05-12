from spingflow.training.utils import create_train_parser
from spingflow.modeling import setup_model_from_args
from spingflow.training.trainer import SpinGFlowTrainer
import torch


def main(args):
    # Get training device
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA was asked, but is not available"
        device = torch.device(args.device)
    elif args.device == "cpu":
        device = torch.device(args.device)

    # Setup model for training
    model = setup_model_from_args(args)

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(
        [
            {
                "params": list(
                    p[1]
                    for p in filter(
                        lambda p: p[0] != "flow_model.logZ", model.named_parameters()
                    )
                ),
                "lr": args.lr,
            },
            {"params": model.flow_model.logZ, "lr": args.lr * args.logZ_lr_factor},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.factor,
        patience=args.patience,
        min_lr=3 * 10 ** (-6),
    )

    # Create trainer from args
    trainer = SpinGFlowTrainer(
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

    # Training
    trainer.train()
    print("Training complete!")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated()/(1024**3):.1f} GB")


if __name__ == "__main__":
    # Create parser and get arguments
    parser = create_train_parser()
    args = parser.parse_args()
    main(args)
