from spingflow.modeling import setup_model_from_args
from spingflow.training.logging import create_summary_writer
from spingflow.training.optimizing import (
    setup_optimizer_from_args,
    setup_scheduler_from_args,
)
from spingflow.training.trainer import SpinGFlowTrainer
from spingflow.training.utils import create_train_parser, create_hparams_dict_from_args
import torch


def main(args):
    hparams = create_hparams_dict_from_args(args)

    # Get training device
    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA was asked, but is not available"
    device = torch.device(args.device)

    # Setup model for training
    model = setup_model_from_args(hparams)

    # Setup optimizer and scheduler
    optimizer = setup_optimizer_from_args(hparams, model)
    scheduler = setup_scheduler_from_args(hparams, optimizer)
    logger = create_summary_writer(args, hparams)

    # Create trainer from args
    trainer = SpinGFlowTrainer.create_from_args(
        hparams, model, optimizer, scheduler, logger, device
    )

    # Training
    trainer.train()
    trainer.log_hparams(hparams)
    print("Training complete!")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated()/(1024**3):.1f} GB")


if __name__ == "__main__":
    # Create parser and get arguments
    parser = create_train_parser()
    args = parser.parse_args()
    main(args)
