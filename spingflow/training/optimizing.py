import torch


def setup_optimizer_from_args(hparams, model):
    if hparams.policy == "tb":
        optimizer = torch.optim.Adam(
            [
                {
                    "params": list(
                        p[1]
                        for p in filter(
                            lambda p: p[0] != "flow_model.logZ",
                            model.named_parameters(),
                        )
                    ),
                    "lr": hparams.lr,
                },
                {
                    "params": model.flow_model.logZ,
                    "lr": hparams.lr * hparams.logZ_lr_factor,
                },
            ]
        )
        return optimizer
    elif hparams.policy == "db":
        # for p in model.named_parameters():
        #    print(p[0])
        optimizer = torch.optim.Adam(
            [
                {
                    "params": list(
                        p[1]
                        for p in filter(
                            lambda p: not p[0].startswith("flow_model.state_flow_net"),
                            model.named_parameters(),
                        )
                    ),
                    "lr": hparams.lr,
                },
                {
                    "params": list(
                        p[1]
                        for p in filter(
                            lambda p: p[0].startswith("flow_model.state_flow_net"),
                            model.named_parameters(),
                        )
                    ),
                    "lr": hparams.lr * hparams.logZ_lr_factor,
                },
            ]
        )
        return optimizer
    else:
        raise NotImplementedError()


def setup_scheduler_from_args(hparams, optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=hparams.factor,
        patience=hparams.patience,
        min_lr=3 * 10 ** (-6),
    )
    return scheduler
