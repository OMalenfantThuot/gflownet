from torch.utils.tensorboard import SummaryWriter
import os


class CustomLogger(SummaryWriter):
    def __init__(self, log_dir, hparams_dict):
        super().__init__(log_dir=log_dir)
        self.hparams_dict = hparams_dict

    def log_hparams(self, final_metrics):
        self.add_hparams(self.hparams_dict, final_metrics)


def create_summary_writer(args, hparams_dict):
    if args.run_name:
        base_log_dir = os.path.join(args.log_dir, args.run_name)
    else:
        base_log_dir = os.path.join(
            args.log_dir, f"N{hparams_dict.N}_T{hparams_dict.temperature}"
        )

    log_dir_counter = 0
    while True:
        test_log_dir = base_log_dir + f"_{log_dir_counter:04}"
        if os.path.exists(test_log_dir):
            log_dir_counter += 1
        else:
            break

    logger = CustomLogger(log_dir=test_log_dir, hparams_dict=hparams_dict)
    return logger
