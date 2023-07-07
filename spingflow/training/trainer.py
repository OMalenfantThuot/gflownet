from spingflow.modeling import IsingFullGFlowModel
from spingflow.training.policies import get_policy
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
import numpy as np
import os
import torch


class SpinGFlowTrainer:
    def __init__(
        self,
        model: IsingFullGFlowModel,
        policy: str,
        temperature: float,
        max_traj: int,
        batch_size: int,
        val_interval: int,
        val_batch_size: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        logger: SummaryWriter,
        epsilon: Optional[float] = 0.0,
        device: Optional[torch.device] = torch.device("cpu"),
        checkpoint_interval: Optional[int] = 50,
        kept_checkpoints: Optional[int] = 3,
    ):
        self.model = model
        self.policy = get_policy(policy, self.model)
        self.temperature = temperature
        self.max_traj = max_traj
        self.batch_size = batch_size
        self.val_interval = val_interval
        self.val_batch_size = val_batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.epsilon = epsilon
        self.device = device
        self.checkpoint_interval = checkpoint_interval
        self.kept_checkpoints = kept_checkpoints
        self.final_metrics = None

    def train(self):
        self.model.to(self.device)

        # Set initial values
        n_traj, n_batches = 0, 0
        logZ_values = []
        val_counter, checkpoint_counter, plotting_counter = 0, 1, 0
        logZ_converged = False

        # Training loop
        while n_traj < self.max_traj:
            # Check for validation
            if n_batches % self.val_interval == 0:
                # Validation trajectories
                _, val_loss, logZ = self.validation_step()
                val_dict = {"val/loss": val_loss, "logZ_converged": logZ_converged}

                self.log_validation_values(n_traj, val_dict, logZ)

                # Checkpoint if needed
                val_counter, checkpoint_counter = self.checkpoint(
                    val_counter, checkpoint_counter
                )

                # Learning rate scheduling
                if logZ_converged:
                    self.scheduler.step(val_loss)
                else:
                    logZ_values.append(logZ)
                    if len(logZ_values) > 50:
                        logZ_history = np.array(logZ_values[-50:])
                        max_diff = np.max(np.abs(logZ_history - logZ_history[-1]))
                        if max_diff / logZ_history[-1] < 0.002:
                            logZ_converged = True
                        logZ_values = logZ_values[-50:]

            # Training trajectories
            _, loss = self.training_step()
            n_traj += self.batch_size
            n_batches += 1

            # Backprop
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.logger.flush()
        # Final checkpointing and plotting
        _ = self.checkpoint(
            self.checkpoint_interval, checkpoint_counter + 1, final=True
        )
        self.calculate_final_metrics()

    def validation_step(self):
        # Batched trajectories without gradient accumulation
        self.model.eval()
        batch = self.model.create_input_batch(batch_size=self.val_batch_size).to(
            self.device
        )
        state, loss = self.policy.training_trajectory_and_metrics(
            batch, self.temperature, self.epsilon
        )
        logZ = self.model.get_current_logZ().item()
        return state, loss.item(), logZ

    def training_step(self):
        # Batched trajectories with gradient accumulation
        self.model.train()
        batch = self.model.create_input_batch(batch_size=self.batch_size).to(
            self.device
        )
        state, loss = self.policy.training_trajectory_and_metrics(
            batch, self.temperature, self.epsilon
        )
        return state, loss

    def log_validation_values(self, n_traj, val_dict, logZ):
        # Save values
        lr = self.optimizer.param_groups[0]["lr"]
        logZ_lr = self.optimizer.param_groups[1]["lr"]

        for k, v in val_dict.items():
            self.logger.add_scalar(k, v, n_traj)
        self.logger.add_scalar("logZ", logZ, n_traj)
        self.logger.add_scalars(
            "learning_rates", {"lr": lr, "logZ_lr": logZ_lr}, n_traj
        )

        # print values (maybe no longer needed)
        print(f"--n_traj: {n_traj:.3g}")
        print(f"---- Val loss: {val_dict['val/loss']:.4g} --- Model logZ: {logZ:.4g}")
        print(f"---- Learning rates: {lr:.4e} {logZ_lr:.4e}")

    def checkpoint(self, val_counter, checkpoint_counter, final=False):
        # Check if checkpoint is needed
        if val_counter == self.checkpoint_interval:
            if not final:
                torch.save(self.model.state_dict(), f"ckpt_{checkpoint_counter}.torch")
                try:
                    os.remove(f"ckpt_{checkpoint_counter-self.kept_checkpoints}.torch")
                except FileNotFoundError:
                    pass
            else:
                torch.save(self.model.state_dict(), "final_model.torch")
            # Reset counter
            val_counter = 0
            checkpoint_counter += 1
        else:
            # Increment counter
            val_counter += 1
        return val_counter, checkpoint_counter

    def calculate_final_metrics(self):
        val_losses = []
        for _ in range(int(np.ceil(10**6 // self.val_batch_size))):
            val_losses.append(self.validation_step()[1])

        final_val_loss = np.mean(val_losses)
        logZ = self.model.get_current_logZ().item()
        self.final_metrics = {
            "final/val/loss": final_val_loss,
            "logZ": logZ,
        }

    def log_hparams(self, hparams):
        self.logger.add_hparams(dict(hparams.items()), self.final_metrics)

    @classmethod
    def create_from_args(cls, hparams, model, optimizer, scheduler, logger, device):
        trainer = SpinGFlowTrainer(
            model=model,
            policy=hparams.policy,
            temperature=hparams.temperature,
            max_traj=hparams.max_traj,
            batch_size=hparams.batch_size,
            val_interval=hparams.val_interval,
            val_batch_size=hparams.val_batch_size,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            device=device,
        )
        return trainer
