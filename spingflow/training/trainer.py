import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from spingflow.modeling import IsingFullGFlowModel


class SpinGFlowTrainer:
    def __init__(
        self,
        model: IsingFullGFlowModel,
        temperature: float,
        max_traj: int,
        batch_size: int,
        val_interval: int,
        val_batch_size: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        logger: SummaryWriter,
        device: torch.device = torch.device("cpu"),
        checkpoint_interval: int = 50,
        kept_checkpoints: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_traj = max_traj
        self.batch_size = batch_size
        self.val_interval = val_interval
        self.val_batch_size = val_batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.device = device
        self.checkpoint_interval = checkpoint_interval
        self.kept_checkpoints = kept_checkpoints
        self.final_metrics = None

    def train(self):
        self.model.to(self.device)

        # Set initial values
        n_traj, n_batches = 0, 0
        logZ_values = []
        # val_n_traj, val_losses = [], []
        val_counter, checkpoint_counter, plotting_counter = 0, 1, 0
        logZ_converged = False

        # Training loop
        while n_traj < self.max_traj:
            # Check for validation
            if n_batches % self.val_interval == 0:
                # Validation trajectories
                _, val_loss = self.validation_step()

                # Save values
                lr = self.optimizer.param_groups[0]["lr"]
                logZ_lr = self.optimizer.param_groups[1]["lr"]

                self.logger.add_scalar("val/loss", val_loss, n_traj)
                self.logger.add_scalar(
                    "logZ", self.model.flow_model.logZ.item(), n_traj
                )
                self.logger.add_scalars(
                    "learning_rates", {"lr": lr, "logZ_lr": logZ_lr}, n_traj
                )
                self.logger.add_scalar("logZ_converged", logZ_converged, n_traj)

                # Log values
                print(f"--n_traj: {n_traj:.3g}")
                print(
                    f"---- Val loss: {val_loss:.4g} --- Model logZ: {self.model.flow_model.logZ.item():.4g}"
                )
                print(f"---- Learning rates: {lr:.4e} {logZ_lr:.4e}")

                # Checkpoint if needed
                val_counter, checkpoint_counter = self.checkpoint(
                    val_counter, checkpoint_counter
                )

                # Learning rate scheduling
                if logZ_converged:
                    self.scheduler.step(val_loss)
                else:
                    logZ_values.append(self.model.flow_model.logZ.item())
                    if len(logZ_values) > 50:
                        logZ_history = np.array(logZ_values[-50:])
                        max_diff = np.max(np.abs(logZ_history - logZ_history[-1]))
                        if max_diff / logZ_history[-1] < 0.005:
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
        # self.plot_metrics(val_n_traj, val_losses, logZ_values)
        self.calculate_final_metrics(logZ_converged)

    def validation_step(self):
        # Batched trajectories without gradient accumulation
        self.model.eval()
        batch = self.model.create_input_batch(batch_size=self.val_batch_size).to(
            self.device
        )
        state, loss = self.training_trajectory_and_metrics(batch)
        return state, loss.item()

    def training_step(self):
        # Batched trajectories with gradient accumulation
        self.model.train()
        batch = self.model.create_input_batch(batch_size=self.batch_size).to(
            self.device
        )
        state, loss = self.training_trajectory_and_metrics(batch)
        return state, loss

    def training_trajectory_and_metrics(self, state):
        # Evaluate initial state (S_0)
        PF, PB, _ = self.model.flow_model.get_logits(state)
        traj_PF, traj_PB = 0, 0

        # Loop on following decisions
        for step in range(1, self.model.N**2 + 1):
            # Use forward probabilities and make choice
            categorical = Categorical(logits=PF)
            choice = categorical.sample()
            new_state = self.model.flow_model.create_new_state_from_choice(
                state, choice
            )
            traj_PF += categorical.log_prob(choice)

            # Check if we are at terminal state
            if step == self.model.N**2:
                # Calculate reward
                logreward = self.model.reward_model.get_logreward(
                    new_state, self.temperature
                )

            # Get forward and backward probabilities for new state
            PF, PB, _ = self.model.flow_model.get_logits(new_state)
            traj_PB += Categorical(logits=PB).log_prob(choice)

            # Reset for next loop
            state = new_state

        # Trajectory balance loss function
        loss = (self.model.flow_model.logZ + traj_PF - traj_PB - logreward) ** 2

        return state, loss.mean()

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

    def calculate_final_metrics(self, logZ_converged):
        val_losses = []
        for _ in range(int(np.ceil(10**6 // self.val_batch_size))):
            val_losses.append(self.validation_step()[1])

        final_val_loss = np.mean(val_losses)
        logZ = self.model.flow_model.logZ.item()
        self.final_metrics = {
            "final/val/loss": final_val_loss,
            "logZ": logZ,
            "logZ/converged": logZ_converged,
        }

    def log_hparams(self, args):
        hparams_dict = {
            "N": args.N,
            "J": args.J,
            "temperature": args.temperature,
            "max_traj": args.max_traj,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "logZ_lr_factor": args.logZ_lr_factor,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "factor": args.factor,
            "model_type": args.model_type,
            "n_layers": args.n_layers,
            "n_hidden": args.n_hidden,
            "conv_n_layers": args.conv_n_layers,
            "conv_norm": args.conv_norm,
            "mlp_norm": args.mlp_norm,
        }
        self.logger.add_hparams(hparams_dict, self.final_metrics)
