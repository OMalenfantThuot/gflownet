import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.distributions.categorical import Categorical
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
        device: torch.device = torch.device("cpu"),
        checkpoint_interval: int = 50,
        kept_checkpoints: int = 3,
        plotting_interval: int = 30,
    ):
        self.model = model
        self.temperature = temperature
        self.max_traj = max_traj
        self.batch_size = batch_size
        self.val_interval = val_interval
        self.val_batch_size = val_batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_interval = checkpoint_interval
        self.kept_checkpoints = kept_checkpoints
        self.plotting_interval = plotting_interval

    def train(self):
        self.model.to(self.device)

        # Set initial values
        n_traj, n_batches = 0, 0
        logZ_values = []
        val_n_traj, train_n_traj = [], []
        val_losses, train_losses = [], []
        val_counter, checkpoint_counter, plotting_counter = 0, 1, 0
        log_Z_converged = False

        # Training loop
        while n_traj < self.max_traj:
            # Check for validation
            if n_batches % self.val_interval == 0:
                # Validation trajectories
                _, val_loss = self.validation_step()

                # Save values
                val_losses.append(val_loss)
                val_n_traj.append(n_traj)
                logZ_values.append(self.model.flow_model.logZ.item())
                lr = self.optimizer.param_groups[0]["lr"]
                logZ_lr = self.optimizer.param_groups[1]["lr"]

                # Log values
                print(f"--n_traj: {n_traj:.3g}")
                print(
                    f"---- Val loss: {val_loss:.4g} --- Model logZ: {logZ_values[-1]:.4g}"
                )
                print(f"---- Learning rates: {lr:.4e} {logZ_lr:.4e}")

                # Checkpoint if needed
                val_counter, checkpoint_counter = self.checkpoint(
                    val_counter, checkpoint_counter
                )

                # Learning rate scheduling
                if log_Z_converged:
                    self.scheduler.step(val_loss)
                else:
                    if len(logZ_values) > 50:
                        logZ_history = np.array(logZ_values[-50:])
                        max_diff = np.max(np.abs(logZ_history - logZ_history[-1]))
                        if max_diff / logZ_history[-1] < 0.005:
                            log_Z_converged = True

                # Plotting if needed
                plotting_counter += 1
                if plotting_counter % self.plotting_interval == 0:
                    self.plot_metrics(
                        train_n_traj, train_losses, val_n_traj, val_losses, logZ_values
                    )

            # Training trajectories
            _, loss = self.training_step()
            n_traj += self.batch_size
            n_batches += 1

            # Save values
            train_losses.append(loss.item())
            train_n_traj.append(n_traj)

            # Backprop
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Final checkpointing and plotting
        _ = self.checkpoint(
            self.checkpoint_interval, checkpoint_counter + 1, final=True
        )
        self.plot_metrics(
            train_n_traj, train_losses, val_n_traj, val_losses, logZ_values
        )

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

    def plot_metrics(
        self, train_n_traj, train_losses, val_n_traj, val_losses, logZ_values
    ):
        # Metric plots to monitor progress during training
        # and assert results reliability after.
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 14))
        ax1, ax2, ax3 = ax
        ax1.plot(train_n_traj, train_losses, color="b", label="Train loss")
        ax1.plot(val_n_traj, val_losses, color="g", label="Val loss")
        ax1.set_ylim([0, None])
        ax2.semilogy(train_n_traj, train_losses, color="b", label="Train loss")
        ax2.semilogy(val_n_traj, val_losses, color="g", label="Val loss")
        ax3.plot(val_n_traj, logZ_values, color="b", label="Log Z")

        for iax in ax:
            iax.set_xlabel("Number of trajectories")
            iax.legend()
            iax.set_xlim([0, None])
        fig.tight_layout()
        fig.savefig("metrics.png")
