import torch
from spingflow.modeling import IsingFullGFlowModel
from spingflow.training.utils import create_input_batch
from torch.distributions.categorical import Categorical


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
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: str = "cpu",
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

    def train(self):
        n_traj, n_batches, val_counter = 0, 0, 0
        train_losses, val_losses = [], []
        best_loss = np.inf

        while n_traj < self.max_traj:
            if n_batches % val_interval == 0:
                _, val_loss = self.validation_step()
                val_losses.append(val_loss)
                print("--n_traj: ", n_traj)
                print(f"---- Val loss: {val_loss} --- Model logZ: {model.logZ.item()}")

                best_loss, val_counter = self.checkpoint(val_loss, best_loss)
                self.scheduler.step(val_loss)

                # if val_counter >= val_patience:
                #    print("Reducing learning rate")
                #    for p in optimizer.param_groups:
                #        p["lr"] *= gamma
                #    val_counter = 0

            _, loss = self.training_step()
            n_traj += self.batch_size
            n_batches += 1

            train_losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        # Stop if scheduler hits minimum learning rate?

    def validation_step(self):
        self.model.eval()
        batch = create_input_batch(N=self.model.N, batch_size=self.val_batch_size).to(
            self.device
        )
        state, loss = self.training_trajectory_and_metrics(batch)
        return state, loss.item()

    def training_step(self):
        self.model.train()
        batch = create_input_batch(N=self.model.N, batch_size=self.batch_size).to(
            self.device
        )
        state, loss = self.training_trajectory_and_metrics(batch)
        return state, loss

    def training_trajectory_and_metrics(self, batch):
        PF, PB = self.model.flow_model.get_logits(batch)
        traj_PF, traj_PB = 0, 0

        for step in range(1, self.model.N**2 + 1):
            categorical = Categorical(logits=PF)
            choice = categorical.sample()
            new_state = self.model.flow_model.create_new_state_from_choice(
                batch, choice
            )
            traj_PF += categorical.log_prob(choice)

            if step == self.model.N**2:
                reward = self.model.reward_model(new_state, self.temperature)

            PF, PB = self.model.flow_model.get_logits(batch)
            traj_PB += Categorical(logits=PB).log_prob(choice)

            state = new_state

        loss = (
            self.model.flow_model.logZ + traj_PF - traj_PB - torch.log(reward).clip(-20)
        ) ** 2

        return state, loss.mean()

    def checkpoint(self, val_loss, best_loss, val_counter):
        if val_loss < best_loss:
            val_counter = 0
            best_loss = val_loss
            torch.save(self.model.state_dict(), "best_model.pth")
        else:
            val_counter += 1
        return best_loss, val_counter