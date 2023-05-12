from spingflow.modeling import BaseFlowModel
import torch


def setup_mlp(input_dim, output_dim, n_layers, n_hidden, batch_norm=False):
    net = torch.nn.Sequential(
        torch.nn.Linear(input_dim, n_hidden), torch.nn.LeakyReLU()
    )
    if n_layers > 2:
        for _ in range(n_layers - 2):
            if batch_norm:
                net.append(torch.nn.BatchNorm1d(n_hidden))
            net.append(torch.nn.Linear(n_hidden, n_hidden))
            net.append(torch.nn.LeakyReLU())
    if batch_norm:
        net.append(torch.nn.BatchNorm1d(n_hidden))
    net.append(torch.nn.Linear(n_hidden, output_dim))
    return net


class SimpleIsingFlowModel(BaseFlowModel):
    def __init__(self, N, n_layers=2, n_hidden=256):
        N = int(N)
        internal_net = setup_mlp(
            input_dim=2 * N**2,
            output_dim=2 * 2 * N**2,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )
        super().__init__(N=N, internal_net=internal_net)


class ConvolutionsModule(torch.nn.Module):
    def __init__(self, N, conv_n_layers=3, batch_norm=False):
        super().__init__()

        self.N = int(N)
        self.batch_norm = batch_norm

        self.layers = torch.nn.ModuleList()
        if self.batch_norm:
            self.norm_layers = torch.nn.ModuleList()

        in_channels, out_channels = 1, 32
        for _ in range(conv_n_layers):
            layer = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=0,
            )
            self.layers.append(layer)
            if self.batch_norm:
                self.norm_layers.append(torch.nn.BatchNorm2d(out_channels))
            in_channels = out_channels
            out_channels = out_channels // 2

        self.output_dim = in_channels * self.N**2
        self.activation = torch.nn.LeakyReLU()

    def forward(self, states):
        states = (states[..., : self.N**2] + -1 * states[..., self.N**2 :]).reshape(
            -1, 1, self.N, self.N
        )
        for i in range(len(self.layers)):
            states = torch.nn.functional.pad(states, pad=(1, 1, 1, 1), mode="circular")
            states = self.layers[i](states)
            states = self.activation(states)
            if self.batch_norm:
                states = self.norm_layers[i](states)

        states = states.reshape(-1, self.output_dim)
        return states


class ConvIsingFlowModel(BaseFlowModel):
    def __init__(
        self,
        N,
        conv_n_layers=3,
        mlp_n_layers=2,
        mlp_n_hidden=256,
        conv_batch_norm=False,
        mlp_batch_norm=False,
    ):
        N = int(N)
        convolutions_module = ConvolutionsModule(
            N=N, conv_n_layers=conv_n_layers, batch_norm=conv_batch_norm
        )
        mlp_module = setup_mlp(
            input_dim=convolutions_module.output_dim,
            output_dim=2 * 2 * N**2,
            n_layers=mlp_n_layers,
            n_hidden=mlp_n_hidden,
            batch_norm=mlp_batch_norm,
        )
        internal_net = torch.nn.Sequential(convolutions_module, mlp_module)
        super().__init__(N=N, internal_net=internal_net)
