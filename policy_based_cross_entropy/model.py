import torch
import numpy as np


class MLP(torch.nn.Module):
    """
    Since this model is used for policy-based problems, the output activation function is soft-max
    to estimate the probabilities corresponding to each action.
    """

    def __init__(self, n_input, n_output, n_hidden=None, random_state=110):

        assert (isinstance(n_hidden, list)) or (n_hidden is None), 'n_hidden should be either None or a list.'

        super(MLP, self).__init__()
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.n_output = n_output
        self.random_state = random_state
        self.device = torch.device("cpu")

        if n_hidden is None:
            nodes = [n_input, n_output]
        else:
            nodes = [n_input] + n_hidden + [n_output]
        self.nodes = nodes
        self.n_weights = sum([nodes[i] * nodes[i + 1] for i in range(len(nodes) - 1)])
        self.n_bias = sum([nodes[i + 1] for i in range(len(nodes) - 1)])
        self.tot_params = self.n_weights + self.n_bias

        if n_hidden is None:
            self.output_layer = torch.nn.Linear(n_input, n_output)
            self.weights_size = [self.output_layer.weight.shape]
            self.bias_size = [self.output_layer.bias.shape]

        else:
            self.hidden_layer = torch.nn.ModuleList()
            for i, n in enumerate(n_hidden):
                if i == 0:
                    layer = torch.nn.Linear(n_input, n)
                else:
                    layer = torch.nn.Linear(n_hidden[i - 1], n)
                self.hidden_layer.append(layer)
            self.output_layer = torch.nn.Linear(n_hidden[-1], n_output)

            self.weights_size = [x.weight.shape for x in self.hidden_layer] + [self.output_layer.weight.shape]
            self.bias_size = [x.bias.shape for x in self.hidden_layer] + [self.output_layer.bias.shape]

        # initialize the network
        self._initialize(random_state=self.random_state)

    def _initialize(self, random_state):
        """
        initializes the weights and parameters
        """
        np.random.seed(random_state)
        init_params = np.random.randn(self.tot_params)
        self.set_params_array(init_params)

    def set_params_array(self, params):

        assert isinstance(params, np.ndarray), 'params should be numpy array.'
        assert len(params) == self.tot_params, f'params length should be {self.tot_params}.'

        weights = params[:self.n_weights]
        bias = params[self.n_weights:]

        if self.n_hidden is None:
            w = torch.tensor(weights, dtype=torch.float32)
            self.output_layer.weight.data.copy_(w.view_as(self.output_layer.weight.data))

            b = torch.tensor(bias, dtype=torch.float32)
            self.output_layer.bias.data.copy_(b.view_as(self.output_layer.bias.data))

        else:
            w_sizes = [0] + list(np.cumsum([np.prod(list(x.weight.shape)) for x in self.hidden_layer])) + [
                self.n_weights]
            b_sizes = [0] + list(np.cumsum([np.prod(list(x.bias.shape)) for x in self.hidden_layer])) + [self.n_bias]

            for i, hidden_layer in enumerate(self.hidden_layer):
                w = torch.tensor(weights[w_sizes[i]:w_sizes[i + 1]], dtype=torch.float32)
                self.hidden_layer[i].weight.data.copy_(w.view_as(hidden_layer.weight.data))

                b = torch.tensor(bias[b_sizes[i]:b_sizes[i + 1]], dtype=torch.float32)
                self.hidden_layer[i].bias.data.copy_(b.view_as(hidden_layer.bias.data))

            w = torch.tensor(weights[w_sizes[-2]:w_sizes[-1]], dtype=torch.float32)
            self.output_layer.weight.data.copy_(w.view_as(self.output_layer.weight.data))

            b = torch.tensor(bias[b_sizes[-2]:b_sizes[-1]], dtype=torch.float32)
            self.output_layer.bias.data.copy_(b.view_as(self.output_layer.bias.data))

    def get_params_array(self):
        weights = []
        biases = []

        for name, param in self.named_parameters():
            if name.endswith('weight'):
                weights.append(param.data.cpu().numpy())
            if name.endswith('bias'):
                biases.append(param.data.cpu().numpy())

        weights = np.concatenate([x.reshape(-1) for x in weights])
        biases = np.concatenate([x.reshape(-1) for x in biases])
        params = np.concatenate([weights, biases])
        return params

    def forward(self, x):
        x_tensor = torch.from_numpy(x).to(self.device)
        if self.n_hidden is not None:
            for hidden_layer in self.hidden_layer:
                x_tensor = torch.nn.functional.relu(hidden_layer(x_tensor))
        z = torch.tanh(self.output_layer(x_tensor))
        return z
