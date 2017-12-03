import torch.nn as nn
import torch.nn.functional as F

#################################################
# Model


class RCNN(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size,
            kernel_sizes,
            pooling_sizes,
            **kwargs):
        super(RCNN, self).__init__()
        assert len(hidden_sizes) == len(kernel_sizes)
        assert len(hidden_sizes) == len(pooling_sizes)
        assert len(hidden_sizes) > 0

        self.hidden_size = hidden_sizes[-1]
        self.output_size = output_size

        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()

        # Convolutional and pooling layers
        for i in range(len(kernel_sizes)):
            self.convs.append(
                nn.Conv1d(
                    # input_channels
                    input_size if i == 0 else hidden_sizes[i - 1],
                    hidden_sizes[i],  # output_channels
                    kernel_sizes[i],  # kernel_size
                    **kwargs))
            self.pools.append(nn.AvgPool1d(pooling_sizes[i]))

        # Gated recurrent unit layer (uses size of last convolutional layer)
        self.gru = nn.GRU(hidden_sizes[-1], hidden_sizes[-1])

        # Output layer
        self.out = nn.Linear(hidden_sizes[-1], output_size)

    # NOTE TO TRISTAN: Use tanh or sigmoid??? Paper uses sigmoids for
    # convolutional layers, but \
    # output should probably be a tanh

    def forward(self, input, hidden):
        # Feed only input vector through convs+pools, instead of
        # (input + hidden)
        # This is because the GRU incorporates the hidden layer using a gate
        # instead

        x = input
        x = x.transpose(0, 1)
        x = x.unsqueeze(0)

        # Convolutional and pooling layers
        for i in range(len(self.convs)):
            x = F.sigmoid(self.convs[i](x))
            x = self.pools[i](x)

        x = x.transpose(1, 2).transpose(0, 1)

        # Gated recurrent unit layer
        output, hidden = self.gru(x, hidden)

        output = output.view(output.size(0), self.hidden_size)

        # Output layer
        output = F.sigmoid(self.out(output))

        reduced = output.mean(0)

        return reduced, output, hidden
