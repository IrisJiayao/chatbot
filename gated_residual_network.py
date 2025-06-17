import torch
from torch import nn
from add_and_norm import AddAndNorm
from time_distributed_linear import LinearLayer
from gated_linear_unit import GLU

class GRN(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=None, dropout_rate=None, return_gate=False, use_time_distributed=True, batch_first=False):
        super(GRN, self).__init__()

        # Define output size (default is hidden size if not provided)
        self.output_size = hidden_layer_size if output_size is None else output_size
        self.return_gate = return_gate

        # Define Linear layers for main and residual paths
        self.linear_layer = LinearLayer(input_size, self.output_size, use_time_distributed, batch_first)
        self.hidden_linear_layer1 = LinearLayer(input_size, hidden_layer_size, use_time_distributed, batch_first)
        self.hidden_context_layer = LinearLayer(hidden_layer_size, hidden_layer_size, use_time_distributed, batch_first)
        self.hidden_linear_layer2 = LinearLayer(hidden_layer_size, hidden_layer_size, use_time_distributed, batch_first)

        # Define activation and normalization components
        self.elu = nn.ELU()
        self.glu = GLU(hidden_layer_size, self.output_size, dropout_rate, use_time_distributed, batch_first)
        self.add_and_norm = AddAndNorm(hidden_layer_size=self.output_size)

    def forward(self, x, context=None):
        # Residual connection path
        if self.output_size is None:
            skip = x
        else:
            skip = self.linear_layer(x)

        # Main transformation path
        hidden = self.hidden_linear_layer1(x)
        if context is not None:
            hidden = hidden + self.hidden_context_layer(context)
        hidden = self.elu(hidden)
        hidden = self.hidden_linear_layer2(hidden)

        # Gating mechanism through GLU
        gate_layer, gate = self.glu(hidden)

        # Residual connection followed by normalization
        if self.return_gate:
            return self.add_and_norm(skip, gate_layer), gate
        else:
            return self.add_and_norm(skip, gate_layer)


# # ✅ GRN testing code
# if __name__ == "__main__":
#     # Define test configuration
#     batch_size = 5       # Number of samples
#     time_steps = 10      # Number of time steps per sample
#     input_size = 8       # Input feature size per time step
#     hidden_size = 16     # Hidden layer size of GRN
#     output_size = 12     # Output feature size of GRN (defaults to hidden_size if None)
#     dropout_rate = 0.1   # Dropout rate for regularization
#
#     # Instantiate the GRN module
#     grn = GRN(
#         input_size=input_size,
#         hidden_layer_size=hidden_size,
#         output_size=output_size,
#         dropout_rate=dropout_rate,
#         batch_first=True,   # Input shape format: (batch_size, time_steps, feature_dim)
#         return_gate=True    # Return gate weights for inspection/debugging
#     )
#
#     # Generate random input: (batch_size, time_steps, input_size)
#     x = torch.rand(batch_size, time_steps, input_size)
#
#     # Generate optional random context input
#     context = torch.rand(batch_size, time_steps, hidden_size)
#
#     # Perform forward pass
#     output, gate = grn(x, context)
#
#     # Display output shapes
#     print("GRN output shape:", output.shape)  # Expected: (5, 10, 12)
#     print("GRN gate weights shape:", gate.shape)  # Expected: (5, 10, 12)