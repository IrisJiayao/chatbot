import torch
from torch import nn
from gated_residual_network import GRN  # Your custom GRN module

class LSTMCombine(nn.Module):
    """
    Implements "Scheme C" for variable selection:
    - Assumes input shape: [batch, time, embedding_dim, num_vars]
    - Each variable is first projected via a GRN individually.
    - The results are flattened and passed through another GRN to produce attention weights.
    """

    def __init__(
        self,
        embedding_dim,        # Embedding dimension of each variable
        num_vars,             # Total number of variables
        hidden_layer_size,    # Output size after GRN projection
        dropout_rate,
        use_time_distributed=False,
        batch_first=True
    ):
        super(LSTMCombine, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_vars = num_vars
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed
        self.batch_first = batch_first

        # 1) GRN to compute selection weights from the flattened vector
        #    Input = num_vars * hidden_size, Output = num_vars (1 score per variable)
        self.flattened_grn = GRN(
            input_size=self.num_vars * self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            output_size=self.num_vars,
            dropout_rate=self.dropout_rate,
            return_gate=True,  # Optionally returns the static gate
            use_time_distributed=self.use_time_distributed,
            batch_first=self.batch_first
        )

        # 2) Per-variable GRNs: one GRN for each variable
        #    Each maps from embedding_dim -> hidden_layer_size
        self.single_variable_grns = nn.ModuleList([
            GRN(
                input_size=self.embedding_dim,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,  # defaults to hidden_layer_size
                dropout_rate=self.dropout_rate,
                return_gate=False,
                use_time_distributed=self.use_time_distributed,
                batch_first=self.batch_first
            )
            for _ in range(self.num_vars)
        ])

        # 3) Softmax to normalize selection scores into attention weights
        self.softmax = nn.Softmax(dim=2)  # Apply along the num_vars dimension

    def forward(self, embedding, additional_context=None):
        """
        Args:
            embedding: Tensor of shape [batch, time, embedding_dim, num_vars]
            additional_context: Optional static context of shape [batch, hidden_layer_size]

        Returns:
            temporal_ctx: [batch, time, hidden_layer_size] - weighted combination result
            sparse_weights: [batch, time, 1, num_vars] - soft attention weights per variable
            static_gate: [batch, time, num_vars] or None
        """
        batch_size, time_steps, embed_dim, num_vars = embedding.shape

        assert embed_dim == self.embedding_dim, "Embedding dim mismatch"
        assert num_vars == self.num_vars, "Number of variables mismatch"

        # Project each variable separately using GRNs
        trans_emb_list = []
        for i in range(num_vars):
            var_i = embedding[..., i]  # [batch, time, embedding_dim]
            var_i_transformed = self.single_variable_grns[i](var_i)  # -> [batch, time, hidden_size]
            trans_emb_list.append(var_i_transformed)

        # Stack along last dimension -> [batch, time, hidden_size, num_vars]
        transformed_embedding = torch.stack(trans_emb_list, dim=-1)

        # Flatten across variables: [batch, time, hidden_size * num_vars]
        flattened_embedding = transformed_embedding.view(
            batch_size, time_steps, self.hidden_layer_size * num_vars
        )

        # Get variable selection scores from flattened GRN
        if additional_context is not None:
            # Expand context: [batch, 1, hidden_size]
            expanded_static_context = additional_context.unsqueeze(1)
            sparse_weights, static_gate = self.flattened_grn(flattened_embedding, expanded_static_context)
        else:
            sparse_weights = self.flattened_grn(flattened_embedding)
            static_gate = None

        # Normalize weights and add an extra dim for broadcasting: [batch, time, 1, num_vars]
        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)

        # Apply variable-wise weighting: [batch, time, hidden_size, num_vars] * weights
        combined = transformed_embedding * sparse_weights

        # Sum over variables -> [batch, time, hidden_size]
        temporal_ctx = combined.sum(dim=-1)

        return temporal_ctx, sparse_weights, static_gate
