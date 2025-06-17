import torch
from torch import nn

from lstm_combine import LSTMCombine   # 这是你按方案C改好的 LSTMCombine
from static_combine import StaticCombine
from gated_residual_network import GRN

class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        num_static,           # 静态变量个数
        historical_num_vars,
        future_num_vars,             # 动态变量个数（比如 6）
        hidden_layer_size,    # 隐层大小 (如 16)
        dropout_rate,
        ts_embedding_dim=8,   # 时间序列的 embedding 维度 (可自行修改)
        batch_first=True
    ):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.num_static = num_static
        self.historical_num_vars = historical_num_vars        # 动态变量个数
        self.future_num_vars = future_num_vars
        self.ts_embedding_dim = ts_embedding_dim
        self.batch_first = batch_first
        self.dropout_rate = dropout_rate

        # 1️⃣ 静态变量选择 (Static Variable Selection)
        #    假设输入形状: [batch_size, num_static, feature_dim=hidden_layer_size]
        #    输出: [batch_size, hidden_layer_size], 以及 [batch_size, num_static, 1] 的权重
        self.static_combine_and_mask = StaticCombine(
            input_size=hidden_layer_size,
            num_static=num_static,
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            batch_first=batch_first
        )
        self.static_context_variable_selection_grn = GRN(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            output_size=None,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            return_gate=False,
            batch_first=self.batch_first)

        self.static_context_enrichment_grn = GRN(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                return_gate=False,
                batch_first=self.batch_first)

        self.static_context_state_h_grn = GRN(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            output_size=None,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            return_gate=False,
            batch_first=self.batch_first)

        self.static_context_state_c_grn = GRN(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            output_size=None,
            dropout_rate=self.dropout_rate,
            use_time_distributed=False,
            return_gate=False,
            batch_first=self.batch_first)

        # 2️⃣ 给历史和未来的动态输入做一个线性投影，把 [batch, time, num_vars] -> [batch, time, ts_embedding_dim*num_vars]
        #    然后 reshape 成 [batch, time, ts_embedding_dim, num_vars]
        #    这样才能与 LSTMCombine(embedding_dim=ts_embedding_dim, num_vars=...) 对齐

        # print(f"num_vars: {num_vars}, ts_embedding_dim: {ts_embedding_dim}")

        # self.historical_projection = nn.Linear(historical_num_vars, ts_embedding_dim * historical_num_vars)
        # self.future_projection = nn.Linear(future_num_vars, ts_embedding_dim * future_num_vars)

        # 3️⃣ LSTMCombine (Variable Selection for Historical)
        #    输入形状: [batch, time, ts_embedding_dim, num_vars]
        #    输出形状: [batch, time, hidden_layer_size]
        self.historical_lstm_combine_and_mask = LSTMCombine(
            embedding_dim=ts_embedding_dim,
            num_vars=historical_num_vars,
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=False,
            batch_first=batch_first
        )

        # 4️⃣ LSTMCombine (Variable Selection for Future)
        self.future_lstm_combine_and_mask = LSTMCombine(
            embedding_dim=ts_embedding_dim,
            num_vars=future_num_vars,
            hidden_layer_size=hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=False,
            batch_first=batch_first
        )

    def forward(self, static_inputs, historical_inputs, future_inputs):
        """
        参数:
          static_inputs:    (batch_size, num_static, feature_dim=hidden_layer_size)
          historical_inputs:(batch_size, encoder_steps, num_vars)
          future_inputs:    (batch_size, decoder_steps, num_vars)

        返回:
          static_encoder:      [batch_size, hidden_layer_size]
          historical_features: [batch_size, encoder_steps, hidden_layer_size]
          future_features:     [batch_size, decoder_steps, hidden_layer_size]
          static_weights:      [batch_size, num_static, 1]
          historical_flags:    [batch_size, encoder_steps, 1, num_vars]
          future_flags:        [batch_size, decoder_steps, 1, num_vars]
        """

        batch_size, encoder_steps,_, _ = historical_inputs.shape
        _, decoder_steps,_, _ = future_inputs.shape

        # 1. Do the static variable selection first
        #  output: static_encoder -> [batch, hidden_layer_size]
        static_encoder, static_weights = self.static_combine_and_mask(static_inputs)
        static_context_variable_selection = self.static_context_variable_selection_grn(static_encoder)
        static_context_enrichment = self.static_context_enrichment_grn(static_encoder)
        static_context_state_h = self.static_context_state_h_grn(static_encoder)
        static_context_state_c = self.static_context_state_c_grn(static_encoder)

        # 2. Make history variable selection by LSTMCombine
        historical_features, historical_flags, _ = self.historical_lstm_combine_and_mask(
            historical_inputs,static_context_variable_selection
        )

        # 3. Do the same for future inputs
        future_features, future_flags, _ = self.future_lstm_combine_and_mask(

            future_inputs, static_context_variable_selection
        )

        return (static_encoder, historical_features, future_features, static_weights,
                historical_flags, future_flags,static_context_state_h,
                static_context_state_c, static_context_enrichment)


# # =============== 测试代码 ===============
# if __name__ == "__main__":
#     batch_size = 5
#     num_static = 4
#     num_vars = 6        # 动态变量个数
#     hidden_size = 16
#     encoder_steps = 10
#     decoder_steps = 5
#     dropout_rate = 0.1
#     ts_embedding_dim = 8  # 时间序列 embedding 维度
#
#     # 初始化 VSN
#     vsn = VariableSelectionNetwork(
#         num_static=num_static,
#         num_vars=num_vars,
#         hidden_layer_size=hidden_size,
#         dropout_rate=dropout_rate,
#         ts_embedding_dim=ts_embedding_dim,
#         batch_first=True
#     )
#
#     # 生成静态变量输入 [batch, num_static, hidden_size]
#     static_inputs = torch.rand(batch_size, num_static, hidden_size)
#
#     # 生成历史/未来时间序列输入 [batch, time, num_vars]
#     historical_inputs = torch.rand(batch_size, encoder_steps, num_vars)
#     future_inputs = torch.rand(batch_size, decoder_steps, num_vars)
#
#     # 前向传播
#     (
#         static_encoder,
#         historical_features,
#         future_features,
#         static_weights,
#         historical_flags,
#         future_flags,
#         static_context_state_h,
#         static_context_state_c,
#         static_context_enrichment
#     ) = vsn(static_inputs, historical_inputs, future_inputs)
#
#     # 打印形状
#     print("静态变量选择后：", static_encoder.shape)       # [5, 16]
#     print("历史变量选择后：", historical_features.shape)  # [5, 10, 16]
#     print("未来变量选择后：", future_features.shape)      # [5,  5, 16]
#     print("静态变量权重：   ", static_weights.shape)      # [5, 4, 1]
#     print("历史变量权重：   ", historical_flags.shape)    # [5, 10, 1, 6]
#     print("未来变量权重：   ", future_flags.shape)        # [5,  5, 1, 6]
