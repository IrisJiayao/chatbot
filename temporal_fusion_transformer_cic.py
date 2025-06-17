import math
import torch
import ipdb
import json
from torch import nn
from base_model import BaseModel
from add_and_norm import AddAndNorm
from gated_residual_network import GRN
from gated_linear_unit import GLU
from time_distributed_linear import LinearLayer
from variable_selection_network import VariableSelectionNetwork
from time_distributed_linear import TimeDistributed
from interpretable_multi_head_attention import InterpretableMultiHeadAttention

class TemporalFusionTransformer(BaseModel):
    def __init__(self,raw_params):
        super(TemporalFusionTransformer, self).__init__()

        params = dict(raw_params)  # copy locally
        print(params)

        self.time_steps = int(params['total_time_steps'])  # 总时间步
        self.input_size = int(params['input_size'])  # 输入维度
        self.output_size = int(params['output_size'])  # 输出维度
        self.category_counts = json.loads(str(params['category_counts']))  # 类别变量的类别数
        self.num_heads = int(params['num_heads'])  # 注意力头数
        # self.offset = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.scale = nn.Parameter(torch.tensor([1.5]), requires_grad=False)
        self.offset = nn.Parameter(torch.tensor([0.2]), requires_grad=False)
        print("Parsed Parameters:", json.dumps(params, indent=2))

        # Relevant indices for TFT
        self._input_obs_loc = json.loads(str(params['input_obs_loc']))
        #self._static_input_loc = json.loads(str(params['static_input_loc']))
        self._static_input_loc = params['static_input_loc']

        self._known_regular_input_idx = json.loads(
            str(params['known_regular_inputs']))
        self._known_categorical_input_idx = json.loads(
            str(params['known_categorical_inputs']))
        self._unknown_time_features_idx = json.loads(
            str(params['unknown_time_features']))
        # Network params
        self.quantiles = list(params['quantiles'])
        self.device = str(params['device'])
        self.hidden_layer_size = int(params['hidden_layer_size'])
        self.dropout_rate = float(params['dropout_rate'])
        self.max_gradient_norm = float(params['max_gradient_norm'])
        self.learning_rate = float(params['lr'])
        self.minibatch_size = int(params['batch_size'])
        self.num_epochs = int(params['epochs'])
        self.early_stopping_patience = int(params['early_stopping_patience'])

        self.num_encoder_steps = int(params['num_encoder_steps'])
        self.num_stacks = int(params['stack_size'])
        self.num_heads = int(params['num_heads'])
        self.batch_first = True
        self.num_static = len(self._static_input_loc)  # 静态变量个数


        self.num_inputs = len(self._known_regular_input_idx) + len(self._known_categorical_input_idx) +  len(self._unknown_time_features_idx)

        self.num_inputs_decoder = len(self._known_regular_input_idx) + len(self._known_categorical_input_idx)

        time_steps = self.time_steps
        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables

        self.static_input_layer = nn.Linear(1, self.hidden_layer_size)
        # print(f"self.static_input_layer weight shape: {self.static_input_layer.weight.shape}")

        self.time_varying_embedding_layer = LinearLayer(input_size=1, size=self.hidden_layer_size,
                                                        use_time_distributed=True, batch_first=self.batch_first)

        self._input_placeholder = None
        self._attention_components = None
        self._prediction_parts = None

        # for i, emb in enumerate(self.embeddings):
        #     print(f"Embedding[{i}] num_embeddings: {emb.num_embeddings}, category_counts[i]: {self.category_counts[i]}")

        self.vsn=VariableSelectionNetwork(
            num_static=self.num_static,
            historical_num_vars=self.num_inputs,
            future_num_vars = self.num_inputs_decoder,
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            ts_embedding_dim=self.hidden_layer_size,
            batch_first=self.batch_first
        )

        self.lstm_encoder = nn.LSTM(input_size=self.hidden_layer_size, hidden_size=self.hidden_layer_size,
                                    batch_first=self.batch_first)
        self.lstm_decoder = nn.LSTM(input_size=self.hidden_layer_size, hidden_size=self.hidden_layer_size,
                                    batch_first=self.batch_first)

        self.lstm_glu = GLU(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            batch_first=self.batch_first)
        self.lstm_glu_add_and_norm = AddAndNorm(hidden_layer_size=self.hidden_layer_size)

        self.static_enrichment_grn = GRN(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            output_size=None,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            return_gate=True,
            batch_first=self.batch_first)

        self.self_attn_layer = InterpretableMultiHeadAttention(self.num_heads, self.hidden_layer_size,
                                                               dropout_rate=self.dropout_rate)

        self.self_attention_glu = GLU(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            batch_first=self.batch_first)
        self.self_attention_glu_add_and_norm = AddAndNorm(hidden_layer_size=self.hidden_layer_size)

        self.decoder_grn = GRN(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            output_size=None,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            return_gate=False,
            batch_first=self.batch_first)

        self.final_glu = GLU(
            input_size=self.hidden_layer_size,
            hidden_layer_size=self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            batch_first=self.batch_first)
        self.final_glu_add_and_norm = AddAndNorm(hidden_layer_size=self.hidden_layer_size)

        self.output_layer =LinearLayer(
            input_size=self.hidden_layer_size,
            size=self.output_size * len(self.quantiles),
            use_time_distributed=True,
            batch_first=self.batch_first)
        self.debug_mode = True
        self.column_index_map = {
            'day_of_week': 16,
            'is_holiday': 17,
            'protocol': 15,
        }

        def create_dynamic_embeddings(self, inputs=None):
            """
            Dynamically create embeddings with robust configuration

            Args:
                inputs (torch.Tensor, optional): Input tensor to derive actual category counts

            Returns:
                nn.ModuleList of embedding layers
            """
            base_category_counts = self.category_counts.copy()

            embeddings = nn.ModuleList()
            for i, category_count in enumerate(base_category_counts):
                if inputs is not None:
                    unique_values = torch.unique(inputs[:, :, -len(base_category_counts) + i])
                    num_embeddings = max(2, len(unique_values))
                else:
                    num_embeddings = max(2, category_count)

                embedding = nn.Embedding(
                    num_embeddings=num_embeddings,
                    embedding_dim=self.hidden_layer_size
                )
                embeddings.append(embedding)

            return embeddings

        self.embeddings = create_dynamic_embeddings(self, None)
        # self.diagnose_features()



    def get_decoder_mask(self, self_attn_inputs):
        """Returns causal mask to apply for self-attention layer.

        Args:
          self_attn_inputs: Inputs to self-attention layer to determine mask shape
        # """
        try:
            # 打印调试信息
            # print(f"self_attn_inputs 形状: {self_attn_inputs.shape}")

            len_s = self_attn_inputs.shape[1]  # 时间步长度
            bs = self_attn_inputs.shape[0]  # 批次大小

            # 创建因果掩码
            mask = torch.triu(torch.ones(len_s, len_s), diagonal=1).to("cuda")
            mask = mask.unsqueeze(0).expand(bs, -1, -1).to("cuda")

            return mask
        except Exception as e:
            print(f"掩码生成错误: {e}")

        return mask

    def diagnose_categorical_inputs(self, inputs):
        """
        Comprehensive diagnostic method for categorical inputs

        Args:
            inputs (torch.Tensor): Input tensor to be analyzed
        """
        # Calculate categorical variable information
        num_regular_variables = self.input_size - len(self.category_counts)

        # Separate regular and categorical inputs
        regular_inputs = inputs[:, :, :num_regular_variables]
        categorical_inputs = inputs[:, :, num_regular_variables:]

        # Detailed analysis of each categorical column
        for i in range(categorical_inputs.shape[2]):
            column_data = categorical_inputs[:, :, i]
            unique_values = torch.unique(column_data)

    def verify_known_categorical_inputs(self):
        """
        Detailed verification of known categorical input indices and mappings
        """
        # print("\n🕵️ Known Categorical Inputs Verification 🕵️")
        #
        # # Print column index mapping
        # print("Column Index Mapping:")
        # for column_name, index in self.column_index_map.items():
        #     print(f"  {column_name}: {index}")
        #
        # # Print details of known categorical inputs
        # print("\nKnown Categorical Inputs:")
        # for input_col in self._known_categorical_input_idx:
        #     print(f"  {input_col}")

    def comprehensive_input_diagnosis(self, inputs):
        """
        Comprehensive diagnosis of input data across multiple dimensions

        Args:
            inputs (torch.Tensor): Input tensor
        """
        # print("\n📋 Comprehensive Input Data Diagnosis Report 📋")
        #
        # # Basic information
        # print(f"Input tensor shape: {inputs.shape}")
        # print(f"Total feature count: {inputs.shape[2]}")

        # Variable type statistics
        num_regular_variables = self.input_size - len(self.category_counts)
        num_categorical_variables = len(self.category_counts)

        # print("\nVariable Type Statistics:")
        # print(f"Number of regular variables: {num_regular_variables}")
        # print(f"Number of categorical variables: {num_categorical_variables}")

        # Detailed input column information
        # print("\nInput Column Details:")
        # for i in range(inputs.shape[2]):
        #     column_data = inputs[:, :, i]
        #     print(f"\nColumn {i}:")
        #     print(f"  Data type: {column_data.dtype}")
        #     print(f"  Minimum value: {column_data.min()}")
        #     print(f"  Maximum value: {column_data.max()}")
        #     print(f"  Mean value: {column_data.float().mean()}")

    def get_tft_embeddings(self, all_inputs):
        if self.debug_mode:
            self.diagnose_categorical_inputs(all_inputs)
            self.verify_known_categorical_inputs()

        time_steps = self.time_steps
        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables
        embedding_sizes = [
            self.hidden_layer_size for i, size in enumerate(self.category_counts)
        ]

        regular_inputs, categorical_inputs \
            = all_inputs[:, :, :num_regular_variables], \
            all_inputs[:, :, num_regular_variables:]
        # 关键修改：将分类输入转换为长整型
        categorical_inputs = categorical_inputs.long()
        categorical_inputs = categorical_inputs.to(self.device)
        regular_inputs = regular_inputs.to(self.device)

        if self.debug_mode:
            self.comprehensive_input_diagnosis(all_inputs)


        try:
            embedded_inputs = [
                self.embeddings[i](categorical_inputs[:, :, i].long())
                for i in range(len(self.embeddings))
            ]
        except IndexError as e:
            print(f"\n embed creation error: {e}")
            print(f" {categorical_inputs.shape}")
            print(f" Number of embeddings: {len(self.embeddings)}")
            print(f" Category number: {self.category_counts}")

            for i in range(categorical_inputs.shape[2]):
                try:
                    unique_values = torch.unique(categorical_inputs[:, :, i])
                    print(f" column {i}:")
                    print(f" unique value: {unique_values}")
                    print(f" Max: {unique_values.max()}")
                    self.embeddings[i](categorical_inputs[:, :, i].long())
                except Exception as col_error:
                    print(f" Column {i} error: {col_error}")

            raise

        try:

            import numpy as np
            if self._static_input_loc:
                static_inputs = []
                for i in self._static_input_loc:
                    # Gets the I-th feature of the first time step of the entire batch
                    static_column = all_inputs[:, 0, i].cpu().numpy()

                    if np.all(np.abs(static_column) < 1e-6):  # Use thresholds to check for values that are close to zero
                        print(f"Warning: The static input column {i} approaches zero, and the default all-zero tensor will be used")
                        # Make sure all tensors have the same shape: [batch_size, hidden_layer_size]
                        reg_i = torch.zeros((all_inputs.shape[0], self.hidden_layer_size), device=self.device)
                        static_inputs.append(reg_i)
                    else:
                        # Processing through the static input layer
                        input_tensor = all_inputs[:, 0, i:i + 1]
                        input_tensor = input_tensor.float().to(self.device)
                        reg_i = self.static_input_layer(input_tensor)

                        # Key fix: Make sure all tensors have the same shape - drop to 2D if reg_i is 3D
                        if len(reg_i.shape) == 3:
                            reg_i = reg_i.squeeze(1)

                        static_inputs.append(reg_i)

                # If you do end up with a static input, stack
                if static_inputs:
                    # Make sure all static input tensors have the same shape before stacking
                    static_inputs = torch.stack(static_inputs, dim=1)
                    # print(f"堆叠后static_inputs形状: {static_inputs.shape}")
                else:
                    static_inputs = torch.zeros((all_inputs.shape[0], 1, self.hidden_layer_size), device=self.device)
                    print(f"Use the default static_inputs, shape: {static_inputs.shape}")
            else:
                print("Warning: There is no static input position")
                static_inputs = torch.zeros((all_inputs.shape[0], 1, self.hidden_layer_size), device=self.device)


            obs_inputs = torch.stack([
                self.time_varying_embedding_layer(regular_inputs[Ellipsis, i:i + 1].float())
                for i in self._input_obs_loc
            ], dim=-1)

            wired_embeddings = []
            for i in range(categorical_inputs.shape[2]):
                actual_index = i + num_regular_variables

                if actual_index not in self._known_categorical_input_idx and actual_index not in self._input_obs_loc:
                    e = self.embeddings[i](categorical_inputs[:, :, i])
                    wired_embeddings.append(e)
            unknown_inputs = []
            for i in range(regular_inputs.shape[-1]):
                if i not in self._known_regular_input_idx and i not in self._input_obs_loc and i not in self._static_input_loc:
                    e = self.time_varying_embedding_layer(regular_inputs[Ellipsis, i:i + 1])
                    unknown_inputs.append(e)


            if unknown_inputs + wired_embeddings:
                unknown_inputs = torch.stack(unknown_inputs + wired_embeddings, dim=-1)
            else:
                unknown_inputs = None

            # A priori known inputs
            known_regular_inputs = []

            for i in self._known_regular_input_idx:
                if i not in self._static_input_loc:
                    # Adding an index range check
                    if i >= regular_inputs.shape[-1]:
                        print(f"Warning: Index {i} is out of feature range {regular_inputs.shape[-1]}")
                        continue

                    # 检查张量形状和值
                    feature_slice = regular_inputs[..., i:i + 1]
                    # print(f"特征 {i} 形状: {feature_slice.shape}, 非空元素数: {torch.count_nonzero(feature_slice)}")

                    if feature_slice.shape[-1] == 0 or feature_slice.numel() == 0:
                        print(f"Warning: Feature {i} is an empty tensor, skipped")
                        continue

                    known_regular_inputs.append(
                        self.time_varying_embedding_layer(feature_slice.float()))

            known_categorical_inputs = []

            index_mapping = {original_idx: new_idx for new_idx, original_idx in enumerate(self._known_categorical_input_idx)}

            for original_idx in self._known_categorical_input_idx:
                new_idx = index_mapping[original_idx]  # 映射到 0, 1
                known_categorical_inputs.append(embedded_inputs[new_idx])

            known_combined_layer = torch.stack(known_regular_inputs + known_categorical_inputs, dim=-1)

            return unknown_inputs, known_combined_layer, obs_inputs, static_inputs
        except Exception as e:
            print(f"Embed handling global error: {e}")
            raise




    def forward(self, x):  # type: (torch.Tensor) -> torch.Tensor
        time_steps = self.time_steps
        encoder_steps = self.num_encoder_steps
        # self.input_size = self.input_size
        all_inputs = x.to(self.device)
        # print(f"all_inputs.shape: {all_inputs.shape}")
        unknown_inputs, known_combined_layer, obs_inputs, static_inputs = self.get_tft_embeddings(all_inputs)
        # print(f"static_inputs: {static_inputs}")  # 确保 static_inputs 不是 None
        if static_inputs is None:
            raise ValueError("static_inputs is None! Check why it is not initialized.")
        if unknown_inputs is not None:
            historical_inputs = torch.cat([
                unknown_inputs[:, :encoder_steps, :],
                known_combined_layer[:, :encoder_steps, :],
                obs_inputs[:, :encoder_steps, :]
            ], dim=-1)

        else:
            historical_inputs = torch.cat([
                  known_combined_layer[:, :encoder_steps, :],
                  obs_inputs[:, :encoder_steps, :]
              ], dim=-1)


        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, encoder_steps:, :]

        self.num_inputs = len(self._known_regular_input_idx) + self.output_size + obs_inputs.shape[-1]

        (
            static_encoder,
            historical_features,
            future_features,
            static_weights,
            historical_flags,
            future_flags,
            static_context_state_h,
            static_context_state_c,
            static_context_enrichment
        ) = self.vsn(static_inputs, historical_inputs, future_inputs)

        history_lstm, (state_h, state_c) = self.lstm_encoder(historical_features, (static_context_state_h.unsqueeze(0),
                                                        static_context_state_c.unsqueeze(0)))
        future_lstm, _ = self.lstm_decoder(future_features, (state_h, state_c))
        lstm_layer = torch.cat([history_lstm, future_lstm], dim=1)

        # Apply gated skip connection
        input_embeddings = torch.cat([historical_features, future_features], dim=1)

        lstm_layer, _ = self.lstm_glu(lstm_layer)
        temporal_feature_layer = self.lstm_glu_add_and_norm(lstm_layer, input_embeddings)

        # Static enrichment layers
        expanded_static_context = static_context_enrichment.unsqueeze(1)
        enriched, _ = self.static_enrichment_grn(temporal_feature_layer, expanded_static_context)

        # Decoder self attention
        mask = self.get_decoder_mask(enriched)
        x, self_att = self.self_attn_layer(enriched, enriched, enriched, mask)#, attn_mask=mask.repeat(self.num_heads, 1, 1))

        x, _ = self.self_attention_glu(x)
        x = self.self_attention_glu_add_and_norm(x, enriched)

        # Nonlinear processing on outputs
        decoder = self.decoder_grn(x)
        # Final skip connection
        decoder, _ = self.final_glu(decoder)
        transformer_layer = self.final_glu_add_and_norm(decoder, temporal_feature_layer)
        # Attention components for explainability
        attention_components = {
            # Temporal attention weights
            'decoder_self_attn': self_att,
            # Static variable selection weights
            'static_flags': static_weights[Ellipsis, 0],
            # Variable selection weights of past inputs
            'historical_flags': historical_flags[Ellipsis, 0, :],
            # Variable selection weights of future inputs
            'future_flags': future_flags[Ellipsis, 0, :]
        }

        outputs = self.output_layer(transformer_layer[:, self.num_encoder_steps:, :])

        # print(f"[DEBUG] outputs min: {outputs.min().item():.3f}, max: {outputs.max().item():.3f}")
        return outputs, all_inputs, attention_components







