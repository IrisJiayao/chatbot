import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from test_dataset import flow_byts_values


class Dataset(Dataset):
    """
    Pseudo-TFT compatible Dataset for single-variable LSTM prediction.
    """

    def __init__(self, file_path, time_steps=96, num_encoder_steps=48):
        self.data = pd.read_csv(file_path)

        self.time_steps = time_steps
        self.num_encoder_steps = num_encoder_steps

        # ✅ Use only target-related columns (for univariate prediction)
        # Add delta and moving average features
        # self.data["flow_byts_s_diff"] = self.data["flow_byts_s"].diff().fillna(0)
        self.data["flow_pkts_s_diff"] = self.data["flow_pkts_s"].diff().fillna(0)
        self.data["flow_byts_s_ma_5"] = self.data["flow_byts_s"].rolling(window=5, min_periods=1).mean()
        self.data["flow_byts_s_ma_10"] = self.data["flow_byts_s"].rolling(window=10, min_periods=1).mean()

        # Feature selection
        self.feature_columns = [
            'flow_byts_s', 'flow_pkts_s', 'tot_fwd_pkts', 'tot_bwd_pkts', 'pkt_len_min',
            'flow_byts_s_diff', 'flow_pkts_s_diff',
            'flow_byts_s_ma_5', 'flow_byts_s_ma_10',
            'flow_byts_s_diff_ma_5', 'flow_byts_s_ma_10'
        ]
        self.column_index_map = {col: idx for idx, col in enumerate(self.feature_columns)}

        # Input and target extraction
        self.inputs = self.data[self.feature_columns].values.astype(np.float32)
        self.target = self.data[["flow_byts_s"]].values.astype(np.float32)

        # Dummy group ID for single group setup
        self.group_ids = np.zeros((len(self.data), 1), dtype=int)

        # TFT-compatible index definitions
        self.input_obs_loc = [0]  # Only target variable
        self.static_input_loc = []  # No static inputs
        self.known_regular_inputs = []  # No time-related known inputs
        self.known_categorical_inputs = []
        self.unknown_time_features = [0]  # Only the target variable

        self.time_idx = np.arange(len(self.data))  # Use row index as time index
        self.target_scaler = None

        # Save original series for inverse transformation (for difference restoration)
        self.flow_byts_s_orig = self.data["flow_byts_s"].values.copy()

    def __getitem__(self, index):
        max_index = len(self.inputs) - self.time_steps - self.num_encoder_steps
        if index > max_index:
            index = np.random.randint(0, max_index)

        # Extract encoder and decoder ranges
        encoder_start = index
        encoder_end = encoder_start + self.time_steps
        decoder_end = encoder_end + self.num_encoder_steps

        # Input features
        x = self.inputs[encoder_start:encoder_end]

        # Ground truth values
        y = self.target[encoder_end:decoder_end]

        # Pad y if it is shorter than required
        if len(y) < self.num_encoder_steps:
            y = np.pad(y, ((0, self.num_encoder_steps - len(y)), (0, 0)), mode='constant')

        # Retrieve the last value of input sequence for inverse transformation (i.e., for differencing recovery)
        input_start_value = self.flow_byts_s_orig[encoder_end - 1]

        # Reshape for pseudo-TFT input: [T, 1, 1]
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)
        input_start_value = torch.tensor([input_start_value], dtype=torch.float32)

        return {
            'inputs': x,
            'outputs': y,
            'time_idx': torch.tensor(
                np.array(self.time_idx[index + self.time_steps: index + self.time_steps + self.num_encoder_steps]),
                dtype=torch.int64
            ).unsqueeze(-1),
            'group_ids': torch.tensor(self.group_ids[index], dtype=torch.int64),
            "input_start_value": input_start_value  # 🔥 Starting value for inverse differencing
        }

    def __len__(self):
        return len(self.inputs)
