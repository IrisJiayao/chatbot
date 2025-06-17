import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class Dataset(Dataset):
    """
    PyTorch Dataset class for Temporal Fusion Transformer (TFT)
    """

    def __init__(self, file_path, time_steps=96, num_encoder_steps=48):
        """
        :param file_path: Path to the CSV file containing training or testing data
        """
        # ✅ Load data from CSV
        self.data = pd.read_csv(file_path)

        self.time_steps = time_steps  # Number of input time steps
        self.num_encoder_steps = num_encoder_steps  # Number of forecast steps

        # ✅ Set the target variable (Flow Bytes per Second)
        self.target = self.data[["flow_byts_s_avg"]].values

        # ✅ Convert time column to UNIX timestamp
        self.time_idx = pd.to_datetime(self.data["time_window"], errors='coerce').astype(int) // 10 ** 9

        # ✅ Encode group ID features (e.g., dominant_protocol)
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        self.data['file_source'] = encoder.fit_transform(self.data['file_source'])
        self.data['port_category'] = encoder.fit_transform(self.data['port_category'])
        self.group_ids = self.data[["dominant_protocol"]].values

        # ✅ Define selected feature columns
        self.feature_columns = [
            # Known time-varying continuous features
            "flow_iat_mean_avg", "flow_iat_std_avg", "flow_iat_max_avg", "flow_iat_min_avg",
            "hour_sin", "hour_cos",
            "minute_sin", "minute_cos",
            "second_sin", "second_cos",
            "total_flow_count_ma_5", "total_flow_count_ma_10",
            # Unknown time-varying continuous features
            "flow_byts_s_avg", "flow_pkts_s_avg", "tot_fwd_pkts_sum", "tot_bwd_pkts_avg",
            "pkt_len_min_min", "tcp_ratio", "udp_ratio", "icmp_ratio", "is_attack_window",
            "flow_byts_s_avg_ma_5", "flow_byts_s_avg_ma_10", "flow_pkts_s_avg_ma_5", "flow_pkts_s_avg_ma_10",
            # Static features
            "dominant_protocol", "port_category", "file_source",
            # Known categorical features
            "day_of_week", "is_weekend", "is_business_hours"
        ]

        # ✅ Build column name to index mapping
        self.column_index_map = {col: idx for idx, col in enumerate(self.feature_columns)}
        print("🔹 Original column → index mapping:", self.column_index_map)

        # ✅ Extract all input features
        self.inputs = self.data[self.feature_columns].values.astype(np.float32)

        # ✅ Index of target variable in input
        self.input_obs_loc = [self.column_index_map["flow_byts_s_avg"]]
        print(f"Max value of 'flow_byts_s_avg' in dataset: {self.inputs[:, self.input_obs_loc[0]].max()}")

        # ✅ Index of static features
        self.static_input_loc = [self.column_index_map[col] for col in ["dominant_protocol", "port_category", "file_source"]]

        # ✅ Index of known continuous time-varying features
        self.known_regular_inputs = [
            self.column_index_map[col] for col in [
                "flow_iat_mean_avg", "flow_iat_std_avg", "flow_iat_max_avg", "flow_iat_min_avg",
                "hour_sin", "hour_cos",
                "minute_sin", "minute_cos",
                "second_sin", "second_cos",
                "total_flow_count_ma_5", "total_flow_count_ma_10",
            ] if col in self.column_index_map
        ]

        # ✅ Index of known categorical time-varying features
        self.known_categorical_inputs = [
            self.column_index_map[col] for col in ["day_of_week", "is_weekend", "is_business_hours"] if col in self.column_index_map
        ]

        # ✅ Index of unknown time-varying features
        self.unknown_time_features = [self.column_index_map[col] for col in [
            "flow_byts_s_avg", "flow_pkts_s_avg", "tot_fwd_pkts_sum", "tot_bwd_pkts_avg",
            "pkt_len_min_min", "tcp_ratio", "udp_ratio", "icmp_ratio", "is_attack_window",
            "flow_byts_s_avg_ma_5", "flow_byts_s_avg_ma_10", "flow_pkts_s_avg_ma_5", "flow_pkts_s_avg_ma_10"
        ] if col in self.column_index_map]

        # ✅ Target scaler (disabled)
        self.target_scaler = None

    def __getitem__(self, index):
        """
        Return a single sample from the dataset
        """
        max_attempts = 10  # Maximum number of retries to find a valid sample
        high_value_threshold = np.percentile(self.target, 90)  # Top 10% threshold
        high_value_indices = np.where(self.target >= high_value_threshold)[0]  # Indices with high target values

        for _ in range(max_attempts):
            if np.random.rand() < 0.5 and len(high_value_indices) > 0:
                # 50% chance to select from high target value indices
                index = np.random.choice(high_value_indices)
            else:
                # Otherwise randomly select from full dataset
                index = np.random.randint(0, len(self.inputs) - self.time_steps - self.num_encoder_steps)

            if index + self.time_steps + self.num_encoder_steps <= len(self.target):
                break
        else:
            # If no valid sample found, return zero-filled tensor
            print(f"⚠️ Skipping sample {index}, no valid data")
            return {
                "inputs": torch.zeros((self.time_steps, len(self.feature_columns))),
                "outputs": torch.zeros((self.num_encoder_steps, 1)),
                "group_ids": torch.zeros(1, dtype=torch.int64)
            }

        # Extract feature input sequence
        x = self.inputs[index: index + self.time_steps]
        if len(x) < self.time_steps:
            x = np.pad(x, ((0, self.time_steps - len(x)), (0, 0)), mode='constant')

        # Extract corresponding target sequence
        y = self.target[index + self.time_steps: index + self.time_steps + self.num_encoder_steps]
        y = np.array(y)

        # Ensure at least one meaningful target value
        while np.max(y) < 0.5:
            index = np.random.randint(0, len(self.inputs) - self.time_steps - self.num_encoder_steps)
            y = self.target[index + self.time_steps: index + self.time_steps + self.num_encoder_steps]

        # If target is completely empty, skip the sample
        if len(y) == 0:
            print(f"Warning: Index {index} has empty y, replacing with zeros.")
            return None

        # Pad target if shorter than required
        if len(y) < self.num_encoder_steps:
            pad_width = self.num_encoder_steps - len(y)
            y = np.pad(y, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)

        # Debugging assertions
        assert not np.any(np.isnan(x)), f"x contains NaN at index {index}"
        assert not np.any(np.isnan(y)), f"y contains NaN at index {index}"
        assert x.shape[0] == self.time_steps, f"Error: x.shape={x.shape}, expected (96, N), index={index}"

        y_max = np.max(y)

        return {
            'inputs': torch.tensor(x, dtype=torch.float32),
            'outputs': torch.tensor(y, dtype=torch.float32),
            'time_idx': torch.tensor(
                np.array(self.time_idx[index + self.time_steps: index + self.time_steps + self.num_encoder_steps]),
                dtype=torch.int64
            ).unsqueeze(-1),  # Ensure shape is [num_encoder_steps, 1]
            'group_ids': torch.tensor(self.group_ids[index], dtype=torch.int64),
        }

    def __len__(self):
        """
        Return the total number of rows (samples) in the dataset
        """
        return self.inputs.shape[0]
