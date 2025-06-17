from dataset.Dataset import Dataset

import pandas as pd
import torch
import numpy as np
class CESNETDataset(Dataset):
    """
    PyTorch Dataset for Temporal Fusion Transformer (TFT) with CESNET-TimeSeries24 dataset.
    Supports multi-IP address processing.
    """

    def __init__(self, file_path, time_steps=96, num_encoder_steps=48):
        """
        :param file_path: Path to the CSV file containing training or testing data
        """
        # Load dataset from CSV file
        self.data = pd.read_csv(file_path)

        # Basic configuration
        self.time_steps = time_steps  # Length of the time window
        self.num_encoder_steps = num_encoder_steps  # Number of encoder steps

        # Ensure that IP ID encoding column exists
        if 'ip_id_encoded' not in self.data.columns and 'ip_id' in self.data.columns:
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder()
            self.data['ip_id_encoded'] = encoder.fit_transform(self.data['ip_id'])

        # Set time index for time-series
        self.time_idx = self.data["id_time"].values

        # Set group IDs using encoded IP ID (used for grouping samples)
        if 'ip_id_encoded' in self.data.columns:
            self.group_ids = self.data[['ip_id_encoded']].values
        else:
            # Fallback: use a dummy group ID
            self.group_ids = np.zeros((len(self.data), 1), dtype=int)

        # Set target variable (prefer n_bytes_log if available)
        target_col = "n_bytes_log" if "n_bytes_log" in self.data.columns else "n_bytes"
        self.target = self.data[[target_col]].values

        # Define feature columns
        self.feature_columns = [
            # Unknown time-varying features
            "n_bytes_log", "n_packets_log", "n_flows_log",
            "tcp_udp_ratio_packets", "tcp_udp_ratio_bytes",
            "dir_ratio_packets", "dir_ratio_bytes",
            "tcp_udp_diff", "dir_ratio_diff",
            "average_n_dest_ip", "average_n_dest_asn", "average_n_dest_ports",
            "std_n_dest_ip", "std_n_dest_asn", "std_n_dest_ports",
            "avg_duration",
            "n_bytes_lag1", "n_bytes_lag6", "n_bytes_lag24",
            "n_bytes_rolling_mean_6", "n_bytes_rolling_std_6", "n_bytes_rolling_max_6",
            "n_packets_lag1", "n_packets_lag6", "n_packets_lag24",
            "n_flows_lag1", "n_flows_lag6", "n_flows_lag24",
            "n_bytes_zscore", "is_anomaly", "is_active",
            # Static features
            "ip_id_encoded","avg_asn_static","avg_duration_static",
            # Known time-varying features
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            # Known categorical features
            "is_weekend", "is_holiday", "day_of_week"
        ]

        # Remove features that are not in the dataset
        self.feature_columns = [col for col in self.feature_columns if col in self.data.columns]

        # Map column names to their indices
        self.column_index_map = {col: idx for idx, col in enumerate(self.feature_columns)}
        print("🔹 Feature column index map:", self.column_index_map)

        # Extract final input features
        self.inputs = self.data[self.feature_columns].values.astype(np.float32)

        # Index of target variable in the feature array
        target_col = "n_bytes_log"
        self.input_obs_loc = [self.column_index_map[target_col]] if target_col in self.column_index_map else [0]

        # Index positions of static features
        static_features = ["ip_id_encoded","avg_asn_static","avg_duration_static"]
        self.static_input_loc = [self.column_index_map[col] for col in static_features
                                 if col in self.column_index_map]

        # Index positions of known continuous time features
        known_features = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
        self.known_regular_inputs = [self.column_index_map[col] for col in known_features
                                     if col in self.column_index_map]

        # Index positions of known categorical features
        categorical_features = ["is_weekend", "is_holiday","day_of_week"]
        self.known_categorical_inputs = [self.column_index_map[col] for col in categorical_features
                                         if col in self.column_index_map]

        # Index positions of unknown time-varying features (excluding known/static/target)
        unknown_features = [col for col in self.feature_columns
                            if col not in static_features + known_features + categorical_features + [target_col]]
        self.unknown_time_features = [self.column_index_map[col] for col in unknown_features]

        # No target scaling by default
        self.target_scaler = None

        # Build a map of IP addresses to their available data indices
        if 'ip_id_encoded' in self.data.columns:
            self.ip_indices = {}
            for ip_id in self.data['ip_id_encoded'].unique():
                indices = self.data.index[self.data['ip_id_encoded'] == ip_id].tolist()
                if len(indices) >= self.time_steps + self.num_encoder_steps:
                    self.ip_indices[ip_id] = indices

            print(f"Available IPs: {len(self.ip_indices)}")

        print(f"Dataset initialized. Number of features: {len(self.feature_columns)}, samples: {len(self.data)}")
        print(f"Target column: {target_col}, index: {self.input_obs_loc}")

        # Check number of samples per IP
        from collections import defaultdict
        ip_lengths = defaultdict(int)
        for gid in self.group_ids:
            ip_lengths[int(gid)] += 1

        print("Number of samples per IP:")
        for ip, count in ip_lengths.items():
            flag = "Yes" if count >= self.time_steps + self.num_encoder_steps else "No"
            print(f"IP {ip}: {count} samples {flag}")

        import matplotlib.pyplot as plt

    def __getitem__(self, index):
        """Get a single data point with multi-IP sampling strategy"""
        # Randomly select an IP and sample from its time-series data
        if hasattr(self, 'ip_indices') and self.ip_indices:
            ip_id = np.random.choice(list(self.ip_indices.keys()))
            ip_index_list = self.ip_indices[ip_id]

            # Ensure enough data length
            if len(ip_index_list) >= self.time_steps + self.num_encoder_steps:
                max_start_idx = len(ip_index_list) - self.time_steps - self.num_encoder_steps
                start_idx_pos = np.random.randint(0, max_start_idx)
                index = ip_index_list[start_idx_pos]
            else:
                # Fallback to default sampling
                max_idx = max(0, len(self.inputs) - self.time_steps - self.num_encoder_steps)
                index = np.random.randint(0, max_idx)
        else:
            # Use basic sampling when no IP grouping
            max_idx = max(0, len(self.inputs) - self.time_steps - self.num_encoder_steps)
            index = np.random.randint(0, max_idx)

        # Ensure index is within valid range
        if index + self.time_steps + self.num_encoder_steps > len(self.inputs):
            index = max(0, len(self.inputs) - self.time_steps - self.num_encoder_steps)

        # Extract input and output sequences
        x = self.inputs[index:index + self.time_steps]
        y = self.target[index + self.time_steps:index + self.time_steps + self.num_encoder_steps]

        # Pad sequence if not long enough
        if len(x) < self.time_steps:
            x = np.pad(x, ((0, self.time_steps - len(x)), (0, 0)), mode='constant')
        if len(y) < self.num_encoder_steps:
            y = np.pad(y, ((0, self.num_encoder_steps - len(y)), (0, 0)), mode='constant')

        # Get group ID
        group_id = self.group_ids[index]

        # Get time indices
        time_indices = self.time_idx[index + self.time_steps:index + self.time_steps + self.num_encoder_steps]
        if len(time_indices) < self.num_encoder_steps:
            last_idx = time_indices[-1] if len(time_indices) > 0 else 0
            time_indices = np.append(time_indices, [last_idx] * (self.num_encoder_steps - len(time_indices)))

        # Return structured dictionary
        return {
            'inputs': torch.tensor(x,   dtype=torch.float32),
            'outputs': torch.tensor(y, dtype=torch.float32),
            'time_idx': torch.tensor(time_indices, dtype=torch.int64).unsqueeze(-1),
            'group_ids': torch.tensor(group_id, dtype=torch.int64),
        }

    def __len__(self):
        """Return the number of samples in the dataset"""
        if hasattr(self, 'ip_indices') and self.ip_indices:
            # Count total number of valid sequences from all IPs
            return sum(max(0, len(indices) - self.time_steps - self.num_encoder_steps)
                       for indices in self.ip_indices.values())
        else:
            # Simple length estimation
            return max(0, len(self.inputs) - self.time_steps - self.num_encoder_steps)
