import joblib
import pandas as pd
import numpy as np
import tarfile
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
from abc import ABC, abstractmethod


class GenericDataFormatter:
    """
    Abstract base class that provides common data processing methods
    """

    def __init__(self):
        # Use MinMaxScaler (can be swapped for RobustScaler if needed for outliers)
        self.scaler = MinMaxScaler()
        self.label_encoders = {}

    def normalize(self, df, columns):
        """
        Apply normalization to specified numerical columns using the defined scaler

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list
            List of numerical column names to normalize

        Returns:
        --------
        pd.DataFrame
            Dataframe with normalized columns
        """
        df[columns] = self.scaler.fit_transform(df[columns])
        return df

    def handle_missing_values(self, df):
        """
        Handle missing and infinite values by replacing them with median values

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe

        Returns:
        --------
        pd.DataFrame
            Cleaned dataframe
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df

    def create_time_features(self, df, timestamp_col):
        """
        Create time-related features from a timestamp column

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        timestamp_col : str
            Name of the timestamp column

        Returns:
        --------
        pd.DataFrame
            Dataframe with added time features
        """
        df = df.copy()

        # Ensure timestamp column is in datetime format
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')

        # Fill missing timestamps with a default value
        mask = df[timestamp_col].isna()
        if mask.any():
            print(f"Filling {mask.sum()} missing timestamps")
            df.loc[mask, timestamp_col] = pd.Timestamp('2023-01-01')

        # Extract hour and weekday
        df["hour"] = df[timestamp_col].dt.hour
        df["day_of_week"] = df[timestamp_col].dt.dayofweek

        # Periodic (cyclical) encoding for hour and weekday
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Binary weekend indicator
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        return df

    def split_data(self, df, group_col='ip_id', time_col='id_time', train_ratio=0.7, val_ratio=0.15):
        """
        Split the dataset into train/validation/test sets by group, preserving time order.

        Parameters:
        -----------
        df : pd.DataFrame
            The full input dataframe.
        group_col : str
            Column name used to group data (e.g., IP address).
        time_col : str
            Timestamp column name used for chronological sorting.
        train_ratio : float
            Proportion of data to use for training.
        val_ratio : float
            Proportion of data to use for validation.

        Returns:
        --------
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame
        """
        train_list, val_list, test_list = [], [], []

        for group, group_df in df.groupby(group_col):
            group_df_sorted = group_df.sort_values(by=time_col)
            n = len(group_df_sorted)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train = group_df_sorted.iloc[:n_train]
            val = group_df_sorted.iloc[n_train:n_train + n_val]
            test = group_df_sorted.iloc[n_train + n_val:]

            train_list.append(train)
            val_list.append(val)
            test_list.append(test)

        train_df = pd.concat(train_list).reset_index(drop=True)
        val_df = pd.concat(val_list).reset_index(drop=True)
        test_df = pd.concat(test_list).reset_index(drop=True)

        return train_df, val_df, test_df

    def save_splits(self, train_df, val_df, test_df, output_dir):
        """
        Save train/val/test splits to CSV files.

        Parameters:
        -----------
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame
        output_dir : str
            Directory path where CSVs will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        print(f"Processed data saved to: {output_dir}")


import joblib
import pandas as pd
import numpy as np
import tarfile
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
from abc import ABC, abstractmethod


class CESNETDataFormatter(GenericDataFormatter):
    """
    Custom data formatter for the CESNET-TimeSeries24 dataset, supporting multi-IP processing.
    """

    def __init__(self):
        super().__init__()
        self.weekends_holidays_df = None

    def load_data(self, sample_path, times_path, weekends_path, num_ips=5):
        """
        Load CESNET-TimeSeries24 sample dataset and handle multiple IP addresses.

        Parameters:
        -----------
        sample_path : str
            Path to the compressed sample data archive.
        times_path : str
            Path to the time mapping archive.
        weekends_path : str
            Path to the weekend and holiday CSV file.
        num_ips : int, optional (default=5)
            Number of IPs to process (limits total samples).
        """
        print("Extracting sample data...")
        with tarfile.open(sample_path, 'r:gz') as tar:
            tar.extractall(path="./temp_cesnet_data")

        print("Extracting time mapping...")
        with tarfile.open(times_path, 'r:gz') as tar:
            tar.extractall(path="./temp_cesnet_data")

        # Load weekend and holiday metadata
        self.weekends_holidays_df = pd.read_csv(weekends_path)

        # Load time ID-to-timestamp mapping
        time_mapping = pd.read_csv("./temp_cesnet_data/times/times_1_hour.csv")

        # Get list of IP sample files
        sample_files = os.listdir("./temp_cesnet_data/ip_addresses_sample/agg_1_hour")
        if not sample_files:
            raise ValueError("No sample files found.")

        sample_files = sample_files[:num_ips]  # Limit number of IPs

        all_ip_dfs = []

        for i, sample_file in enumerate(sample_files):
            print(f"Processing IP {i + 1}/{len(sample_files)}: {sample_file}")
            ip_df = pd.read_csv(f"./temp_cesnet_data/ip_addresses_sample/agg_1_hour/{sample_file}")

            # Add IP identifier column
            ip_id = sample_file.replace('.csv', '')
            ip_df['ip_id'] = ip_id

            # Merge time mapping to convert id_time → actual timestamp
            ip_df = ip_df.merge(time_mapping, on='id_time')

            all_ip_dfs.append(ip_df)
            print(f"  - Loaded {len(ip_df)} rows.")

        combined_df = pd.concat(all_ip_dfs, ignore_index=True)
        print(f"Final combined dataset has {len(combined_df)} rows from {len(sample_files)} IPs.")
        return combined_df

    def add_holiday_features(self, df, time_col):
        """
        Add holiday and holiday type labels to the dataframe.

        Parameters:
        -----------
        df : pd.DataFrame
        time_col : str
            Column containing timestamp

        Returns:
        --------
        pd.DataFrame
        """
        if self.weekends_holidays_df is not None:
            try:
                self.weekends_holidays_df['Date'] = pd.to_datetime(self.weekends_holidays_df['Date'])
                df['date'] = pd.to_datetime(df[time_col]).dt.date
                self.weekends_holidays_df['date'] = self.weekends_holidays_df['Date'].dt.date

                holiday_dict = self.weekends_holidays_df.set_index('date')['Type'].to_dict()

                df['is_holiday'] = df['date'].map(lambda x: 1 if x in holiday_dict else 0)
                df['holiday_type'] = df['date'].map(lambda x: holiday_dict.get(x, 'Regular'))

                df.drop('date', axis=1, inplace=True)
            except Exception as e:
                print(f"Error while processing holidays: {e}")
                df['is_holiday'] = 0
                df['holiday_type'] = 'Regular'
        else:
            df['is_holiday'] = 0
            df['holiday_type'] = 'Regular'

        return df

    def transform_features(self, df):
        """
        Apply log transform and normalization to numeric features.

        Parameters:
        -----------
        df : pd.DataFrame

        Returns:
        --------
        pd.DataFrame
        """
        df = df.copy()

        # 1. Log transform for large-scale traffic features
        for col in ['n_bytes', 'n_packets', 'n_flows']:
            if col in df.columns:
                print(f"{col} raw range: {df[col].min()} to {df[col].max()}")
                df[col] = df[col].replace(0, 1e-6)
                df[col + '_log'] = np.log1p(df[col])
                print(f"{col}_log range after transform: {df[col + '_log'].min()} to {df[col + '_log'].max()}")

        # 2. Normalize selected features
        numeric_features = [
            'n_bytes_log', 'n_packets_log', 'n_flows_log',
            'average_n_dest_ip', 'average_n_dest_asn', 'average_n_dest_ports',
            'avg_duration', 'avg_ttl'
        ]
        numeric_features = [col for col in numeric_features if col in df.columns]

        if numeric_features:
            self.scaler = RobustScaler()
            df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
            print(f"Normalized features: {numeric_features}")

            os.makedirs("./preprocessed/CESNET_500", exist_ok=True)
            joblib.dump(self.scaler, "./preprocessed/CESNET_500/scaler.save")

        return df

    def preprocess(self, df):
        """
        Full preprocessing pipeline for CESNET-TimeSeries24 with per-IP handling.

        Parameters:
        -----------
        df : pd.DataFrame

        Returns:
        --------
        pd.DataFrame
        """
        print("Raw columns:", df.columns.tolist())

        time_col = 'time' if 'time' in df.columns else 'timestamp'

        df = self.create_time_features(df, timestamp_col=time_col)
        df = self.handle_missing_values(df)
        df = self.add_holiday_features(df, time_col)

        df['is_active'] = 1  # Mark all entries as active

        ip_ids = df['ip_id'].unique()
        processed_dfs = []

        for ip_id in ip_ids:
            ip_df = df[df['ip_id'] == ip_id].copy()
            ip_df = self.create_cesnet_features(ip_df)
            ip_df = self.create_static_features(ip_df)
            processed_dfs.append(ip_df)

        processed_df = pd.concat(processed_dfs, ignore_index=True)
        processed_df = self.transform_features(processed_df)

        print("Preprocessing complete. Final shape:", processed_df.shape)
        return processed_df

    def create_cesnet_features(self, df):
        """
        Create CESNET-specific features for a single IP, including lag features.

        Parameters:
        -----------
        df : pd.DataFrame

        Returns:
        --------
        pd.DataFrame
        """
        if 'tcp_udp_ratio_bytes' in df.columns and 'tcp_udp_ratio_packets' in df.columns:
            df['tcp_udp_diff'] = df['tcp_udp_ratio_bytes'] - df['tcp_udp_ratio_packets']

        if 'dir_ratio_bytes' in df.columns and 'dir_ratio_packets' in df.columns:
            df['dir_ratio_diff'] = df['dir_ratio_bytes'] - df['dir_ratio_packets']

        df = df.sort_values('id_time')

        for col in ['n_flows', 'n_packets', 'n_bytes']:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_lag6'] = df[col].shift(6)
                df[f'{col}_lag24'] = df[col].shift(24)

                df[f'{col}_rolling_mean_6'] = df[col].rolling(window=6, min_periods=1).mean()
                df[f'{col}_rolling_std_6'] = df[col].rolling(window=6, min_periods=1).std()
                df[f'{col}_rolling_max_6'] = df[col].rolling(window=6, min_periods=1).max()

        if 'n_bytes' in df.columns:
            mean = df['n_bytes'].mean()
            std = df['n_bytes'].std()
            if std > 0:
                df['n_bytes_zscore'] = (df['n_bytes'] - mean) / std
                df['is_anomaly'] = (abs(df['n_bytes_zscore']) > 3).astype(int)
            else:
                df['n_bytes_zscore'] = 0
                df['is_anomaly'] = 0

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    def create_static_features(self, df):
        """
        Create static features by aggregating long-term IP-level behavior.
        Used for TFT's static_reals.

        Parameters:
        -----------
        df : pd.DataFrame

        Returns:
        --------
        pd.DataFrame
        """
        static_features = ["average_n_dest_asn", "avg_duration"]
        ip_static = df.groupby("ip_id")[static_features].mean().reset_index()

        ip_static = ip_static.rename(columns={
            "average_n_dest_asn": "avg_asn_static",
            "avg_duration": "avg_duration_static"
        })

        df = df.merge(ip_static, on="ip_id", how="left")
        return df



# Main execution block
if __name__ == "__main__":
    # Dataset paths
    sample_path = r"E:\dataset_download\CESNET-TimeSeries24\ip_addresses_sample.tar.gz"
    times_path = r"E:\dataset_download\CESNET-TimeSeries24\times.tar.gz"
    weekends_path = r"E:\dataset_download\CESNET-TimeSeries24\weekends_and_holidays.csv"

    # Initialize the data formatter
    formatter = CESNETDataFormatter()

    # Load raw CESNET data (limit to 500 IPs)
    data_df = formatter.load_data(sample_path, times_path, weekends_path, num_ips=500)

    # Apply preprocessing pipeline
    processed_df = formatter.preprocess(data_df)

    # Split into train / val / test sets
    train_df, val_df, test_df = formatter.split_data(processed_df)

    # Save splits to output directory
    output_dir = r"D:\PythonProject\chatbot\dataset\preprocessed\CESNET_500"
    formatter.save_splits(train_df, val_df, test_df, output_dir)

    print("CESNET data processing completed!")
