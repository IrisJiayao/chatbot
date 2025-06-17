import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler, StandardScaler
from abc import ABC, abstractmethod
import os
import numpy as np


# ============ STEP 1: ABSTRACT BASE CLASS ============ #
from abc import ABC, abstractmethod
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class GenericDataFormatter(ABC):
    """
    Abstract base class for data formatting across different network datasets.
    Defines common preprocessing methods including normalization, time feature extraction,
    categorical encoding, and data splitting.
    """

    def __init__(self):
        # MinMaxScaler by default; can be replaced with RobustScaler if needed
        self.scaler = MinMaxScaler()
        self.label_encoders = {}

    @abstractmethod
    def load_data(self, file_path: str):
        """Loads raw dataset from file (to be implemented by subclass)."""
        pass

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs dataset-specific preprocessing logic (to be implemented by subclass)."""
        pass

    def normalize(self, df: pd.DataFrame, columns: list):
        """
        Normalize specified numeric columns using MinMaxScaler.

        Parameters:
        - df : pd.DataFrame
        - columns : list[str]
        """
        df[columns] = self.scaler.fit_transform(df[columns])
        return df

    def encode_categorical(self, df: pd.DataFrame, columns: list):
        """
        Encode categorical columns using LabelEncoder.

        Parameters:
        - df : pd.DataFrame
        - columns : list[str]
        """
        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        return df

    def handle_missing_values(self, df: pd.DataFrame):
        """
        Replace infinite values with NaN, and fill missing numeric values with column median.

        Parameters:
        - df : pd.DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df

    def create_time_features(self, df: pd.DataFrame, timestamp_col: str):
        """
        Extract cyclical time features from pre-parsed 'hour', 'minute', 'second' columns.

        Parameters:
        - df : pd.DataFrame
        - timestamp_col : str (not used in logic but kept for compatibility)
        """
        df = df.copy()

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
        df["second_sin"] = np.sin(2 * np.pi * df["second"] / 60)
        df["second_cos"] = np.cos(2 * np.pi * df["second"] / 60)

        return df

    def split_data_by_ratio(self, df: pd.DataFrame, time_col='time_window', ratio=(0.8, 0, 0.2)):
        """
        Split dataset chronologically based on a timestamp column.

        Parameters:
        - df : pd.DataFrame
        - time_col : str, timestamp column for sorting
        - ratio : tuple(float), ratios for train/val/test

        Returns:
        - train_df, val_df, test_df : pd.DataFrames
        """
        df = df.sort_values(time_col)
        total = len(df)
        train_end = int(total * ratio[0])
        val_end = train_end + int(total * ratio[1])

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        return train_df, val_df, test_df

    def save_splits(self, train_df, val_df, test_df, output_dir):
        """
        Save train, validation, and test splits to CSV files.
        Includes checks to ensure 'flow_byts_s' is normalized within [0, 1].

        Parameters:
        - train_df, val_df, test_df : pd.DataFrame
        - output_dir : str, path to save output files
        """
        for dataset_name, dataset in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if "flow_byts_s" in dataset.columns:
                min_val = dataset["flow_byts_s"].min()
                max_val = dataset["flow_byts_s"].max()
                print(f"[Before Save] {dataset_name} - flow_byts_s range: {min_val:.6f} to {max_val:.6f}")
                if min_val < 0 or max_val > 1:
                    print(f"Warning: {dataset_name} 'flow_byts_s' not normalized!")

        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        print(f"Saved processed data to {output_dir}")

        for file_name in ["train.csv", "val.csv", "test.csv"]:
            file_path = os.path.join(output_dir, file_name)
            check_df = pd.read_csv(file_path)
            if "flow_byts_s" in check_df.columns:
                min_val = check_df["flow_byts_s"].min()
                max_val = check_df["flow_byts_s"].max()
                print(f"[After Load] {file_name} - flow_byts_s range: {min_val:.6f} to {max_val:.6f}")

    def create_features(self, df: pd.DataFrame):
        """
        Create rolling window and ratio-based features from raw columns.

        Features include:
        - Moving averages (5, 10 windows)
        - Standard deviation
        - Directional packet ratio
        - Byte-to-packet ratio
        - Binary attack flag (if 'attack_ratio' column exists)

        Parameters:
        - df : pd.DataFrame

        Returns:
        - df : pd.DataFrame with new features
        """
        for col in ['total_bytes', 'total_flow_count', 'flow_byts_s_avg', 'flow_pkts_s_avg']:
            if col in df.columns:
                df[f'{col}_ma_5'] = df[col].rolling(window=5, min_periods=1).mean()
                df[f'{col}_ma_10'] = df[col].rolling(window=10, min_periods=1).mean()

        df["flow_byts_s_avg_diff"] = df["flow_byts_s_avg"].diff()
        df["flow_byts_s_avg_std_5s"] = df["flow_byts_s_avg"].rolling(5).std()
        df["fwd_bwd_ratio"] = df["tot_fwd_pkts_sum"] / (df["tot_bwd_pkts_sum"] + 1e-5)
        df["byts_pkts_ratio"] = df["flow_byts_s_avg"] / (df["flow_pkts_s_avg"] + 1e-5)

        df.fillna(method='bfill', inplace=True)

        if 'attack_ratio' in df.columns:
            df['is_attack_window'] = (df['attack_ratio'] > 0).astype(int)

        return df




# ============ STEP 2: CUSTOM FORMATTER FOR NETWORK TRAFFIC ============ #
class TrafficDataFormatter(GenericDataFormatter):
    """Custom implementation for processing general network traffic datasets."""

    def load_data(self, file_path: str):
        """Loads dataset from a CSV file."""
        return pd.read_csv(file_path)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the complete preprocessing pipeline:
        - Column cleaning
        - Time feature extraction
        - Missing value handling
        - Log transform + normalization
        - Feature engineering
        - Scaler saving
        """

        # Clean column names: remove spaces and special characters
        df.rename(columns=lambda x: x.strip().replace(" ", "_").replace("/", "_").lower(), inplace=True)
        original_df = df.copy()
        print("1. Original flow_byts_s_avg range:", df["flow_byts_s_avg"].min(), df["flow_byts_s_avg"].max())
        print("DataFrame Columns:", df.columns.tolist())

        # Extract time-based cyclical features
        df = self.create_time_features(df, timestamp_col="timestamp")

        # Handle missing or infinite values
        df = self.handle_missing_values(df)

        # Apply custom feature generation
        df = self.create_features(df)
        print("2. After feature generation, flow_byts_s_avg range:", df["flow_byts_s_avg"].min(), df["flow_byts_s_avg"].max())

        # Pre-normalization transformations
        for col in ['flow_byts_s_avg', 'flow_pkts_s_avg', 'tot_fwd_pkts_sum', 'tot_bwd_pkts_sum']:
            if col in df.columns:
                df[col] = df[col].replace(0, np.random.uniform(1e-6, 1e-4))  # Avoid log(0)
                df[col] = np.log1p(df[col])  # Apply log(1 + x)
                before_norm = df[col].copy()
                print(f"3. After log transform, {col} range: {df[col].min()} to {df[col].max()}")

        print("Standardizing numeric features...")

        # Select numeric columns (excluding time/categorical fields)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['hour', 'minute', 'second', 'day_of_week', 'is_weekend',
                        'is_business_hours', 'dominant_protocol', 'is_attack_window',
                        'file_source', 'dominant_port', 'port_category']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Normalize all numeric columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])

        # Validate normalization of flow_byts_s_avg
        if col in df[numeric_cols]:
            if df[col].min() < 0 or df[col].max() > 1:
                print("Warning: Normalization may not have applied correctly!")
                print("   - Values before normalization:", before_norm.iloc[0:5].tolist())
                print("   - Values after normalization:", df['flow_byts_s_avg'].iloc[0:5].tolist())

                # Attempt re-normalization using fallback
                print("   - Trying to re-apply normalization...")
                from sklearn.preprocessing import MinMaxScaler
                temp_scaler = MinMaxScaler()
                values = df[col].values.reshape(-1, 1)
                df[col] = temp_scaler.fit_transform(values).flatten()
                print("   - After forced normalization:", df[col].min(), df[col].max())
            print("8. Final check before return: flow_byts_s_avg =", df[col].min(), "to", df[col].max())

        print("9. Row count before/after processing:", len(original_df), "->", len(df))

        # Save scaler for inference reuse
        dataset_name = "CSE-CIC-IDS2018_1s"
        save_path = f"D:/PythonProject/chatbot/dataset/preprocessed/{dataset_name}"
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(save_path, "scaler.save"))

        print("Final column overview:")
        print(df.columns.tolist())


        return df

    def split_data_by_ratio(self, df: pd.DataFrame, time_col='time_window', ratio=(0.8, 0, 0.2)):
        """
        Chronologically split data into train/val/test sets based on timestamp.

        Parameters:
        - df : pd.DataFrame
        - time_col : str, column to sort by time
        - ratio : tuple, percentage split for train/val/test
        """
        df = df.sort_values(time_col)
        total = len(df)
        train_end = int(total * ratio[0])
        val_end = train_end + int(total * ratio[1])
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        return train_df, val_df, test_df


# ============ STEP 3: EXECUTION FOR A DATASET ============ #
if __name__ == "__main__":
    dataset_paths = {
        "CSE-CIC-IDS2018_1s": r"D:\PythonProject\chatbot\dataset\cicids2018_1s_aggregated.csv",
    }


    for dataset_name, file_path in dataset_paths.items():
        print(f"Processing {dataset_name}...")

        formatter = TrafficDataFormatter()
        df = formatter.load_data(file_path)
        processed_df = formatter.preprocess(df)

        # Use continuous time segment segmentation and add group_id
        train_df, val_df, test_df = formatter.split_data_by_ratio(processed_df)
        formatter.save_splits(train_df, val_df, test_df,
                              output_dir=f"D:/PythonProject/chatbot/dataset/preprocessed/{dataset_name}")

    print("✅ All datasets processed successfully!")

