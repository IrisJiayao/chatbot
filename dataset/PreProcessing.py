import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
from abc import ABC, abstractmethod
import os


# ============ STEP 1: ABSTRACT BASE CLASS ============ #
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
import pandas as pd
import numpy as np
import os

class GenericDataFormatter(ABC):
    """
    Abstract base class for data formatting across different network datasets.
    Provides standard methods for normalization, encoding, time feature creation, etc.
    """

    def __init__(self):
        # MinMaxScaler (can be replaced with RobustScaler if needed)
        self.scaler = MinMaxScaler()
        self.label_encoders = {}

    @abstractmethod
    def load_data(self, file_path: str):
        """Load raw dataset from a file (to be implemented by subclasses)."""
        pass

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform dataset-specific preprocessing (to be implemented by subclasses)."""
        pass

    def normalize(self, df: pd.DataFrame, columns: list):
        """Apply MinMax scaling to the specified numeric columns."""
        df[columns] = self.scaler.fit_transform(df[columns])
        return df

    def encode_categorical(self, df: pd.DataFrame, columns: list):
        """Encode categorical features using LabelEncoder."""
        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        return df

    def handle_missing_values(self, df: pd.DataFrame):
        """Replace inf/NaN values with the median of numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df

    def create_time_features(self, df: pd.DataFrame, timestamp_col: str):
        """
        Extract time-based features from a timestamp column:
        - hour, minute, second
        - sine/cosine cyclical encoding
        - weekday, weekend, holiday tag
        """
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')

        mask = df[timestamp_col].isna()
        if mask.any():
            print(f"Filling {mask.sum()} missing timestamps with default value.")
            df.loc[mask, timestamp_col] = pd.Timestamp('2023-01-01')

        # Extract basic time features
        df["hour"] = df[timestamp_col].dt.hour
        df["minute"] = df[timestamp_col].dt.minute
        df["second"] = df[timestamp_col].dt.second

        # Cyclical encoding for time
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
        df["second_sin"] = np.sin(2 * np.pi * df["second"] / 60)
        df["second_cos"] = np.cos(2 * np.pi * df["second"] / 60)

        # Weekday and weekend flag
        df["day_of_week"] = df[timestamp_col].dt.dayofweek
        df["is_holiday"] = df["day_of_week"].apply(lambda x: 1 if x in [5, 6] else 0)  # Saturday/Sunday = holiday

        return df

    def split_data(self, df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
        """
        Split the dataset into training, validation, and test sets by ratio.

        Parameters:
        -----------
        train_ratio : float
        val_ratio : float

        Returns:
        --------
        (train_df, val_df, test_df)
        """
        train_size = int(len(df) * train_ratio)
        val_size = int(len(df) * val_ratio)

        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]

        return train_df, val_df, test_df

    def save_splits(self, train_df, val_df, test_df, output_dir):
        """
        Save the train/val/test splits to CSV files and check normalization.

        Also prints warning if `flow_byts_s` is not properly normalized.
        """
        for dataset_name, dataset in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if "flow_byts_s" in dataset.columns:
                min_val = dataset["flow_byts_s"].min()
                max_val = dataset["flow_byts_s"].max()
                print(f"[Before saving] {dataset_name} flow_byts_s range: {min_val:.6f} to {max_val:.6f}")
                if min_val < 0 or max_val > 1:
                    print(f"Warning: {dataset_name} flow_byts_s not properly normalized!")

        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        print(f"Saved processed splits to {output_dir}")

        # Post-save verification
        for file_name in ["train.csv", "val.csv", "test.csv"]:
            file_path = os.path.join(output_dir, file_name)
            check_df = pd.read_csv(file_path)
            if "flow_byts_s" in check_df.columns:
                min_val = check_df["flow_byts_s"].min()
                max_val = check_df["flow_byts_s"].max()
                print(f"[After load] {file_name} flow_byts_s range: {min_val:.6f} to {max_val:.6f}")

    def create_features(self, df: pd.DataFrame):
        """
        Generate additional features including moving averages and protocol flags.

        - Adds 5-min moving average of flow_byts_s.
        - Converts protocol to one-hot encoding (TCP/UDP/ICMP/Other).
        - Encodes IP addresses and protocol as labels.
        """
        if "flow_byts_s" in df.columns:
            df["flow_bytes_ma_5min"] = df["flow_byts_s"].rolling(window=5, min_periods=1).mean()
        else:
            print("Warning: 'flow_byts_s' column not found!")

        # One-hot protocol encoding
        df["is_tcp"] = (df["protocol"] == 6).astype(int)
        df["is_udp"] = (df["protocol"] == 17).astype(int)
        df["is_icmp"] = (df["protocol"] == 1).astype(int)
        df["is_other"] = (df["protocol"] == 0).astype(int)

        # Label encoding of categorical columns
        encoder = LabelEncoder()
        cols_to_encode = [col for col in ["source_ip", "destination_ip", "protocol"] if col in df.columns]
        for col in cols_to_encode:
            df[col] = encoder.fit_transform(df[col])

        return df



# ============ STEP 2: CUSTOM FORMATTER FOR NETWORK TRAFFIC ============ #
class TrafficDataFormatter(GenericDataFormatter):
    """
    Custom formatter for processing generic network traffic datasets.
    Inherits preprocessing utilities from GenericDataFormatter.
    """

    def load_data(self, file_path: str):
        """Load dataset from a CSV file."""
        return pd.read_csv(file_path)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline to traffic dataset:
        - Time features
        - Missing value handling
        - Feature generation
        - Clipping, log transform, normalization
        - Saving fitted scaler
        """

        # Rename columns for consistency
        df.rename(columns=lambda x: x.strip().replace(" ", "_").replace("/", "_").lower(), inplace=True)
        original_df = df.copy()

        print("1. Original flow_byts_s range:", df["flow_byts_s"].min(), df["flow_byts_s"].max())
        print("DataFrame columns:", df.columns.tolist())

        # Step 1: Time features and missing values
        df = self.create_time_features(df, timestamp_col="timestamp")
        df = self.handle_missing_values(df)
        df = self.create_features(df)

        print("2. After feature creation, flow_byts_s range:", df["flow_byts_s"].min(), df["flow_byts_s"].max())

        # Step 2: Clipping to remove extreme values (top 1%)
        upper_limit = df["flow_byts_s"].quantile(0.99)
        print(f"3. Clipping upper limit (99th percentile): {upper_limit:.4f}")
        df["flow_byts_s"] = df["flow_byts_s"].clip(lower=0, upper=upper_limit)
        print("4. After clipping, flow_byts_s range:", df["flow_byts_s"].min(), df["flow_byts_s"].max())

        # Replace zeros with small positive values (avoid log(0))
        df["flow_byts_s"] = df["flow_byts_s"].replace(0, np.random.uniform(1e-6, 1e-4))
        print("5. After replacing zeros, flow_byts_s range:", df["flow_byts_s"].min(), df["flow_byts_s"].max())

        # Step 3: Create differencing and moving average features (before log transform)
        df["flow_byts_s_diff"] = df["flow_byts_s"].diff().fillna(0)
        df["flow_byts_s_diff_ma_5"] = df["flow_byts_s_diff"].rolling(window=5, min_periods=1).mean()
        df["flow_byts_s_ma_10"] = df["flow_byts_s"].rolling(window=10, min_periods=1).mean()

        # Step 4: Log transformation
        df["flow_byts_s"] = np.log1p(df["flow_byts_s"])
        print("6. After log transform, flow_byts_s range:", df["flow_byts_s"].min(), df["flow_byts_s"].max())

        # Step 5: Normalize
        numeric_cols_to_normalize = ["flow_byts_s"]
        before_norm = df["flow_byts_s"].copy()
        df[numeric_cols_to_normalize] = self.scaler.fit_transform(df[numeric_cols_to_normalize])
        print("7. After normalization, flow_byts_s range:", df["flow_byts_s"].min(), df["flow_byts_s"].max())

        # Check normalization correctness
        if df["flow_byts_s"].min() < 0 or df["flow_byts_s"].max() > 1:
            print("Warning: Normalization may have failed.")
            print("   - Before normalization:", before_norm.iloc[0:5].tolist())
            print("   - After normalization:", df["flow_byts_s"].iloc[0:5].tolist())

            print("   - Attempting to re-apply normalization...")
            from sklearn.preprocessing import MinMaxScaler
            temp_scaler = MinMaxScaler()
            df["flow_byts_s"] = temp_scaler.fit_transform(df[["flow_byts_s"]])
            print("   - After forced normalization:", df["flow_byts_s"].min(), df["flow_byts_s"].max())

        print("8. Final check before return, flow_byts_s range:", df["flow_byts_s"].min(), df["flow_byts_s"].max())
        print("9. Row count change after processing:", len(original_df), "->", len(df))

        # Step 6: Save scaler for inference usage
        dataset_name = "CSE-CIC-IDS2018"
        save_path = f"D:/PythonProject/chatbot/dataset/preprocessed/{dataset_name}"
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(save_path, "scaler.save"))

        # Log scaler internals
        print("10. Scaler object saved.")
        print("    - Scaler range:", self.scaler.data_min_, self.scaler.data_max_)

        return df



# ============ STEP 3: EXECUTION FOR A DATASET ============ #
if __name__ == "__main__":
    dataset_paths = {
        "CSE-CIC-IDS2018": r"D:\PythonProject\chatbot\dataset\CSE-CIC-IDS2018NormalTraffic.csv",
    }


    for dataset_name, file_path in dataset_paths.items():
        print(f"Processing {dataset_name}...")

        formatter = TrafficDataFormatter()
        df = formatter.load_data(file_path)
        processed_df = formatter.preprocess(df)

        train_df, val_df, test_df = formatter.split_data(processed_df)
        formatter.save_splits(train_df, val_df, test_df,
                              output_dir=f"D:/PythonProject/chatbot/dataset/preprocessed/{dataset_name}")

    print("✅ All datasets processed successfully!")

