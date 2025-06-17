import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import glob
import gc


def process_cicids_data(input_dir, output_file, sampling_interval=1):
    """
    Process the CIC-IDS2018 dataset and aggregate network traffic records into fixed time windows
    """
    # Find all CSV files in the input directory
    all_files = glob.glob(os.path.join(input_dir, "*.csv"))

    # Store processed data from all files
    all_data = []

    # Process each file one by one
    for file_path in all_files:
        print(f"Processing file: {file_path}")

        try:
            # Check file size in MB
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"File size: {file_size_mb:.2f} MB")

            # Use chunked reading for large files
            if file_size_mb > 500:
                print("Processing large file in chunks...")
                chunks_data = []
                chunk_size = 100000

                for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size,
                                                              low_memory=False, on_bad_lines='skip')):
                    if chunk_num % 10 == 0:
                        print(f"Processing chunk #{chunk_num}...")

                    # Identify timestamp column
                    if 'Timestamp' in chunk.columns:
                        time_col = 'Timestamp'
                    elif 'timestamp' in chunk.columns:
                        time_col = 'timestamp'
                    else:
                        print("Warning: No timestamp column found in chunk, skipping.")
                        continue

                    # Safely parse timestamp column
                    try:
                        if chunk_num == 0:
                            print(f"Timestamp sample: {chunk[time_col].iloc[:5].tolist()}")

                        try:
                            # Try parsing using specified format
                            chunk[time_col] = pd.to_datetime(chunk[time_col], format='%d/%m/%Y %H:%M:%S',
                                                             errors='coerce')
                        except:
                            # Fallback: infer format automatically
                            chunk[time_col] = pd.to_datetime(chunk[time_col], errors='coerce')

                        # Drop rows with invalid timestamps
                        chunk = chunk.dropna(subset=[time_col])

                        # Check timestamp validity
                        min_date = chunk[time_col].min()
                        max_date = chunk[time_col].max()

                        # If timestamps are earlier than 2018, they may be invalid
                        if min_date.year < 2018:
                            print(f"Warning: Possible timestamp error, minimum time: {min_date}")
                            if min_date.year == 1970:
                                print("Attempting to fix Unix timestamp...")
                                chunk[time_col] = pd.to_datetime(chunk[time_col].astype(int) / 1000, unit='s')

                        # Re-check after possible fix
                        min_date = chunk[time_col].min()
                        max_date = chunk[time_col].max()
                        if min_date.year < 2018 or max_date.year > 2018:
                            print(f"Warning: Timestamps still out of range: {min_date} to {max_date}")
                            chunk = chunk[(chunk[time_col].dt.year == 2018)]

                        if len(chunk) == 0:
                            print("Warning: No valid data left after filtering, skipping chunk.")
                            continue
                    except Exception as e:
                        print(f"Timestamp conversion error: {e}")
                        continue

                    # Create fixed-size time windows (rounded down)
                    chunk['time_window'] = chunk[time_col].dt.floor(f'{sampling_interval}s')

                    # Add this chunk to list
                    chunks_data.append(chunk)

                # Combine all chunks
                if not chunks_data:
                    print("No valid chunks found, skipping file.")
                    continue

                df = pd.concat(chunks_data)
                print(f"Total rows after merging chunks: {len(df)}")

                # Free memory
                del chunks_data
                gc.collect()
            else:
                # For small files, read directly
                df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')

                print(f"\n===== Detailed analysis for file: {file_path} =====")

                # Identify timestamp column
                if 'Timestamp' in df.columns:
                    time_col = 'Timestamp'
                elif 'timestamp' in df.columns:
                    time_col = 'timestamp'
                else:
                    print(f"Warning: No timestamp column found in {file_path}, skipping.")
                    continue

                # Convert timestamps
                try:
                    print(f"Timestamp sample: {df[time_col].iloc[:5].tolist()}")

                    try:
                        df[time_col] = pd.to_datetime(df[time_col], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                    except:
                        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

                    # Drop invalid timestamps
                    df = df.dropna(subset=[time_col])

                    # Print timestamp range for validation
                    min_date = df[time_col].min()
                    max_date = df[time_col].max()
                    print(f"File time range: {max_date - min_date}, from {min_date} to {max_date}")

                    if min_date.year < 2018:
                        print("Warning: Filtering to keep only 2018 data.")
                        df = df[(df[time_col].dt.year == 2018)]
                except Exception as e:
                    print(f"Timestamp conversion error: {e}")
                    continue

                # Create fixed-size time windows (rounded down)
                df['time_window'] = df[time_col].dt.floor(f'{sampling_interval}s')

                # Print hourly record distribution
                time_counts = df[time_col].dt.hour.value_counts().sort_index()
                print(f"Record distribution by hour:\n{time_counts}")

            # Calculate record count per time window
            window_density = df.groupby('time_window').size()
            print(f"Avg records per window: {window_density.mean():.2f}")
            print(f"Max records in any window: {window_density.max()}")
            print(f"Min records in any window: {window_density.min()}")



            # 标记攻击流量
            if 'Label' in df.columns:
                df['is_attack'] = (~df['Label'].str.contains('BENIGN', case=False, na=False)).astype(int)
            elif 'label' in df.columns:
                df['is_attack'] = (~df['label'].str.contains('BENIGN', case=False, na=False)).astype(int)
            else:
                df['is_attack'] = 0
            # Label attack flows (non-BENIGN as 1, BENIGN as 0)
            if 'Label' in df.columns:
                df['is_attack'] = (~df['Label'].str.contains('BENIGN', case=False, na=False)).astype(int)
            elif 'label' in df.columns:
                df['is_attack'] = (~df['label'].str.contains('BENIGN', case=False, na=False)).astype(int)
            else:
                df['is_attack'] = 0  # Assume benign if no label column is found

            # Process the 'Protocol' column and convert to numeric if needed
            if 'Protocol' in df.columns:
                if not pd.api.types.is_numeric_dtype(df['Protocol']):
                    print("Protocol column is not numeric, attempting to convert...")
                    try:
                        # Try direct numeric conversion
                        df['Protocol'] = pd.to_numeric(df['Protocol'], errors='coerce')
                        df['Protocol'] = df['Protocol'].fillna(-1).astype(int)
                        protocol_column = 'Protocol'
                    except:
                        # Fallback: use string-to-number mapping
                        protocol_map = {'tcp': 6, 'TCP': 6, 'udp': 17, 'UDP': 17, 'icmp': 1, 'ICMP': 1}
                        df['Protocol_numeric'] = df['Protocol'].map(protocol_map)
                        df['Protocol_numeric'] = df['Protocol_numeric'].fillna(-1).astype(int)
                        protocol_column = 'Protocol_numeric'
                else:
                    protocol_column = 'Protocol'
            else:
                protocol_column = None
                print("Protocol column not found.")

            # Ensure selected features are converted to numeric (if needed)
            numeric_features = ['Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
                                'Flow Bytes/s', 'Flow Packets/s', 'Total Fwd Packets', 'Total Backward Packets',
                                'Packet Length Min']

            for feature in numeric_features:
                if feature in df.columns:
                    df[feature] = pd.to_numeric(df[feature], errors='coerce')
                    df[feature] = df[feature].fillna(0)

            # Verify and convert all aggregation-relevant columns to numeric
            for feature_name, column_name in {
                'flow_iat_mean': 'Flow IAT Mean',
                'flow_iat_std': 'Flow IAT Std',
                'flow_iat_max': 'Flow IAT Max',
                'flow_iat_min': 'Flow IAT Min',
                'flow_byts_s': 'Flow Bytes/s',
                'flow_pkts_s': 'Flow Packets/s',
                'tot_fwd_pkts': 'Total Fwd Packets',
                'tot_bwd_pkts': 'Total Backward Packets',
                'pkt_len_min': 'Packet Length Min'
            }.items():
                if column_name in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[column_name]):
                        print(f"Warning: Column {column_name} is not numeric, trying to convert...")
                        try:
                            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
                            df[column_name] = df[column_name].fillna(0)
                            print(f"Successfully converted {column_name} to numeric.")
                        except Exception as conv_err:
                            print(f"Failed to convert {column_name}: {conv_err}")
                            # If conversion fails, remove from aggregation dictionary
                            agg_dict.pop(f'{feature_name}_avg', None)
                            agg_dict.pop(f'{feature_name}_max', None)
                            agg_dict.pop(f'{feature_name}_sum', None)
                            agg_dict.pop(f'{feature_name}_min', None)

            # Begin aggregation setup
            print("Starting data aggregation...")

            # Initialize aggregation dictionary
            agg_dict = {'total_flow_count': (time_col, 'count')}

            # Define how each feature should be aggregated
            for feature, column in {
                'flow_iat_mean': 'Flow IAT Mean',
                'flow_iat_std': 'Flow IAT Std',
                'flow_iat_max': 'Flow IAT Max',
                'flow_iat_min': 'Flow IAT Min',
                'flow_byts_s': 'Flow Byts/s',
                'flow_pkts_s': 'Flow Pkts/s',
                'tot_fwd_pkts': 'Tot Fwd Pkts',
                'tot_bwd_pkts': 'Tot Bwd Pkts',
                'pkt_len_min': 'Pkt Len Min'
            }.items():
                if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                    if feature in ['flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min']:
                        agg_dict[f'{feature}_avg'] = (column, 'mean')
                    elif feature in ['flow_byts_s', 'flow_pkts_s']:
                        agg_dict[f'{feature}_avg'] = (column, 'mean')
                        agg_dict[f'{feature}_max'] = (column, 'max')
                        agg_dict[f'{feature}_std'] = (column, 'std')
                        agg_dict[f'{feature}_min'] = (column, 'min')
                    elif feature in ['tot_fwd_pkts', 'tot_bwd_pkts']:
                        agg_dict[f'{feature}_sum'] = (column, 'sum')
                        agg_dict[f'{feature}_avg'] = (column, 'mean')
                    elif feature == 'pkt_len_min':
                        agg_dict[f'{feature}_min'] = (column, 'min')
                    elif feature == 'tcp_flag':
                        agg_dict[f'{feature}_sum'] = (column, 'sum')
                    elif feature == 'dst_port':
                        agg_dict[f'{feature}_unique'] = (column, 'unique')

            # === Add protocol distribution ratios per time window ===
            if protocol_column:
                # (Deprecated approach) Compute all three ratios together using a dict per group
                protocol_groups = df.groupby('time_window')[protocol_column].apply(
                    lambda x: {
                        'tcp_ratio': ((x == 6).sum() / len(x)) if len(x) > 0 else 0,
                        'udp_ratio': ((x == 17).sum() / len(x)) if len(x) > 0 else 0,
                        'icmp_ratio': ((x == 1).sum() / len(x)) if len(x) > 0 else 0
                    }
                ).apply(pd.Series)

            # Compute protocol ratios separately for better clarity and avoid MultiIndex
            if protocol_column and protocol_column in df.columns and pd.api.types.is_numeric_dtype(df[protocol_column]):
                # TCP ratio: percentage of flows with protocol = 6
                tcp_ratio = df.groupby('time_window').apply(
                    lambda group: ((group[protocol_column] == 6).sum() / len(group)) if len(group) > 0 else 0
                )
                tcp_ratio.name = 'tcp_ratio'

                # UDP ratio: percentage of flows with protocol = 17
                udp_ratio = df.groupby('time_window').apply(
                    lambda group: ((group[protocol_column] == 17).sum() / len(group)) if len(group) > 0 else 0
                )
                udp_ratio.name = 'udp_ratio'

                # ICMP ratio: percentage of flows with protocol = 1
                icmp_ratio = df.groupby('time_window').apply(
                    lambda group: ((group[protocol_column] == 1).sum() / len(group)) if len(group) > 0 else 0
                )
                icmp_ratio.name = 'icmp_ratio'

                # Combine the three protocol ratios into a single DataFrame
                protocol_groups = pd.DataFrame({
                    'tcp_ratio': tcp_ratio,
                    'udp_ratio': udp_ratio,
                    'icmp_ratio': icmp_ratio
                })
            else:
                protocol_groups = None
                print(f"Warning: Protocol column {protocol_column} not available or not numeric, skipping protocol ratio.")

            # === Add attack flow statistics to aggregation dictionary ===
            agg_dict['attack_flow_count'] = ('is_attack', 'sum')   # total number of attack flows
            agg_dict['attack_ratio'] = ('is_attack', 'mean')       # proportion of attacks in each time window
            try:
                # Perform group-by aggregation based on time window
                aggregated = df.groupby('time_window').agg(**agg_dict)

                # Merge protocol ratio if available
                if protocol_column:
                    try:
                        # Ensure index types match
                        if not isinstance(protocol_groups.index, type(aggregated.index)):
                            protocol_groups.index = pd.to_datetime(protocol_groups.index)

                        aggregated = pd.merge(
                            aggregated, protocol_groups,
                            left_index=True, right_index=True,
                            how='left'
                        )
                        print("Successfully merged protocol ratio data.")
                    except Exception as e:
                        print(f"Error merging protocol ratios: {e}")

                # ===== Dst Port aggregation (mode, entropy, well-known port ratio) =====
                if 'Dst Port' in df.columns:
                    try:
                        # Most frequent destination port (mode)
                        dominant_port = df.groupby('time_window')['Dst Port'].agg(
                            lambda x: x.mode().iloc[0] if not x.mode().empty else -1
                        )
                        dominant_port.name = 'dominant_port'

                        # Entropy of port distribution (diversity)
                        def calc_entropy(series):
                            counts = series.value_counts(normalize=True)
                            return -(counts * np.log2(counts + 1e-9)).sum()

                        port_entropy = df.groupby('time_window')['Dst Port'].agg(calc_entropy)
                        port_entropy.name = 'port_entropy'

                        # Ratio of well-known ports (port < 1024)
                        well_known_ratio = df.groupby('time_window')['Dst Port'].apply(
                            lambda x: (x < 1024).sum() / len(x) if len(x) > 0 else 0
                        )
                        well_known_ratio.name = 'well_known_ratio'

                        # Combine all port statistics
                        dstport_stats = pd.concat([dominant_port, port_entropy, well_known_ratio], axis=1)
                    except Exception as e:
                        print(f"Error aggregating Dst Port statistics: {e}")
                        dstport_stats = None
                else:
                    dstport_stats = None

                # Merge Dst Port statistics into main DataFrame
                if dstport_stats is not None:
                    try:
                        if not isinstance(dstport_stats.index, type(aggregated.index)):
                            dstport_stats.index = pd.to_datetime(dstport_stats.index)

                        aggregated = pd.merge(
                            aggregated, dstport_stats,
                            left_index=True, right_index=True,
                            how='left'
                        )
                        print("Successfully merged Dst Port statistics.")

                        # Convert dominant port to categorical labels (e.g., web, ssh, mail)
                        if 'dominant_port' in aggregated.columns:
                            def map_port_category(port):
                                try:
                                    port = int(port)
                                    if port in [80, 443, 8080]:
                                        return "web"
                                    elif port == 53:
                                        return "dns"
                                    elif port == 22:
                                        return "ssh"
                                    elif port in [25, 110, 143]:
                                        return "mail"
                                    elif port < 1024:
                                        return "well_known"
                                    elif port < 49152:
                                        return "registered"
                                    else:
                                        return "dynamic"
                                except:
                                    return "unknown"

                            aggregated['port_category'] = aggregated['dominant_port'].apply(map_port_category)
                            print("Port category field generated.")
                    except Exception as e:
                        print(f"Error merging Dst Port features: {e}")

                # Determine which protocol dominates each window (0=TCP, 1=UDP, 2=ICMP)
                if all(col in aggregated.columns for col in ['tcp_ratio', 'udp_ratio', 'icmp_ratio']):
                    def dominant_protocol_fn(row):
                        if row['tcp_ratio'] >= max(row['udp_ratio'], row['icmp_ratio']):
                            return 0  # TCP
                        elif row['udp_ratio'] >= row['icmp_ratio']:
                            return 1  # UDP
                        else:
                            return 2  # ICMP

                    aggregated['dominant_protocol'] = aggregated[['tcp_ratio', 'udp_ratio', 'icmp_ratio']].apply(
                        dominant_protocol_fn, axis=1
                    )

                print(f"Aggregation complete. Number of time windows: {len(aggregated)}")

                # Check and fix index type
                print(f"Index type after aggregation: {type(aggregated.index)}")
                if isinstance(aggregated.index, pd.MultiIndex):
                    print("MultiIndex detected. Flattening to single-level index...")
                    aggregated = aggregated.reset_index()
                    aggregated['time_window'] = pd.to_datetime(aggregated['time_window'])
                    aggregated.set_index('time_window', inplace=True)
                elif not isinstance(aggregated.index, pd.DatetimeIndex):
                    print("Converting index to DatetimeIndex...")
                    aggregated = aggregated.reset_index()
                    aggregated['time_window'] = pd.to_datetime(aggregated['time_window'])
                    aggregated.set_index('time_window', inplace=True)

                # Extract time-based features from the datetime index
                aggregated['hour'] = aggregated.index.hour
                aggregated['minute'] = aggregated.index.minute
                aggregated['second'] = aggregated.index.second
                aggregated['day_of_week'] = aggregated.index.dayofweek
                aggregated['is_weekend'] = (aggregated.index.dayofweek >= 5).astype(int)
                aggregated['is_business_hours'] = ((aggregated.index.hour >= 9) &
                                                   (aggregated.index.hour < 17) &
                                                   (aggregated.index.dayofweek < 5)).astype(int)

                # Add file name source as a column
                file_name = os.path.basename(file_path)
                aggregated['file_source'] = file_name

                # Append result to global list
                all_data.append(aggregated)
                print(f"Successfully processed file: {file_path}")
            except Exception as e:
                print(f"Error during aggregation: {e}")
                import traceback
                print(traceback.format_exc())
                continue

        except Exception as e:
            print(f"Error processing file: {e}")
            print("Skipping this file.")
            continue

    # === Combine all per-file results into one DataFrame ===
    if all_data:
        try:
            combined_data = pd.concat(all_data)

            # Sort by time index
            combined_data = combined_data.sort_index()
            print(combined_data.index)

            # Check if column index is multi-level
            if isinstance(combined_data.columns, pd.MultiIndex):
                print("Detected multi-level column structure:")
                print(combined_data.columns)

            # Save combined result to CSV
            combined_data.to_csv(output_file)
            print(f"Aggregated data saved to: {output_file}")
            print(f"Total number of time windows: {len(combined_data)}")

            # === Check and rename unnamed column '0' if it exists ===
            if 0 in combined_data.columns:
                if 'tcp_ratio' in combined_data.columns and 'udp_ratio' in combined_data.columns:
                    # Correlate with known protocol ratio columns
                    tcp_corr = combined_data[0].corr(combined_data['tcp_ratio'])
                    udp_corr = combined_data[0].corr(combined_data['udp_ratio'])
                    icmp_corr = combined_data[0].corr(combined_data['icmp_ratio']) if 'icmp_ratio' in combined_data.columns else 0

                    print(f"Correlation with tcp_ratio: {tcp_corr}")
                    print(f"Correlation with udp_ratio: {udp_corr}")
                    if 'icmp_ratio' in combined_data.columns:
                        print(f"Correlation with icmp_ratio: {icmp_corr}")

                    # Rename based on the highest correlation
                    if max(tcp_corr, udp_corr, icmp_corr) == tcp_corr:
                        combined_data.rename(columns={0: 'other_tcp_ratio'}, inplace=True)
                    elif max(tcp_corr, udp_corr, icmp_corr) == udp_corr:
                        combined_data.rename(columns={0: 'other_udp_ratio'}, inplace=True)
                    elif max(tcp_corr, udp_corr, icmp_corr) == icmp_corr:
                        combined_data.rename(columns={0: 'other_icmp_ratio'}, inplace=True)
                    else:
                        combined_data.rename(columns={0: 'unknown_protocol_ratio'}, inplace=True)
                else:
                    # Fallback: assign a generic name
                    combined_data.rename(columns={0: 'unknown_protocol_ratio'}, inplace=True)

                print(
                    f"Renamed column '0' to '{combined_data.columns[combined_data.columns.get_loc(0) if 0 in combined_data.columns else combined_data.columns.get_loc('unknown_protocol_ratio')]}'"
                )

            return combined_data
        except Exception as e:
            print(f"Error while merging data: {e}")
            return None
    else:
        print("No data found to process.")
        return None
# Example usage
def main():
    input_directory = "E:/dataset_download/CIC-IDS2018"  # Path to raw CSV files
    output_file = r"D:\PythonProject\chatbot\dataset\cicids2018_1s_aggregated.csv"  # Output path

    # Run the full data processing and aggregation
    aggregated_data = process_cicids_data(input_directory, output_file)

    # Display statistics from final result
    if aggregated_data is not None:
        print("Dataset summary statistics:")
        print(aggregated_data.describe())

        # Show number and percentage of windows containing attacks
        attack_windows = aggregated_data[aggregated_data['attack_ratio'] > 0]
        if len(attack_windows) > 0:
            print(
                f"Number of windows with attacks: {len(attack_windows)} "
                f"({len(attack_windows) / len(aggregated_data) * 100:.2f}%)"
            )
        else:
            print("No attack windows found.")


if __name__ == "__main__":
    main()
