# # # import pandas as pd
# # # import numpy as np
# # #
# # # class TrafficDataFormatter:
# # #     """负责格式化预测结果（反归一化），但不进行归一化或特征处理"""
# # #
# # #     def __init__(self, scaler):
# # #         """接收 dataset 对象，以便使用其 `target_scaler`"""
# # #         self.target_scaler = scaler  # 直接使用 `Dataset` 里已经训练好的 `target_scaler`
# # #         # self.default_scaler = dataset.default_scaler
# # #         self.missing_identifiers = set()
# # #
# # #     def format_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
# # #         """将预测结果从归一化状态转换回原始单位"""
# # #
# # #         if self.target_scaler is None:
# # #             raise ValueError("Scalers have not been set!")
# # #
# # #         df_list = []
# # #         for identifier, sliced in predictions.groupby("identifier"):
# # #             sliced_copy = sliced.copy()
# # #
# # #             # # 反归一化
# # #             # for col in sliced.columns:
# # #             #     if col not in {"forecast_time", "identifier"}:
# # #             #         if identifier in self.target_scaler:
# # #             #             sliced_copy[col] = self.target_scaler[identifier].inverse_transform(
# # #             #                 sliced_copy[col].values.reshape(-1, 1)
# # #             #             )
# # #             #         else:
# # #             #             print(f"⚠️ Warning: identifier {identifier} not found in target_scaler!")
# # #
# # #             # 反归一化
# # #             for col in sliced.columns:
# # #                 if col not in {"forecast_time", "identifier"}:
# # #                     identifier = str(identifier)  # 统一转换
# # #                     if identifier in self.target_scaler:
# # #                         sliced_copy[col] = self.target_scaler[identifier].inverse_transform(
# # #                             sliced_copy[col].values.reshape(-1, 1)
# # #                         )
# # #                     else:
# # #                         #print(f"⚠️ Warning: identifier {identifier} not found in target_scaler! Using default_scaler.")
# # #                         self.missing_identifiers.add(identifier)
# # #                         # print(f"⚠️ Warning: identifier {identifier} not found in target_scaler! Using default_scaler.")
# # #                         sliced_copy[col] = self.default_scaler.inverse_transform(
# # #                             sliced_copy[col].values.reshape(-1, 1)
# # #                         )  # 用默认的 scaler
# # #
# # #             df_list.append(sliced_copy)
# # #
# # #         return pd.concat(df_list, axis=0)
# # #
# # import pandas as pd
# # from sklearn.preprocessing import MinMaxScaler
# # import joblib
# #
# # class TrafficDataFormatter:
# #     """仅用于反归一化预测数据"""
# #
# #     def __init__(self, scaler_path):
# #         """直接加载预处理阶段保存的MinMaxScaler"""
# #         self.scaler = joblib.load(scaler_path)
# #
# #     def format_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
# #         """将预测结果从归一化状态转换回原始单位"""
# #         predictions_copy = predictions.copy()
# #
# #         skip_columns = ["identifier", "forecast_time"]
# #         for col in predictions_copy.columns:
# #             if col not in skip_columns:
# #                 predictions_copy[col] = self.scaler.inverse_transform(
# #                     predictions_copy[col].values.reshape(-1, 1)
# #                 ).flatten()
# #
# #         return predictions_copy
#
#
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# import joblib
#
# class TrafficDataFormatter:
#     """仅用于反归一化预测数据"""
#
#     def __init__(self, scaler_path):
#         """直接加载预处理阶段保存的MinMaxScaler"""
#         self.scaler = joblib.load(scaler_path)
#         self.scaled_columns = ["flow_byts_s", "flow_pkts_s", "flow_bytes_ma_5min"]
#
#     def format_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
#         """将预测结果从归一化状态转换回原始单位"""
#         predictions_copy = predictions.copy()
#
#         skip_columns = ["identifier", "forecast_time"]
#
#         # 检查并准备数据进行反归一化
#         data_to_inverse = predictions_copy.drop(columns=skip_columns, errors='ignore').values
#
#         # 若只有单列，扩展成三列（因为Scaler要求3列）
#         if data_to_inverse.shape[1] == 1:
#             data_to_inverse = pd.DataFrame(
#                 data_to_inverse.repeat(len(self.scaled_columns), axis=1),
#                 columns=self.scaled_columns
#             )
#         else:
#
#             #data_to_inverse = pd.DataFrame(data_to_inverse, columns=self.scaled_columns[:data_to_inverse.shape[1]])
#             data_to_inverse = pd.DataFrame(data_to_inverse, columns=[f"t+{i}" for i in range(data_to_inverse.shape[1])])
#
#         # 反归一化
#         inverse_transformed = self.scaler.inverse_transform(data_to_inverse)
#
#         # 仅取回原本的预测列
#         for idx, col in enumerate(predictions_copy.columns):
#             if col not in skip_columns:
#                 predictions_copy[col] = inverse_transformed[:, idx]
#
#         return predictions_copy


import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TrafficDataFormatter:
    """仅用于反归一化 flow_byts_s 单列数据"""

    def __init__(self, scaler_path):
        """加载只针对 flow_byts_s 训练的 MinMaxScaler"""
        self.scaler = joblib.load(scaler_path)

    def format_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        只对 `flow_byts_s` 做反归一化，若 predictions 里包含多列时间步 (t+0, t+1, ...),
        需要展平后再 reshape 回去
        """
        predictions_copy = predictions.copy()

        # 如果 predictions 中没有 flow_byts_s，就直接返回
        if "flow_byts_s" not in predictions_copy.columns:
            return predictions_copy

        # 提取 shape (batch_size, time_steps) => flatten => (batch_size * time_steps, )
        arr_2d = predictions_copy["flow_byts_s"].values  # shape (batch_size*time_steps, )
        # 也可能是 (batch_size,) 但一般是 flattened

        # 若是 2D，比如 (256,48)，则需要先 flatten
        # 这里直接检查维度
        if arr_2d.ndim == 2:
            # flatten => (256*48,)
            arr_2d = arr_2d.ravel()

        # reshape => (N,1)
        arr_2d = arr_2d.reshape(-1,1)

        # 反归一化
        inv = self.scaler.inverse_transform(arr_2d).ravel()  # shape (N,)

        # 再 reshape 回原来的形状
        # 需要知道 batch_size 和 time_steps
        # 如果你知道 time_steps=48，可以用:
        #   batch_size = len(predictions_copy)  # 这里需确保 DataFrame 行数 == batch_size
        #   inv = inv.reshape(batch_size, 48)

        # 但目前 predictions_copy 可能是 (batch_size, 49) => "identifier" + 48个列...
        # 如果你只存了一列 "flow_byts_s" (flatten后), 直接赋值即可
        predictions_copy["flow_byts_s"] = inv  # 若 DataFrame 行数 == N


        return predictions_copy

