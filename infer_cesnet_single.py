import numpy as np
import logging
import torch
import pandas as pd

from time import time
from torch.utils.data import DataLoader
from config.config import Config
from dataset.Dataset import Dataset
from temporal_fusion_transformer import TemporalFusionTransformer
from utils import (
    QuantileLoss, symmetric_mean_absolute_percentage_error,
    unnormalize_tensor, plot_temporal_serie
)
from dataset.traffic_data_formatter import TrafficDataFormatter
from dataset.Dataset_cesnet import CESNETDataset


class InferenceSingleStep:
    """
    Class for loading and testing the pre-trained model
    """

    def __init__(self, cnf):
        self.cnf = cnf
        test_path = r"D:\PythonProject\chatbot\dataset\preprocessed\CESNET\val.csv"

        # ✅ 加载测试数据
        self.dataset_test = CESNETDataset(test_path)
        self.formatter = TrafficDataFormatter(
            scaler_path=r"D:\PythonProject\chatbot\dataset\preprocessed\CESNET\scaler.save")

        # ✅ 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ 自动设置 input_size
        input_dim = self.dataset_test.inputs.shape[-1]
        self.cnf.all_params["input_size"] = input_dim
        self.cnf.all_params["input_obs_loc"] = self.dataset_test.input_obs_loc

        # ✅ 初始化模型
        self.model = TemporalFusionTransformer(self.cnf.all_params).to(self.device)

        # ✅ 读取最优模型权重
        self.load_checkpoint()

        # ✅ 初始化测试 DataLoader
        self.test_loader = DataLoader(
            dataset=self.dataset_test, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=False, pin_memory=True
        )

        # ✅ 初始化损失函数
        self.loss = QuantileLoss(cnf.quantiles)

        # ✅ 记录日志
        self.log_file = self.cnf.exp_log_path / "inference_log.txt"
        logging.basicConfig(
            filename=self.log_file,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )

    def load_checkpoint(self):
        from pathlib import Path
        """加载最佳模型"""
        # ck_path = self.cnf.exp_log_path / f"{self.cnf.exp_name}_best.pth"
        ck_path = Path(r"D:\PythonProject\chatbot\log\CESNET\03-25-2025-21-34-44\epoch_16.pth")
        if ck_path.exists():
            checkpoint = torch.load(ck_path, map_location='cuda', weights_only=False)
            self.model.load_state_dict(checkpoint['model'])
            # self.model.load_state_dict(torch.load(ck_path,weights_only=False))
            print(f"[Loaded best model from '{ck_path}']")
        else:
            raise FileNotFoundError(f"Checkpoint '{ck_path}' not found!")

    def inverse_transform(self, arr):
        """将标准化的数据转换回原始尺度"""
        # 确保输入是CPU张量
        if isinstance(arr, torch.Tensor):
            if arr.is_cuda:
                arr = arr.cpu()  # 从GPU移到CPU
            arr = arr.numpy()

        # 假设目标列是第0列
        target_col = 0

        # 创建dummy数组来执行inverse_transform
        dummy = np.zeros((arr.shape[0], self.formatter.scaler.scale_.shape[0]))
        dummy[:, target_col] = arr[:, 0]  # 现在arr已经是numpy数组

        inv = self.formatter.scaler.inverse_transform(dummy)[:, target_col]
        return np.expm1(inv)





    def visualize_attention(self, attention, components):
        """可视化注意力权重"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os

        # 创建保存图像的目录
        save_dir = self.cnf.exp_log_path / "attention_viz"
        os.makedirs(save_dir, exist_ok=True)

        # 提取注意力权重
        temporal_attention = attention[0].cpu().numpy().squeeze()  # [时间步, 时间步]
        variable_attention = attention[1].cpu().numpy().squeeze()  # [变量数]

        # 获取变量名称（如果有的话）
        variable_names = getattr(self.dataset_test, 'feature_names',
                                 [f"特征{i}" for i in range(variable_attention.shape[0])])

        # 1. 绘制时间注意力热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(temporal_attention, cmap='viridis', annot=False)
        plt.title('时间步注意力分布')
        plt.xlabel('目标时间步')
        plt.ylabel('源时间步')
        plt.tight_layout()
        plt.savefig(save_dir / "temporal_attention.png")
        plt.close()

        # 2. 绘制变量注意力条形图
        plt.figure(figsize=(12, 6))
        plt.bar(variable_names, variable_attention)
        plt.title('变量重要性分布')
        plt.xlabel('特征')
        plt.ylabel('注意力权重')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_dir / "variable_attention.png")
        plt.close()

        print(f"注意力可视化结果已保存至: {save_dir}")

    def run_inference_single_step(self):
        """
        运行单步预测 - 对每个样本只预测下一个时间步
        不进行递推预测，只使用模型编码器部分编码固定的时间窗口(num_encoder_steps)
        """
        try:
            # 设置模型为评估模式
            self.model.eval()

            # 存储所有预测和真实值
            all_preds = []
            all_trues = []

            total_steps = len(self.dataset_test)
            print(f"🚀 开始单步预测，总共 {total_steps} 条记录")

            with torch.no_grad():
                for idx in range(total_steps):
                    try:
                        # 获取一个样本
                        sample = self.dataset_test[idx]

                        # 准备输入数据 - 已经包含了固定的编码器时间窗口
                        x = sample['inputs'].unsqueeze(0).to(self.device)  # shape: [1, input_len, feature_dim]

                        # 执行模型预测，处理可能的错误
                        try:
                            # 对当前时间窗口预测下一个时间步
                            out, _, _ = self.model(x)
                            pred = out[0, -1, 1].cpu().numpy()  # 中位数预测
                        except RuntimeError as e:
                            if "stack expects each tensor to be equal size" in str(e):
                                print(f"样本 {idx} 处理失败，跳过: {e}")
                                continue
                            else:
                                raise e

                        # 获取真实值
                        y_true = sample['outputs'][0, 0].item()

                        # 保存预测和真实值
                        all_preds.append(pred)
                        all_trues.append(y_true)

                        # 打印进度
                        if idx % 100 == 0 or idx == total_steps - 1:
                            print(f"[{idx + 1}/{total_steps}] Pred: {pred:.4f}, True: {y_true:.4f}")
                    except Exception as e:
                        print(f"处理样本 {idx} 时出错: {e}")
                        continue  # 跳过有问题的样本

                # 检查是否有有效预测
                if len(all_preds) == 0:
                    print("没有成功的预测，无法继续计算指标")
                    return

                # 转换为张量并添加维度
                pred_tensor = torch.tensor(all_preds).unsqueeze(1)
                true_tensor = torch.tensor(all_trues).unsqueeze(1)

                # 反归一化数据
                try:
                    pred_inv = self.inverse_transform(pred_tensor)
                    true_inv = self.inverse_transform(true_tensor)
                except Exception as e:
                    print(f"反归一化失败: {e}")
                    # 如果反归一化失败，使用原始值
                    pred_inv = pred_tensor.numpy()
                    true_inv = true_tensor.numpy()

                # 计算和打印指标
                self.calculate_metrics(pred_inv, true_inv)

                # 可视化结果
                self.visualize_results(true_inv, pred_inv)

                return pred_inv, true_inv

        except Exception as e:
            print(f"预测过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def calculate_metrics(self, pred_inv, true_inv):
        """计算和打印评估指标"""
        try:
            # 计算SMAPE
            smape = symmetric_mean_absolute_percentage_error(pred_inv, true_inv)

            # 计算MSE
            mse = np.mean((pred_inv - true_inv) ** 2)

            # 计算R²和Pearson相关系数（需要处理可能的NaN值）
            valid_mask = ~np.isnan(true_inv.flatten()) & ~np.isnan(pred_inv.flatten())
            if np.sum(valid_mask) > 1:  # 至少需要两个有效点
                r2 = np.corrcoef(true_inv.flatten()[valid_mask], pred_inv.flatten()[valid_mask])[0, 1] ** 2
                pearson = np.corrcoef(true_inv.flatten()[valid_mask], pred_inv.flatten()[valid_mask])[0, 1]
            else:
                r2 = np.nan
                pearson = np.nan

            # 计算平均预测/实际比率
            ratio = np.mean(pred_inv / true_inv)

            # 打印结果
            print(f"\n🎯 单步预测评估结果:")
            print(f"  - SMAPE: {smape:.6f}")
            print(f"  - MSE: {mse:.6f}")
            print(f"  - R² Score: {r2:.6f}")
            print(f"  - Pearson Correlation: {pearson:.6f}")
            print(f"  - 平均预测/实际比率: {ratio:.6f}")

            # 记录到日志
            logging.info(
                f"单步预测评估结果: SMAPE={smape:.6f}, MSE={mse:.6f}, R2={r2:.6f}, Pearson={pearson:.6f}, Ratio={ratio:.6f}")

            return {
                "smape": smape,
                "mse": mse,
                "r2": r2,
                "pearson": pearson,
                "ratio": ratio
            }
        except Exception as e:
            print(f"计算指标时出错: {e}")
            return None

    def visualize_results(self, true_inv, pred_inv):
        """可视化预测结果"""
        import matplotlib.pyplot as plt
        import os

        # 创建保存图像的目录
        save_dir = self.cnf.exp_log_path / "visualizations"
        os.makedirs(save_dir, exist_ok=True)

        # 绘制整体预测曲线
        plt.figure(figsize=(12, 6))
        plt.plot(true_inv.flatten(), label='True', color='blue', alpha=0.7)
        plt.plot(pred_inv.flatten(), label='Predicted', color='red', alpha=0.7)
        plt.legend()
        plt.title("True vs Predicted Values")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)

        # 保存图像
        plt.tight_layout()
        plt.savefig(save_dir / "prediction_overall.png")
        plt.close()  # 关闭图表而不是显示

        # 绘制部分放大图 (如果数据点超过100个)
        if len(true_inv) > 100:
            # 选择最后100个点
            plt.figure(figsize=(12, 6))
            plt.plot(true_inv.flatten()[-100:], label='True', color='blue')
            plt.plot(pred_inv.flatten()[-100:], label='Predicted', color='red')
            plt.legend()
            plt.title("True vs Predicted (Last 100 Points)")
            plt.xlabel("Time Step")
            plt.ylabel("Value")
            plt.grid(True, alpha=0.3)

            # 保存图像
            plt.tight_layout()
            plt.savefig(save_dir / "prediction_last_100.png")
            plt.close()  # 关闭图表而不是显示

        # 绘制误差分布
        plt.figure(figsize=(10, 6))
        errors = pred_inv.flatten() - true_inv.flatten()
        plt.hist(errors, bins=50, alpha=0.7, color='blue')
        plt.title("Prediction Error Distribution")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)

        # 保存图像
        plt.tight_layout()
        plt.savefig(save_dir / "error_distribution.png")
        plt.close()  # 关闭图表而不是显示

        # 绘制散点图
        plt.figure(figsize=(8, 8))
        plt.scatter(true_inv.flatten(), pred_inv.flatten(), alpha=0.5)
        plt.plot([true_inv.min(), true_inv.max()], [true_inv.min(), true_inv.max()], 'r--')
        plt.title("True vs Predicted Scatter Plot")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.grid(True, alpha=0.3)

        # 保存图像
        plt.tight_layout()
        plt.savefig(save_dir / "scatter_plot.png")
        plt.close()  # 关闭图表而不是显示

        print(f"可视化结果已保存至: {save_dir}")


if __name__ == "__main__":
    from config.config_infer import Config  # 确保 config 路径正确

    # 初始化配置
    cnf = Config(conf_file_path=r"D:\PythonProject\chatbot\config\config\CESNET.yaml", exp_name="CESNET")
    inference = Inference(cnf)

    # 运行单步预测（使用固定的编码器时间窗口）
    print("\n=== 运行单步预测 ===")
    print(f"使用编码器时间窗口: {cnf.all_params.get('num_encoder_steps', 48)} 步")

    # 执行单步预测
    pred_inv, true_inv = inference.run_inference_single_step()

    # 如果成功完成预测，可以在这里添加其他分析
    if pred_inv is not None and true_inv is not None:
        print("预测完成 ✅")

        # 计算并打印高估低估样本的比例
        over_estim_count = np.sum(pred_inv > true_inv)
        under_estim_count = np.sum(pred_inv < true_inv)
        total = len(pred_inv)

        print(f"\n📊 预测偏差分析:")
        print(f"  - 高估样本数量: {over_estim_count} ({over_estim_count / total * 100:.1f}%)")
        print(f"  - 低估样本数量: {under_estim_count} ({under_estim_count / total * 100:.1f}%)")
        print(f"  - 准确样本数量: {total - over_estim_count - under_estim_count}")

        # 这里可以添加更多自定义分析...
    else:
        print("预测失败，无法进行后续分析 ❌")