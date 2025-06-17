# -*- coding: utf-8 -*-
# ---------------------
import logging
import random
from time import time
import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.tensorboard import SummaryWriter
from config.config import Config
from dataset.Dataset import Dataset
from temporal_fusion_transformer import TemporalFusionTransformer
from progress_bar import ProgressBar
from utils import QuantileLoss, symmetric_mean_absolute_percentage_error, unnormalize_tensor, plot_temporal_serie, ScaleAwareQuantileLoss
from dataset.traffic_data_formatter import TrafficDataFormatter
from dataset.Dataset_cesnet import CESNETDataset
from utils_modified import combined_quantile_trend_mae_loss
from utils import StrongTrendQuantileLoss


#import data_formatters.utils as utils

class Trainer(object):
    """
    Class for training and test the model
    """

    def __init__(self, cnf):
        # type: (Conf) -> Trainer

        torch.set_num_threads(3)

        # Paths to preprocessed CESNET dataset
        train_path = r"D:\PythonProject\chatbot\dataset\preprocessed\CESNET_100\train.csv"
        val_path = r"D:\PythonProject\chatbot\dataset\preprocessed\CESNET_100\val.csv"
        test_path = r"D:\PythonProject\chatbot\dataset\preprocessed\CESNET_100\test.csv"

        # Set fixed seed for reproducibility
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Load datasets
        self.dataset_train = CESNETDataset(train_path)
        self.dataset_val = CESNETDataset(val_path)
        self.dataset_test = CESNETDataset(test_path)

        # Load formatter for inverse transform
        self.formatter = TrafficDataFormatter(
            scaler_path=r"D:\PythonProject\chatbot\dataset\preprocessed\CESNET_100\scaler.save")

        print(f"Training dataset shape: {self.dataset_train.inputs.shape}")
        print(f"Validation dataset shape: {self.dataset_val.inputs.shape}")
        print(f"Test dataset shape: {self.dataset_test.inputs.shape}")

        self.cnf = cnf
        input_dim = self.dataset_train.inputs.shape[-1]
        self.cnf.all_params["input_size"] = input_dim
        self.cnf.all_params["input_obs_loc"] = self.dataset_train.input_obs_loc

        device = torch.device("cuda")
        self.cnf.device = "cuda"

        self.model = TemporalFusionTransformer(self.cnf.all_params).to(device)

        # Use strong trend quantile loss
        self.loss = StrongTrendQuantileLoss(cnf.quantiles, low_penalty=20.0, trend_weight=1.0, offset_weight=0.3)

        collate_fn = self.cesnet_collate_fn

        self.train_loader = DataLoader(
            dataset=self.dataset_train, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=True, pin_memory=False, drop_last=True, collate_fn=collate_fn
        )
        self.val_loader = DataLoader(
            self.dataset_val, batch_size=cnf.batch_size, shuffle=False,
            num_workers=cnf.n_workers, pin_memory=True, drop_last=False
        )
        self.test_loader = DataLoader(
            dataset=self.dataset_test, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=False, pin_memory=False, drop_last=False
        )

        self.log_path = cnf.exp_log_path
        print(f'tensorboard --logdir={cnf.project_log_path.resolve()}\n')

        # Initialize optimizer and learning rate scheduler
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=cnf.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.002,
            total_steps=self.cnf.epochs * len(self.train_loader),
            pct_start=0.3
        )

        self.sw = SummaryWriter(self.log_path)
        self.log_freq = len(self.train_loader)
        self.train_losses = []
        self.test_loss = []
        self.test_losses = {'p10': [], 'p50': [], 'p90': []}
        self.test_smape = []
        self.filtered_smape = []
        self.test_mse = []
        self.test_r2 = []
        self.test_pearson = []

        self.log_file = self.log_path / "training_log.txt"

        logging.basicConfig(
            filename=self.log_file,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )

        self.epoch = 0
        self.best_test_loss = None

        self.progress_bar = ProgressBar(max_step=self.log_freq, max_epoch=self.cnf.epochs)

        self.load_ck()

        print("Finished preparing datasets.")

    @staticmethod
    def cesnet_collate_fn(batch):
        # Filter out None samples
        batch = [sample for sample in batch if sample is not None]
        if not batch:
            # If all samples are None, return a default empty batch
            return {
                'inputs': torch.zeros((0, 96, 41), dtype=torch.float32),
                'outputs': torch.zeros((0, 48, 1), dtype=torch.float32),
                'time_idx': torch.zeros((0, 48, 1), dtype=torch.int64),
                'group_ids': torch.zeros((0, 1), dtype=torch.int64),
            }

        # Standard collation for valid samples
        inputs = torch.stack([sample['inputs'] for sample in batch])
        outputs = torch.stack([sample['outputs'] for sample in batch])
        time_idx = torch.stack([sample['time_idx'] for sample in batch])
        group_ids = torch.stack([sample['group_ids'] for sample in batch])

        return {
            'inputs': inputs,
            'outputs': outputs,
            'time_idx': time_idx,
            'group_ids': group_ids,
        }

    def load_ck(self):
        """
        load training checkpoint
        """
        ck_path = self.log_path / 'training.ck'
        if ck_path.exists():
            #ck = torch.load(ck_path)
            ck = torch.load(ck_path, map_location="cuda")
            print(f'[loading checkpoint \'{ck_path}\']')
            self.epoch = ck['epoch']
            self.progress_bar.current_epoch = self.epoch
            self.model.load_state_dict(ck['model'])
            self.optimizer.load_state_dict(ck['optimizer'])
            #self.best_test_loss = self.best_test_loss
            self.best_test_loss = ck.get("best_test_loss", None)


    def save_ck(self):
        """
        save training checkpoint
        """

        ck = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_test_loss': self.best_test_loss
        }

        import os
        import glob

        # Ensure log path exists
        os.makedirs(self.log_path, exist_ok=True)

        # Save current epoch checkpoint
        checkpoint_path = os.path.join(self.log_path, f'epoch_{self.epoch}.pth')
        torch.save(ck, checkpoint_path)

        # Manage checkpoint files - keep only the latest 10 epoch checkpoints
        epoch_files = sorted(glob.glob(os.path.join(self.log_path, 'epoch_*.pth')))

        # # Remove oldest checkpoints if more than 10
        # while len(epoch_files) > 10:
        #     oldest_file = epoch_files.pop(0)
        #     os.remove(oldest_file)
        #     print(f"🗑️ Removed old checkpoint: {oldest_file}")

        if self.best_test_loss is None or ck["best_test_loss"] < self.best_test_loss:
            self.best_test_loss = ck["best_test_loss"]
            # torch.save(self.model.state_dict(), self.log_path / self.cnf.exp_name + '_best.pth')
            import os
            torch.save(self.model.state_dict(), os.path.join(self.log_path, f"{self.cnf.exp_name}_best.pth"))

    @staticmethod
    def unnormalize_tensor(target, tensor):
        scale = target.max() - target.min()
        mean = target.mean()
        return tensor * scale + mean

    # Add this method in the Trainer class
    def calculate_regularized_loss(self, y_pred, y_true, attention_components=None):
        """
        Compute regularized loss combining quantile loss, variance encouragement, and entropy regularization.

        Args:
            y_pred (Tensor): Model predictions with quantile outputs [batch, time, quantiles]
            y_true (Tensor): Ground truth values [batch, time]
            attention_components (dict): Optional attention weight outputs from the model

        Returns:
            tuple: (total_loss, quantile_loss_only)
        """
        # 1. Compute base quantile loss
        quantile_loss, _ = self.loss(y_pred, y_true)

        # 2. Add variance encouragement term
        if y_pred.dim() == 3:  # [batch, time, quantiles]
            p50_pred = y_pred[:, :, 1]  # Use p50 (median) prediction
        else:
            p50_pred = y_pred

        # Encourage diversity in predictions by rewarding higher variance
        pred_variance = torch.var(p50_pred, dim=1).mean()
        var_weight = 0.05  # Weight for variance reward
        var_term = -var_weight * torch.log(pred_variance + 1e-6)

        # 3. Add entropy regularization on attention weights
        entropy_loss = 0.0
        entropy_weight = 0.1  # Entropy regularization weight

        if attention_components is not None:
            # Handle historical attention weights
            if 'historical_flags' in attention_components:
                hist_weights = attention_components['historical_flags']
                if hist_weights.dim() > 2:
                    hist_weights = hist_weights.squeeze(2)  # Remove singleton dimension
                hist_weights = torch.clamp(hist_weights, min=1e-8)
                hist_entropy = -torch.sum(hist_weights * torch.log(hist_weights), dim=-1)
                entropy_loss -= entropy_weight * hist_entropy.mean()

            # Handle future attention weights
            if 'future_flags' in attention_components:
                future_weights = attention_components['future_flags']
                if future_weights.dim() > 2:
                    future_weights = future_weights.squeeze(2)
                future_weights = torch.clamp(future_weights, min=1e-8)
                future_entropy = -torch.sum(future_weights * torch.log(future_weights), dim=-1)
                entropy_loss -= entropy_weight * future_entropy.mean()

        # 4. Combine all components
        total_loss = quantile_loss + var_term + entropy_loss

        return total_loss, quantile_loss  # Return total and raw quantile loss

    def train(self):
        """
        Perform one epoch of model training.
        """
        start_time = time()
        self.model.train()

        times = []

        for step, sample in enumerate(self.train_loader):
            t = time()
            self.optimizer.zero_grad()
            try:
                x = sample['inputs'].float().to("cuda")
                output, _, attention_components = self.model.forward(x)
            except Exception as e:
                print(f"Training error: {e}")
                raise
            loss = combined_quantile_trend_mae_loss(output, sample['outputs'].squeeze().float().to("cuda"))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cnf.all_params['max_gradient_norm'])
            self.train_losses.append(loss.item())
            self.optimizer.step()
            self.scheduler.step()

            # Print progress bar
            times.append(time() - t)
            if self.cnf.log_each_step or (not self.cnf.log_each_step and self.progress_bar.progress == 1):
                print(f'\r{self.progress_bar} '
                      f'│ Loss: {np.mean(self.train_losses):.6f} '
                      f'│ ↯: {1 / np.mean(times):5.2f} step/s', end='')
            self.progress_bar.inc()

        # Log average loss for this epoch
        mean_epoch_loss = np.mean(self.train_losses)
        logging.info(f"Epoch {self.epoch}: Train Loss: {mean_epoch_loss:.6f}")
        self.sw.add_scalar(tag='train_loss', scalar_value=mean_epoch_loss, global_step=self.epoch)
        self.train_losses = []

        # Log total time for this epoch
        print(f' │ T: {time() - start_time:.2f} s')

    def inverse_transform_single_feature(self, scaler, values, feature_idx=0):
        """
        Apply inverse transformation to a single feature using a fitted scaler.

        Args:
            scaler: A fitted scaler object with inverse_transform method
            values: The normalized values to inverse transform, shape (n_samples, 1)
            feature_idx: Index of the target feature in the original input space

        Returns:
            numpy.ndarray: The de-normalized values with the same shape as input
        """
        # Create a zero matrix with the same feature dimension as expected by the scaler
        n_features = len(scaler.center_)
        dummy = np.zeros((len(values), n_features))

        # Fill in the target feature at the specified index
        dummy[:, feature_idx] = values.flatten()

        # Apply inverse transform
        transformed = scaler.inverse_transform(dummy)

        # Return only the target feature
        return transformed[:, feature_idx].reshape(values.shape)

    def nonlinear_correction(self, values, base=0.2, high_scale=1.4, threshold=0.5):
        # 注意这里第一个参数是self
        corrected = np.copy(values)
        low_mask = values < threshold
        high_mask = values >= threshold

        # 对低值区域轻微提升
        corrected[low_mask] = values[low_mask] + base

        # 对高值区域更强校正
        corrected[high_mask] = base + values[high_mask] * high_scale

        return corrected

    def test(self):
        """
        Test the model on the test set and generate visualizations.
        """
        # Log test dataset information
        print(f"Test dataset size: {len(self.dataset_test)}")
        print(f"Number of test batches: {len(self.val_loader)}")
        print(f"Batch size: {self.cnf.batch_size}")

        self.model.eval()

        # Import visualization libraries
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for saving plots

        # Initialize containers for evaluation metrics and predictions
        all_outputs = []
        all_targets = []
        all_timestamps = []  # Store timestamps if available

        self.test_losses = {'p10': [], 'p50': [], 'p90': []}
        self.test_loss = []
        self.test_smape = []
        self.test_mse = []
        self.test_r2 = []
        self.test_pearson = []
        self.filtered_smape = []

        # Store predictions and targets for visualization
        sample_predictions = []
        sample_targets = []
        sample_times = []

        t = time()

        with torch.no_grad():
            for step, sample in enumerate(self.val_loader):
                target_values = None
                p10_values = None
                p50_values = None
                p90_values = None
                # 在每个批次处理开始时
                # print(f"处理批次 {step}, 样本数量: {x.shape[0]}")
                if sample is None:
                    print(f"⚠️ Skipping empty batch at step {step}")
                    continue
                device = "cuda"

                self.model = self.model.to(device)

                steps = self.cnf.all_params['num_encoder_steps']
                pred_len = sample['outputs'].shape[1]
                x = sample['inputs'].float().to(device)
                x[:, steps:, 0] = 1

                output, _, attention_components = self.model.forward(x)

                print(f"原始输出形状: {output.shape}")

                # 根据输出的实际维度，适当地进行squeeze
                if output.dim() == 4:  # [batch, time, 1, quantiles]
                    output = output.squeeze(2)
                elif output.dim() == 3:  # [batch, time, quantiles]
                    output = output
                else:
                    print(f"警告：输出维度意外: {output.dim()}")

                print(f"处理后输出形状: {output.shape}")

                # 获取目标值，打印形状
                y = sample['outputs'].to(device)
                print(f"原始目标形状: {y.shape}")

                # 适当调整目标值形状
                if y.dim() == 3:  # [batch, time, 1]
                    y_reshaped = y.squeeze(-1)
                else:
                    y_reshaped = y

                print(f"调整后目标形状: {y_reshaped.shape}")

                # 计算损失
                # loss, _ = combined_quantile_trend_mae_loss(output, y_reshaped)
                loss = combined_quantile_trend_mae_loss(output, y_reshaped)

                # 转换为numpy进行指标计算，打印每个转换后的形状
                output_np = output.detach().cpu().numpy()
                y_np = y_reshaped.detach().cpu().numpy()

                print(f"output_np形状: {output_np.shape}")
                print(f"y_np形状: {y_np.shape}")

                # 如果样本中有时间戳信息，保存它
                if 'timestamps' in sample:
                    timestamps = sample['timestamps'].detach().cpu().numpy()
                    all_timestamps.append(timestamps)
                    # 保存第一个批次的时间戳，用于可视化
                    if step == 0:
                        sample_times = timestamps[0]  # 取第一个样本的时间戳
                else:
                    # 如果没有时间戳，使用序列索引
                    batch_size = y_np.shape[0]
                    seq_length = y_np.shape[1] if y_np.ndim > 1 else 1
                    timestamps = np.arange(seq_length)
                    # 保存第一个批次的时间戳，用于可视化
                    if step == 0:
                        sample_times = timestamps

                # 根据实际维度进行索引，避免维度不匹配错误
                if output_np.ndim == 3 and output_np.shape[2] >= 3:  # [batch, time, quantiles]
                    p10_values = output_np[:, :, 0]
                    p50_values = output_np[:, :, 1]
                    p90_values = output_np[:, :, 2]

                if sample is None:  # 如果batch为空，跳过
                    print(f"⚠️ Skipping empty batch at step {step}")
                    continue

                print(f"处理批次 {step}, 样本数量: {sample['inputs'].shape[0]}")

                # 选择设备 - 此处固定使用cuda
                device = "cuda"  # 或者 "cpu"

                # 确保模型在正确设备上
                self.model = self.model.to(device)

                # 设置输入数据
                steps = self.cnf.all_params['num_encoder_steps']
                pred_len = sample['outputs'].shape[1]
                x = sample['inputs'].float().to(device)
                x[:, steps:, 0] = 1

                # 前向传播
                output, _, attention_components = self.model.forward(x)

                # 打印原始输出形状，帮助调试
                print(f"原始输出形状: {output.shape}")

                # 根据输出的实际维度，适当地进行squeeze
                if output.dim() == 4:  # [batch, time, 1, quantiles]
                    output = output.squeeze(2)
                elif output.dim() == 3:  # [batch, time, quantiles]
                    output = output
                else:
                    print(f"警告：输出维度意外: {output.dim()}")

                print(f"处理后输出形状: {output.shape}")

                # 获取目标值，打印形状
                y = sample['outputs'].to(device)
                print(f"原始目标形状: {y.shape}")

                # 适当调整目标值形状
                if y.dim() == 3:  # [batch, time, 1]
                    y_reshaped = y.squeeze(-1)
                else:
                    y_reshaped = y

                print(f"调整后目标形状: {y_reshaped.shape}")

                # 计算损失
                # loss, _ = self.loss(output, y_reshaped)
                loss = combined_quantile_trend_mae_loss(output, y_reshaped)

                # 转换为numpy进行指标计算，打印每个转换后的形状
                output_np = output.detach().cpu().numpy()
                y_np = y_reshaped.detach().cpu().numpy()

                print(f"output_np形状: {output_np.shape}")
                print(f"y_np形状: {y_np.shape}")

                # 如果样本中有时间戳信息，保存它
                if 'timestamps' in sample:
                    timestamps = sample['timestamps'].detach().cpu().numpy()
                    all_timestamps.append(timestamps)
                    # 保存第一个批次的时间戳，用于可视化
                    if step == 0:
                        sample_times = timestamps[0]  # 取第一个样本的时间戳
                else:
                    # 如果没有时间戳，使用序列索引
                    batch_size = y_np.shape[0]
                    seq_length = y_np.shape[1] if y_np.ndim > 1 else 1
                    timestamps = np.arange(seq_length)
                    # 保存第一个批次的时间戳，用于可视化
                    if step == 0:
                        sample_times = timestamps

                # 根据实际维度进行索引，避免维度不匹配错误
                if output_np.ndim == 3 and output_np.shape[2] >= 3:  # [batch, time, quantiles]
                    p10_values = output_np[:, :, 0]
                    p50_values = output_np[:, :, 1]
                    p90_values = output_np[:, :, 2]

                    # 保存第一个批次的前几个样本，用于可视化
                    if step == 0:
                        for i in range(min(3, output_np.shape[0])):  # 保存最多3个样本
                            sample_predictions.append({
                                'p10': p10_values[i],
                                'p50': p50_values[i],
                                'p90': p90_values[i]
                            })

                elif output_np.ndim == 2:  # [batch, quantiles] 或 [batch*time, quantiles]
                    # 假设这种情况下每行是[p10, p50, p90]
                    if output_np.shape[1] >= 3:
                        p10_values = output_np[:, 0]
                        p50_values = output_np[:, 1]
                        p90_values = output_np[:, 2]

                        # 重塑为[batch, time]格式，如果原本是展平的
                        batch_size = y_np.shape[0]
                        if y_np.ndim > 1:
                            seq_length = y_np.shape[1]
                            if len(p50_values) == batch_size * seq_length:
                                p10_values = p10_values.reshape(batch_size, seq_length)
                                p50_values = p50_values.reshape(batch_size, seq_length)
                                p90_values = p90_values.reshape(batch_size, seq_length)

                        # 保存第一个批次的前几个样本，用于可视化
                        if step == 0:
                            for i in range(min(3, batch_size)):  # 保存最多3个样本
                                if p50_values.ndim > 1:
                                    sample_predictions.append({
                                        'p10': p10_values[i],
                                        'p50': p50_values[i],
                                        'p90': p90_values[i]
                                    })
                                else:
                                    # 如果是1D数组，则整个数组作为一个样本
                                    sample_predictions.append({
                                        'p10': p10_values,
                                        'p50': p50_values,
                                        'p90': p90_values
                                    })
                                    break  # 只保存一个

                    else:
                        print(f"警告: 输出形状不包含足够的分位数: {output_np.shape}")
                        # 假设只有一个分位数
                        p10_values = p50_values = p90_values = output_np.flatten()

                        if step == 0:
                            sample_predictions.append({
                                'p10': p10_values,
                                'p50': p50_values,
                                'p90': p90_values
                            })
                else:
                    print(f"警告: 意外的输出形状: {output_np.shape}")
                    # 使用flatten来应对不明确的形状
                    p10_values = p50_values = p90_values = output_np.flatten()

                    if step == 0:
                        sample_predictions.append({
                            'p10': p10_values,
                            'p50': p50_values,
                            'p90': p90_values
                        })
                # # 应用校正
                # p10_values = self.nonlinear_correction(p10_values)
                # p50_values = self.nonlinear_correction(p50_values)
                # p90_values = self.nonlinear_correction(p90_values)

                # 同样，根据实际维度获取目标值 - 确保每个批次都重新获取
                if y_np.ndim == 3:  # [batch, time, 1]
                    target_values = y_np[:, :, 0]
                    # 保存第一个批次的前几个样本的目标，用于可视化
                    if step == 0:
                        for i in range(min(3, y_np.shape[0])):
                            sample_targets.append(target_values[i])
                elif y_np.ndim == 2:  # [batch, time]
                    target_values = y_np
                    # 保存第一个批次的前几个样本的目标，用于可视化
                    if step == 0:
                        for i in range(min(3, y_np.shape[0])):
                            sample_targets.append(target_values[i])
                else:
                    target_values = y_np.flatten()
                    if step == 0:
                        sample_targets.append(target_values)

                print(f"p50_values形状: {p50_values.shape}")
                print(f"target_values形状: {target_values.shape}")
                print(f"[DEBUG] p50 mean: {p50_values.mean():.4f}, target mean: {target_values.mean():.4f}")
                # 打印原始值范围
                # print(f"应用偏移前 p50 min: {np.min(p50_values):.3f}, max: {np.max(p50_values):.3f}")
                #
                # # 应用偏移
                # offset = 0.4  # 可以根据需要调整
                # p10_values += offset
                # p50_values += offset
                # p90_values += offset

                # # 打印应用偏移后的值范围
                # print(f"应用偏移后 p50 min: {np.min(p50_values):.3f}, max: {np.max(p50_values):.3f}")
                # 计算SMAPE，确保形状一致
                if p50_values.shape == target_values.shape:
                    smape = symmetric_mean_absolute_percentage_error(p50_values, target_values)
                else:
                    print(
                        f"警告: 预测和目标形状不匹配 - p50_values:{p50_values.shape}, target_values:{target_values.shape}")
                    # 尝试重塑到一维并截取相同长度
                    p50_flat = p50_values.flatten()
                    target_flat = target_values.flatten()
                    min_length = min(len(p50_flat), len(target_flat))
                    p50_flat = p50_flat[:min_length]
                    target_flat = target_flat[:min_length]
                    smape = symmetric_mean_absolute_percentage_error(p50_flat, target_flat)
                    print(f"使用截断后的一维数组计算SMAPE: 长度={min_length}")

                print(f"True Min: {np.min(target_values)}, Max: {np.max(target_values)}")
                print(f"Pred Min: {np.min(p50_values)}, Max: {np.max(p50_values)}")

                # 确保所有数据是一维的以进行指标计算
                p10 = p10_values.flatten()
                p50 = p50_values.flatten()
                p90 = p90_values.flatten()
                target = target_values.flatten()

                # 存储当前批次的结果，用于整体指标计算
                all_outputs.append(p50)
                all_targets.append(target)

                # MSE计算
                def mse(y_true, y_pred):
                    return np.mean((y_true - y_pred) ** 2)

                # 如果数据中有极端值，使用安全的最大值
                safe_max_p50 = np.nanmax(p50) if np.nanmax(p50) > 0 else 1.0
                safe_max_target = np.nanmax(target) if np.nanmax(target) > 0 else 1.0

                # 归一化后计算MSE
                mse_val = mse(p50 / safe_max_p50, target / safe_max_target)

                # 计算Pearson相关系数
                def pearson_corr(y_true, y_pred):
                    # 处理NaN和Inf
                    y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
                    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

                    # 检查是否有足够的非零值计算相关性
                    if np.std(y_true) == 0 or np.std(y_pred) == 0:
                        return 0.0

                    return np.corrcoef(y_true, y_pred)[0, 1]

                pearson = pearson_corr(p50, target)

                # 过滤非零值进行评估
                valid_idx = (target > 0)
                if np.sum(valid_idx) > 0:  # 确保有有效值
                    filtered_target = target[valid_idx]
                    filtered_p50 = p50[valid_idx]

                    filtered_smape = symmetric_mean_absolute_percentage_error(filtered_target, filtered_p50)

                    from sklearn.metrics import r2_score
                    # 确保没有NaN或Inf
                    filtered_target = np.nan_to_num(filtered_target, nan=0.0)
                    filtered_p50 = np.nan_to_num(filtered_p50, nan=0.0)

                    r2_test = r2_score(filtered_target, filtered_p50)
                else:
                    filtered_smape = 0.0
                    r2_test = 0.0

                # 计算分位数损失
                self.test_losses['p10'].append(self.loss.numpy_normalised_quantile_loss(p10, target, 0.1))
                self.test_losses['p50'].append(self.loss.numpy_normalised_quantile_loss(p50, target, 0.5))
                self.test_losses['p90'].append(self.loss.numpy_normalised_quantile_loss(p90, target, 0.9))

                # 保存其他指标
                self.test_loss.append(loss.item())
                self.test_smape.append(smape)
                self.test_mse.append(mse_val)
                self.test_r2.append(r2_test)
                self.test_pearson.append(pearson)
                self.filtered_smape.append(filtered_smape)

                # 如果是第一个批次，生成注意力热图
                if step == 0 and attention_components is not None:
                    self._plot_attention_maps(attention_components)

            # 每10个epoch或最后一个epoch时创建预测可视化图表
        if (sample_predictions and sample_targets) and (
                self.epoch % 1 == 0 or self.epoch == self.cnf.all_params['epochs'] - 1):
            # self._plot_predictions(sample_predictions, sample_targets, sample_times)
            self._plot_predictions(sample_predictions, sample_targets, sample_times, sample_indices=[0, 1, 2])

            # 每10个epoch或最后一个epoch时创建整体误差分布图
        if (self.epoch % 1 == 0 or self.epoch == self.cnf.all_params['epochs'] - 1):
            self._plot_error_distribution(all_outputs, all_targets)

        # 合并所有批次的结果用于整体相关性分析
        all_outputs = np.concatenate(all_outputs) if all_outputs else np.array([])
        all_targets = np.concatenate(all_targets) if all_targets else np.array([])

        # 创建预测vs实际值散点图
        # 每10个epoch或最后一个epoch时创建预测vs实际值散点图
        if (self.epoch % 1 == 0 or self.epoch == self.cnf.all_params['epochs'] - 1):
            self._plot_scatter(all_outputs, all_targets)

        # 记录分位数损失
        for k in self.test_losses.keys():
            mean_test_loss = np.mean(self.test_losses[k])
            print(f'\t● AVG {k} Loss on TEST-set: {mean_test_loss:.6f} │ T: {time() - t:.2f} s')
            self.sw.add_scalar(tag=k + '_test_loss', scalar_value=mean_test_loss, global_step=self.epoch)

        # 计算并记录整体指标
        mean_test_loss = np.mean(self.test_loss)
        mean_smape = np.mean(self.test_smape)
        mean_mse = np.mean(self.test_mse)
        mean_r2 = np.mean(self.test_r2)
        mean_pearson = np.mean(self.test_pearson)
        mean_filtered_smape = np.mean(self.filtered_smape)

        print(f'\t● AVG Loss on TEST-set: {mean_test_loss:.6f} │ T: {time() - t:.2f} s')
        print(f'\t● AVG SMAPE on TEST-set: {mean_smape:.6f} │ T: {time() - t:.2f} s')
        print(f'\t● AVG MSE on TEST-set: {mean_mse:.6f} │ T: {time() - t:.2f} s')
        print(f'\t● AVG r2_score on TEST-set: {mean_r2:.6f} │ T: {time() - t:.2f} s')
        print(f'\t● AVG PEARSON on TEST-set: {mean_pearson:.6f} │ T: {time() - t:.2f} s')
        print(f'\t● FILTERED SMAPE on TEST-set: {mean_filtered_smape:.6f} │ T: {time() - t:.2f} s')

        # 记录到TensorBoard
        self.sw.add_scalar(tag='test_smape', scalar_value=mean_smape, global_step=self.epoch)
        self.sw.add_scalar(tag='filtered_smape', scalar_value=mean_filtered_smape, global_step=self.epoch)
        self.sw.add_scalar(tag='test_loss', scalar_value=mean_test_loss, global_step=self.epoch)

        # 记录到日志
        logging.info(
            f"Epoch {self.epoch}: Test Loss: {mean_test_loss:.6f}, SMAPE: {mean_smape:.6f}, filtered_smape: {mean_filtered_smape:.6f}, MSE:{mean_mse:.6f}, R2:{mean_r2:.6f}, PEARSON:{mean_pearson:.6f}")

        # 保存最佳模型
        if self.best_test_loss is None or mean_test_loss < self.best_test_loss:
            self.best_test_loss = mean_test_loss
            import os
            torch.save(self.model.state_dict(), os.path.join(self.log_path, f"{self.cnf.exp_name}_best.pth"))

    def _plot_predictions(self, predictions, targets, times, sample_indices=None):
        """
        绘制多个预测结果与真实值的对比图，支持指定特定样本索引

        Args:
            predictions: 预测结果列表，每个元素包含p10, p50, p90三个分位数
            targets: 真实值列表
            times: 时间戳或索引
            sample_indices: 要绘制的样本索引列表，默认为None
        """
        import matplotlib.pyplot as plt
        import os
        import numpy as np

        # 确保保存图表的目录存在
        plot_dir = os.path.join(self.log_path, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # 如果没有指定sample_indices，使用0, 1, 2
        if sample_indices is None:
            sample_indices = [0, 1, 2]

        # 限制样本数量不超过可用样本
        sample_indices = [idx for idx in sample_indices if idx < len(predictions)]
        num_samples = len(sample_indices)

        # 为每个样本创建单独的图表
        for i, sample_idx in enumerate(sample_indices):
            plt.figure(figsize=(12, 4))

            pred = predictions[sample_idx]
            target = targets[sample_idx]

            # 确保所有数据维度一致
            x_vals = np.arange(len(target)) if len(times) != len(target) else times

            # 绘制实际值
            plt.plot(x_vals, target, 'b-', label='actual value', linewidth=2)

            # 绘制p50预测值（中位数）
            plt.plot(x_vals, pred['p50'], 'r-', label='pred(p50)', linewidth=2)

            # 创建p10-p90的置信区间
            plt.fill_between(x_vals, pred['p10'], pred['p90'],
                             color='r', alpha=0.2, label='10%-90% confidence')

            plt.title(f'Sample {sample_idx}: pred(p50) vs actual value')
            plt.xlabel('time step')
            plt.ylabel('value')
            plt.legend()
            plt.grid(True)

            # 保存每个样本的图表
            plt.savefig(os.path.join(plot_dir, f'sample_{sample_idx}_prediction_epoch_{self.epoch}.png'))
            plt.close()

            # 将图片添加到TensorBoard
            self.sw.add_figure(f'sample_{sample_idx}_prediction', plt.gcf(), self.epoch)

    def _plot_attention_maps(self, attention_components):
        """
        绘制注意力热图

        Args:
            attention_components: 包含注意力权重的字典
        """
        import matplotlib.pyplot as plt
        import os
        import numpy as np
        import seaborn as sns

        # 确保保存图表的目录存在
        plot_dir = os.path.join(self.log_path, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # 处理不同类型的注意力组件
        if 'decoder_self_attn' in attention_components:
            # 获取第一个样本的自注意力矩阵
            attn = attention_components['decoder_self_attn'][0].detach().cpu().numpy()

            # 对于多头注意力，取平均值
            if attn.ndim > 2:
                attn = np.mean(attn, axis=0)

            plt.figure(figsize=(10, 8))
            sns.heatmap(attn, cmap='viridis')
            plt.title('Decoder self-attention heat map')
            plt.xlabel('Key')
            plt.ylabel('value')

            # 保存图表
            plt.savefig(os.path.join(plot_dir, f'decoder_attention_epoch_{self.epoch}.png'))
            plt.close()

            # 将图片添加到TensorBoard
            self.sw.add_figure('decoder_attention', plt.gcf(), self.epoch)

        # 绘制变量选择权重 - 处理多维权重
        for weight_name in ['historical_flags', 'future_flags', 'static_flags']:
            if weight_name in attention_components:
                weights = attention_components[weight_name][0].detach().cpu().numpy()

                # 检查权重的维度并适当处理
                print(f"{weight_name} weight_shape: {weights.shape}")

                if weights.ndim == 1:
                    # 一维权重直接绘制条形图
                    plt.figure(figsize=(12, 4))
                    plt.bar(range(len(weights)), weights)
                    plt.title(f'{weight_name} Variable selection weight')
                    plt.xlabel('index')
                    plt.ylabel('weight')

                    # 保存图表
                    plt.savefig(os.path.join(plot_dir, f'{weight_name}_epoch_{self.epoch}.png'))
                    plt.close()

                    # 将图片添加到TensorBoard
                    self.sw.add_figure(weight_name, plt.gcf(), self.epoch)

                elif weights.ndim == 2:
                    # 二维权重绘制热图
                    plt.figure(figsize=(12, 8))
                    sns.heatmap(weights, cmap='viridis')
                    plt.title(f'{weight_name} Variable selection weight')

                    if weight_name == 'historical_flags':
                        plt.xlabel('feature')
                        plt.ylabel('time step')
                    elif weight_name == 'future_flags':
                        plt.xlabel('feature')
                        plt.ylabel('pred time step')

                    # 保存图表
                    plt.savefig(os.path.join(plot_dir, f'{weight_name}_heatmap_epoch_{self.epoch}.png'))
                    plt.close()

                    # 将图片添加到TensorBoard
                    self.sw.add_figure(f'{weight_name}_heatmap', plt.gcf(), self.epoch)

                    # 同时绘制按时间步平均的权重
                    mean_weights = np.mean(weights, axis=0)
                    plt.figure(figsize=(12, 4))
                    plt.bar(range(len(mean_weights)), mean_weights)
                    plt.title(f'{weight_name} Average variable selection weights')
                    plt.xlabel('feature index')
                    plt.ylabel('average weight')

                    # 保存图表
                    plt.savefig(os.path.join(plot_dir, f'{weight_name}_avg_epoch_{self.epoch}.png'))
                    plt.close()

                    # 将图片添加到TensorBoard
                    self.sw.add_figure(f'{weight_name}_avg', plt.gcf(), self.epoch)

                    # 绘制每个时间步的权重分布
                    plt.figure(figsize=(15, 10))
                    for i in range(min(weights.shape[0], 10)):  # 最多显示10个时间步
                        plt.subplot(5, 2, i + 1)
                        plt.bar(range(weights.shape[1]), weights[i])
                        plt.title(f'time step {i}')
                        if i >= 8:  # 只在底部两个子图显示x轴标签
                            plt.xlabel('feature index')
                        plt.ylabel('weight')

                    plt.tight_layout()
                    # 保存图表
                    plt.savefig(os.path.join(plot_dir, f'{weight_name}_timesteps_epoch_{self.epoch}.png'))
                    plt.close()

                    # 将图片添加到TensorBoard
                    self.sw.add_figure(f'{weight_name}_timesteps', plt.gcf(), self.epoch)

    def _plot_scatter(self, predictions, targets):
        """
        绘制预测值与实际值的散点图

        Args:
            predictions: 所有预测值（1D数组）
            targets: 所有实际值（1D数组）
        """
        import matplotlib.pyplot as plt
        import os
        import numpy as np
        from sklearn.metrics import r2_score

        # 确保保存图表的目录存在
        plot_dir = os.path.join(self.log_path, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # 过滤掉NaN和Inf值
        valid_idx = ~(np.isnan(predictions) | np.isnan(targets) |
                      np.isinf(predictions) | np.isinf(targets))

        if np.sum(valid_idx) > 0:
            pred_filtered = predictions[valid_idx]
            target_filtered = targets[valid_idx]

            # 计算相关性系数和R²
            corr = np.corrcoef(pred_filtered, target_filtered)[0, 1] if len(pred_filtered) > 1 else 0
            r2 = r2_score(target_filtered, pred_filtered)

            plt.figure(figsize=(10, 10))

            # 绘制散点图
            plt.scatter(target_filtered, pred_filtered, alpha=0.5)

            # 添加对角线（完美预测线）
            min_val = min(np.min(pred_filtered), np.min(target_filtered))
            max_val = max(np.max(pred_filtered), np.max(target_filtered))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')

            plt.title(f'pred vs actual (correlation coefficient: {corr:.4f}, R²: {r2:.4f})')
            plt.xlabel('actual value')
            plt.ylabel('pred value')
            plt.grid(True)

            # 保存图表
            plt.savefig(os.path.join(plot_dir, f'scatter_plot_epoch_{self.epoch}.png'))
            plt.close()

            # 将图片添加到TensorBoard
            self.sw.add_figure('scatter_plot', plt.gcf(), self.epoch)

    def _plot_error_distribution(self, predictions, targets):
        """
        绘制误差分布直方图

        Args:
            predictions: 所有预测值列表
            targets: 所有实际值列表
        """
        import matplotlib.pyplot as plt
        import os
        import numpy as np

        # 确保保存图表的目录存在
        plot_dir = os.path.join(self.log_path, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # 合并所有批次的结果
        all_preds = np.concatenate(predictions) if predictions else np.array([])
        all_targets = np.concatenate(targets) if targets else np.array([])

        if len(all_preds) > 0 and len(all_targets) > 0:
            # 计算绝对误差和相对误差
            abs_errors = np.abs(all_preds - all_targets)

            # 过滤掉无效值
            valid_idx = ~(np.isnan(abs_errors) | np.isinf(abs_errors))
            if np.sum(valid_idx) > 0:
                filtered_errors = abs_errors[valid_idx]

                # 绘制绝对误差直方图
                plt.figure(figsize=(10, 6))
                plt.hist(filtered_errors, bins=50, alpha=0.75)
                plt.title('Absolute error distribution')
                plt.xlabel('Absolute error')
                plt.ylabel('freq')
                plt.grid(True)

                # 保存图表
                plt.savefig(os.path.join(plot_dir, f'abs_error_hist_epoch_{self.epoch}.png'))
                plt.close()

                # 将图片添加到TensorBoard
                self.sw.add_figure('abs_error_hist', plt.gcf(), self.epoch)

                # 计算并绘制相对误差（仅对非零目标值）
                non_zero_targets = (all_targets != 0)
                if np.sum(non_zero_targets) > 0:
                    rel_errors = abs_errors[non_zero_targets] / np.abs(all_targets[non_zero_targets])

                    # 过滤掉无效值
                    valid_rel_idx = ~(np.isnan(rel_errors) | np.isinf(rel_errors))
                    if np.sum(valid_rel_idx) > 0:
                        filtered_rel_errors = rel_errors[valid_rel_idx]

                        # 截断极端值，限制在合理范围内，如0-2或更窄
                        filtered_rel_errors = np.clip(filtered_rel_errors, 0, 2)

                        plt.figure(figsize=(10, 6))
                        plt.hist(filtered_rel_errors, bins=50, alpha=0.75)
                        plt.title('Relative error distribution')
                        plt.xlabel('Relative error')
                        plt.ylabel('freq')
                        plt.grid(True)

                        # 保存图表
                        plt.savefig(os.path.join(plot_dir, f'rel_error_hist_epoch_{self.epoch}.png'))
                        plt.close()

                        # 将图片添加到TensorBoard
                        self.sw.add_figure('rel_error_hist', plt.gcf(), self.epoch)

    def run(self):
        """
        start model training procedure (train > test > checkpoint > repeat)
        """
        for _ in range(self.epoch, self.cnf.epochs):
            self.train()

            with torch.no_grad():
                self.test()

            self.epoch += 1
            self.save_ck()

if __name__ == "__main__":
    from config.config import Config  # 确保 config 配置文件路径正确

    # cnf = Config()  # 初始化配置

    cnf = Config(conf_file_path=r"D:\PythonProject\chatbot\config\config\CESNET.yaml", exp_name="CESNET_100")
    cnf.device = "cuda"
    # cnf = Config(conf_file_path=r"D:\PythonProject\chatbot\config\config\synthetic_data.yaml", exp_name="synthetic_data")
    trainer = Trainer(cnf)  # 创建 Trainer 实例
    trainer.run()  # 运行训练
