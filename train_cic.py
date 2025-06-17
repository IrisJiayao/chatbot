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
from temporal_fusion_transformer_cic import TemporalFusionTransformer
from progress_bar import ProgressBar
from utils import QuantileLoss, symmetric_mean_absolute_percentage_error, unnormalize_tensor, plot_temporal_serie, ScaleAwareQuantileLoss
from dataset.traffic_data_formatter import TrafficDataFormatter
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
        train_path = r"D:\PythonProject\chatbot\dataset\preprocessed\CSE-CIC-IDS2018_1s\train.csv"
        val_path = r"D:\PythonProject\chatbot\dataset\preprocessed\CSE-CIC-IDS2018_1s\val.csv"
        test_path = r"D:\PythonProject\chatbot\dataset\preprocessed\CSE-CIC-IDS2018_1s\test.csv"

        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Read data
        self.dataset_train = Dataset(train_path)
        # self.dataset_val = Dataset(val_path)
        self.dataset_test = Dataset(test_path)

        self.formatter = TrafficDataFormatter(scaler_path=r"D:\PythonProject\chatbot\dataset\preprocessed\CSE-CIC-IDS2018_1s\scaler.save")


        # Debug whether the data set is correct
        print(f"Training dataset shape: {self.dataset_train.inputs.shape}")
        # print(f"Validation dataset shape: {self.dataset_val.inputs.shape}")
        print(f"Test dataset shape: {self.dataset_test.inputs.shape}")

        # Automatically obtains input_size for TFT
        self.cnf = cnf
        input_dim = self.dataset_train.inputs.shape[-1]
        self.cnf.all_params["input_size"] = input_dim

        self.cnf.all_params["input_obs_loc"] = self.dataset_train.input_obs_loc

        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cuda")
        self.cnf.device = "cuda"
        # self._select_fixed_sample(num_samples=3)

        self.model = TemporalFusionTransformer(self.cnf.all_params).to(device)

        self.loss =StrongTrendQuantileLoss(cnf.quantiles, low_penalty=20.0, trend_weight=1.0, offset_weight=0.3)


        collate_fn = self.cesnet_collate_fn
        # init train loader
        self.train_loader = DataLoader(
            dataset=self.dataset_train, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=True, pin_memory=False, drop_last=True, collate_fn=collate_fn
        )
        # self.val_loader = DataLoader(self.dataset_val, batch_size=cnf.batch_size, shuffle=False,
        #                              num_workers=cnf.n_workers,  pin_memory=True, drop_last=False)  # 确保这里是False)

        # init test loader
        self.test_loader = DataLoader(
            dataset=self.dataset_test, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=False, pin_memory=False,drop_last=False
        )

        # init logging stuffs
        self.log_path = cnf.exp_log_path
        #print(f'tensorboard --logdir={cnf.project_log_path.abspath()}\n')
        print(f'tensorboard --logdir={cnf.project_log_path.resolve()}\n')


        self.optimizer = optim.Adam(params=self.model.parameters(), lr=cnf.lr, weight_decay=1e-5 ) # Add slight regularization
        # Added a learning rate scheduler
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
        self.test_mse=[]
        self.test_r2 = []
        self.test_pearson = []

        self.log_file = self.log_path / "training_log.txt"



        logging.basicConfig(
            filename=self.log_file,
            filemode="a",  # Add-on mode
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )

        # starting values
        self.epoch = 0
        self.best_test_loss = None

        # init progress bar
        self.progress_bar = ProgressBar(max_step=self.log_freq, max_epoch=self.cnf.epochs)

        # possibly load checkpoint
        self.load_ck()

        print("Finished preparing datasets.")

    @staticmethod
    def cesnet_collate_fn(batch):
        # Filter out the None value
        batch = [sample for sample in batch if sample is not None]
        if not batch:
            # If all samples are None, a default empty batch is created
            return {
                'inputs': torch.zeros((0, 96, 41), dtype=torch.float32),
                'outputs': torch.zeros((0, 48, 1), dtype=torch.float32),
                'time_idx': torch.zeros((0, 48, 1), dtype=torch.int64),
                'group_ids': torch.zeros((0, 1), dtype=torch.int64),
            }

        # Normal processing batch
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

    def calculate_regularized_loss(self, y_pred, y_true, attention_components=None):
        """
        Regularization loss is calculated, combined with quantization loss, variance reward and entropy regularization
        """
        quantile_loss, _ = self.loss(y_pred, y_true)

        if y_pred.dim() == 3:  # [batch, time, quantiles]
            p50_pred = y_pred[:, :, 1]
        else:
            p50_pred = y_pred

        pred_variance = torch.var(p50_pred, dim=1).mean()
        var_weight = 0.05
        var_term = -var_weight * torch.log(pred_variance + 1e-6)

        entropy_loss = 0.0
        entropy_weight = 0.1

        if attention_components is not None:
            if 'historical_flags' in attention_components:
                hist_weights = attention_components['historical_flags']
                if hist_weights.dim() > 2:
                    hist_weights = hist_weights.squeeze(2)
                hist_weights = torch.clamp(hist_weights, min=1e-8)
                hist_entropy = -torch.sum(hist_weights * torch.log(hist_weights), dim=-1)
                entropy_loss -= entropy_weight * hist_entropy.mean()

            if 'future_flags' in attention_components:
                future_weights = attention_components['future_flags']
                if future_weights.dim() > 2:
                    future_weights = future_weights.squeeze(2)
                future_weights = torch.clamp(future_weights, min=1e-8)
                future_entropy = -torch.sum(future_weights * torch.log(future_weights), dim=-1)
                entropy_loss -= entropy_weight * future_entropy.mean()

        total_loss = quantile_loss + var_term + entropy_loss

        return total_loss, quantile_loss  # 返回总损失和原始量化损失

    def train(self):

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
                print(f"Training process error: {e}")
                raise

            loss = combined_quantile_trend_mae_loss(output, sample['outputs'].squeeze().float().to("cuda"))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cnf.all_params['max_gradient_norm'])
            self.train_losses.append(loss.item())
            self.optimizer.step()
            self.scheduler.step()

            # print an incredible progress bar
            times.append(time() - t)
            if self.cnf.log_each_step or (not self.cnf.log_each_step and self.progress_bar.progress == 1):
                print(f'\r{self.progress_bar} '
                      f'│ Loss: {np.mean(self.train_losses):.6f} '
                      f'│ ↯: {1 / np.mean(times):5.2f} step/s', end='')
            self.progress_bar.inc()

        # log average loss of this epoch
        mean_epoch_loss = np.mean(self.train_losses)
        logging.info(f"Epoch {self.epoch}: Train Loss: {mean_epoch_loss:.6f}")
        self.sw.add_scalar(tag='train_loss', scalar_value=mean_epoch_loss, global_step=self.epoch)
        self.train_losses = []

        # log epoch duration
        print(f' │ T: {time() - start_time:.2f} s')


    def inverse_transform_single_feature(self, scaler, values, feature_idx=0):
        """
        A single feature is reverse-normalized

        Args:
            scaler: scaler object for inverse normalization
            values: Values to be de-normalized, with the shape (n_samples, 1)
            feature_idx: The index position of the feature in the original scaler. The default value is 0

        Returns:
            The inverse normalized value has the same shape as the input
        """
        #Creates an all-zero set of numbers that matches the expected shape of scaler
        n_features = len(scaler.center_)
        dummy = np.zeros((len(values), n_features))

        # The value to be reversenormalized is placed in the correct feature position
        dummy[:, feature_idx] = values.flatten()

        # Apply inverse normalization
        transformed = scaler.inverse_transform(dummy)

        # Just return the feature that we care about
        return transformed[:, feature_idx].reshape(values.shape)


    def nonlinear_correction(self, values, base=0.2, high_scale=1.4, threshold=0.5):
        corrected = np.copy(values)
        low_mask = values < threshold
        high_mask = values >= threshold

        # Slight boost to low value areas
        corrected[low_mask] = values[low_mask] + base

        # Stronger correction for high value regions
        corrected[high_mask] = base + values[high_mask] * high_scale

        return corrected

    def test(self):
        """
        Test models on test sets and add visualizations
        """
        print(f" test dataset size: {len(self.dataset_test)}")
        print(f" Number of test batches: {len(self.test_loader)}")
        print(f" Batch size: {self.cnf.batch_size}")


        self.model.eval()

        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        all_outputs = []
        all_targets = []
        all_timestamps = []

        self.test_losses = {'p10': [], 'p50': [], 'p90': []}
        self.test_loss = []
        self.test_smape = []
        self.test_mse = []
        self.test_r2 = []
        self.test_pearson = []
        self.filtered_smape = []

        sample_predictions = []
        sample_targets = []
        sample_times = []

        t = time()
        with torch.no_grad():
            for step, sample in enumerate(self.test_loader):
                # Initialize variables for each batch
                target_values = None
                p10_values = None
                p50_values = None
                p90_values = None

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

                if output.dim() == 4:
                    output = output.squeeze(2)
                elif output.dim() == 3:
                    pass
                else:
                    print(f"Warning: Unexpected output dimension: {output.dim()}")

                y = sample['outputs'].to(device)
                y_reshaped = y.squeeze(-1) if y.dim() == 3 else y

                loss = combined_quantile_trend_mae_loss(output, y_reshaped)

                output_np = output.detach().cpu().numpy()
                y_np = y_reshaped.detach().cpu().numpy()

                print(f"output_np shape: {output_np.shape}")
                print(f"y_np shape: {y_np.shape}")

                if 'timestamps' in sample:
                    timestamps = sample['timestamps'].detach().cpu().numpy()
                    all_timestamps.append(timestamps)
                    if step == 0:
                        sample_times = timestamps[0]
                else:
                    batch_size = y_np.shape[0]
                    seq_length = y_np.shape[1] if y_np.ndim > 1 else 1
                    timestamps = np.arange(seq_length)
                    if step == 0:
                        sample_times = timestamps

                if output_np.ndim == 3 and output_np.shape[2] >= 3:
                    p10_values = output_np[:, :, 0]
                    p50_values = output_np[:, :, 1]
                    p90_values = output_np[:, :, 2]
                    if step == 0:
                        for i in range(min(3, output_np.shape[0])):
                            sample_predictions.append({
                                'p10': p10_values[i],
                                'p50': p50_values[i],
                                'p90': p90_values[i]
                            })
                elif output_np.ndim == 2:
                    if output_np.shape[1] >= 3:
                        p10_values = output_np[:, 0]
                        p50_values = output_np[:, 1]
                        p90_values = output_np[:, 2]

                        batch_size = y_np.shape[0]
                        if y_np.ndim > 1:
                            seq_length = y_np.shape[1]
                            if len(p50_values) == batch_size * seq_length:
                                p10_values = p10_values.reshape(batch_size, seq_length)
                                p50_values = p50_values.reshape(batch_size, seq_length)
                                p90_values = p90_values.reshape(batch_size, seq_length)

                        if step == 0:
                            for i in range(min(3, batch_size)):
                                if p50_values.ndim > 1:
                                    sample_predictions.append({
                                        'p10': p10_values[i],
                                        'p50': p50_values[i],
                                        'p90': p90_values[i]
                                    })
                                else:
                                    sample_predictions.append({
                                        'p10': p10_values,
                                        'p50': p50_values,
                                        'p90': p90_values
                                    })
                                    break
                    else:
                        print(f"Warning: Output shape lacks sufficient quantiles: {output_np.shape}")
                        p10_values = p50_values = p90_values = output_np.flatten()
                        if step == 0:
                            sample_predictions.append({
                                'p10': p10_values,
                                'p50': p50_values,
                                'p90': p90_values
                            })
                else:
                    print(f"Warning: Unexpected output shape: {output_np.shape}")
                    p10_values = p50_values = p90_values = output_np.flatten()
                    if step == 0:
                        sample_predictions.append({
                            'p10': p10_values,
                            'p50': p50_values,
                            'p90': p90_values
                        })

                if y_np.ndim == 3:
                    target_values = y_np[:, :, 0]
                    if step == 0:
                        for i in range(min(3, y_np.shape[0])):
                            sample_targets.append(target_values[i])
                elif y_np.ndim == 2:
                    target_values = y_np
                    if step == 0:
                        for i in range(min(3, y_np.shape[0])):
                            sample_targets.append(target_values[i])
                else:
                    target_values = y_np.flatten()
                    if step == 0:
                        sample_targets.append(target_values)

                print(f"p50_values shape: {p50_values.shape}")
                print(f"target_values shape: {target_values.shape}")
                print(f"[DEBUG] p50 mean: {p50_values.mean():.4f}, target mean: {target_values.mean():.4f}")

                if p50_values.shape == target_values.shape:
                    smape = symmetric_mean_absolute_percentage_error(p50_values, target_values)
                else:
                    print(f"Warning: Shape mismatch - p50: {p50_values.shape}, target: {target_values.shape}")
                    p50_flat = p50_values.flatten()
                    target_flat = target_values.flatten()
                    min_length = min(len(p50_flat), len(target_flat))
                    p50_flat = p50_flat[:min_length]
                    target_flat = target_flat[:min_length]
                    smape = symmetric_mean_absolute_percentage_error(p50_flat, target_flat)
                    print(f"Calculated SMAPE from truncated 1D arrays: length={min_length}")

                print(f"True Min: {np.min(target_values)}, Max: {np.max(target_values)}")
                print(f"Pred Min: {np.min(p50_values)}, Max: {np.max(p50_values)}")

                p10 = p10_values.flatten()
                p50 = p50_values.flatten()
                p90 = p90_values.flatten()
                target = target_values.flatten()

                all_outputs.append(p50)
                all_targets.append(target)

                def mse(y_true, y_pred):
                    return np.mean((y_true - y_pred) ** 2)

                safe_max_p50 = np.nanmax(p50) if np.nanmax(p50) > 0 else 1.0
                safe_max_target = np.nanmax(target) if np.nanmax(target) > 0 else 1.0
                mse_val = mse(p50 / safe_max_p50, target / safe_max_target)

                def pearson_corr(y_true, y_pred):
                    y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
                    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
                    if np.std(y_true) == 0 or np.std(y_pred) == 0:
                        return 0.0
                    return np.corrcoef(y_true, y_pred)[0, 1]

                pearson = pearson_corr(p50, target)

                valid_idx = (target > 0)
                if np.sum(valid_idx) > 0:
                    filtered_target = target[valid_idx]
                    filtered_p50 = p50[valid_idx]

                    filtered_smape = symmetric_mean_absolute_percentage_error(filtered_target, filtered_p50)

                    from sklearn.metrics import r2_score
                    filtered_target = np.nan_to_num(filtered_target, nan=0.0)
                    filtered_p50 = np.nan_to_num(filtered_p50, nan=0.0)

                    r2_test = r2_score(filtered_target, filtered_p50)
                else:
                    filtered_smape = 0.0
                    r2_test = 0.0

                self.test_losses['p10'].append(self.loss.numpy_normalised_quantile_loss(p10, target, 0.1))
                self.test_losses['p50'].append(self.loss.numpy_normalised_quantile_loss(p50, target, 0.5))
                self.test_losses['p90'].append(self.loss.numpy_normalised_quantile_loss(p90, target, 0.9))

                self.test_loss.append(loss.item())
                self.test_smape.append(smape)
                self.test_mse.append(mse_val)
                self.test_r2.append(r2_test)
                self.test_pearson.append(pearson)
                self.filtered_smape.append(filtered_smape)
        if (sample_predictions and sample_targets) and (
                self.epoch % 1 == 0 or self.epoch == self.cnf.all_params['epochs'] - 1):
            # self._plot_predictions(sample_predictions, sample_targets, sample_times)
            self._plot_predictions(sample_predictions, sample_targets, sample_times, sample_indices=[0, 1, 2])

        if (self.epoch % 1 == 0 or self.epoch == self.cnf.all_params['epochs'] - 1):
            self._plot_error_distribution(all_outputs, all_targets)

        # Results from all batches were combined for overall correlation analysis
        all_outputs = np.concatenate(all_outputs) if all_outputs else np.array([])
        all_targets = np.concatenate(all_targets) if all_targets else np.array([])

        if (self.epoch % 1 == 0 or self.epoch == self.cnf.all_params['epochs'] - 1):
            self._plot_scatter(all_outputs, all_targets)

        # Quantile loss was recorded
        for k in self.test_losses.keys():
            mean_test_loss = np.mean(self.test_losses[k])
            print(f'\t● AVG {k} Loss on TEST-set: {mean_test_loss:.6f} │ T: {time() - t:.2f} s')
            self.sw.add_scalar(tag=k + '_test_loss', scalar_value=mean_test_loss, global_step=self.epoch)

        # Calculate and record overall indicators
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

        # Record to TensorBoard
        self.sw.add_scalar(tag='test_smape', scalar_value=mean_smape, global_step=self.epoch)
        self.sw.add_scalar(tag='filtered_smape', scalar_value=mean_filtered_smape, global_step=self.epoch)
        self.sw.add_scalar(tag='test_loss', scalar_value=mean_test_loss, global_step=self.epoch)

        # Record to log
        logging.info(
            f"Epoch {self.epoch}: Test Loss: {mean_test_loss:.6f}, SMAPE: {mean_smape:.6f}, filtered_smape: {mean_filtered_smape:.6f}, MSE:{mean_mse:.6f}, R2:{mean_r2:.6f}, PEARSON:{mean_pearson:.6f}")

        # Best preserved model
        if self.best_test_loss is None or mean_test_loss < self.best_test_loss:
            self.best_test_loss = mean_test_loss
            import os
            torch.save(self.model.state_dict(), os.path.join(self.log_path, f"{self.cnf.exp_name}_best.pth"))

    def _plot_predictions(self, predictions, targets, times, sample_indices=None):
        """
        Plot prediction vs actual for multiple samples, supporting specific sample selection.

        Args:
            predictions: List of prediction dicts, each with 'p10', 'p50', and 'p90' quantiles.
            targets: List of actual target values.
            times: Time indices or timestamps.
            sample_indices: List of sample indices to plot. Defaults to [0, 1, 2] if None.
        """
        import matplotlib.pyplot as plt
        import os
        import numpy as np

        # Ensure plot directory exists
        plot_dir = os.path.join(self.log_path, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Default sample indices if not provided
        if sample_indices is None:
            sample_indices = [0, 1, 2]

        # Make sure indices do not exceed available samples
        sample_indices = [idx for idx in sample_indices if idx < len(predictions)]
        num_samples = len(sample_indices)

        # Generate one figure per sample
        for i, sample_idx in enumerate(sample_indices):
            plt.figure(figsize=(12, 4))

            pred = predictions[sample_idx]
            target = targets[sample_idx]

            # Ensure consistent time axis
            x_vals = np.arange(len(target)) if len(times) != len(target) else times

            # Plot actual values
            plt.plot(x_vals, target, 'b-', label='Actual value', linewidth=2)

            # Plot predicted p50 (median forecast)
            plt.plot(x_vals, pred['p50'], 'r-', label='Predicted (p50)', linewidth=2)

            # Shade between p10 and p90 (prediction interval)
            plt.fill_between(x_vals, pred['p10'], pred['p90'],
                             color='r', alpha=0.2, label='10%-90% confidence interval')

            plt.title(f'Sample {sample_idx}: Predicted (p50) vs Actual')
            plt.xlabel('Time step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)

            # Save the figure
            plt.savefig(os.path.join(plot_dir, f'sample_{sample_idx}_prediction_epoch_{self.epoch}.png'))
            plt.close()

            # Add to TensorBoard
            self.sw.add_figure(f'sample_{sample_idx}_prediction', plt.gcf(), self.epoch)

    def _plot_scatter(self, predictions, targets):
        """
        Plot a scatter diagram comparing predictions to actual values.

        Args:
            predictions: Flattened prediction array.
            targets: Flattened actual value array.
        """
        import matplotlib.pyplot as plt
        import os
        import numpy as np
        from sklearn.metrics import r2_score

        # Ensure plot directory exists
        plot_dir = os.path.join(self.log_path, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Remove NaNs and Infs for clean plotting
        valid_idx = ~(np.isnan(predictions) | np.isnan(targets) |
                      np.isinf(predictions) | np.isinf(targets))

        if np.sum(valid_idx) > 0:
            pred_filtered = predictions[valid_idx]
            target_filtered = targets[valid_idx]

            # Compute correlation and R^2
            corr = np.corrcoef(pred_filtered, target_filtered)[0, 1] if len(pred_filtered) > 1 else 0
            r2 = r2_score(target_filtered, pred_filtered)

            plt.figure(figsize=(10, 10))

            # Scatter plot of predicted vs actual
            plt.scatter(target_filtered, pred_filtered, alpha=0.5)

            # Diagonal line for perfect predictions
            min_val = min(np.min(pred_filtered), np.min(target_filtered))
            max_val = max(np.max(pred_filtered), np.max(target_filtered))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')

            plt.title(f'Predicted vs Actual (Corr: {corr:.4f}, R²: {r2:.4f})')
            plt.xlabel('Actual value')
            plt.ylabel('Predicted value')
            plt.grid(True)

            # Save the plot
            plt.savefig(os.path.join(plot_dir, f'scatter_plot_epoch_{self.epoch}.png'))
            plt.close()

            # Log to TensorBoard
            self.sw.add_figure('scatter_plot', plt.gcf(), self.epoch)

    def _plot_error_distribution(self, predictions, targets):
        """
        Plot histograms of absolute and relative prediction errors.

        Args:
            predictions: List of prediction arrays from all batches.
            targets: List of actual value arrays from all batches.
        """
        import matplotlib.pyplot as plt
        import os
        import numpy as np

        # Ensure directory for saving plots exists
        plot_dir = os.path.join(self.log_path, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Concatenate predictions and targets
        all_preds = np.concatenate(predictions) if predictions else np.array([])
        all_targets = np.concatenate(targets) if targets else np.array([])

        if len(all_preds) > 0 and len(all_targets) > 0:
            # Compute absolute errors
            abs_errors = np.abs(all_preds - all_targets)

            # Remove invalid values (NaN or Inf)
            valid_idx = ~(np.isnan(abs_errors) | np.isinf(abs_errors))
            if np.sum(valid_idx) > 0:
                filtered_errors = abs_errors[valid_idx]

                # Plot histogram of absolute errors
                plt.figure(figsize=(10, 6))
                plt.hist(filtered_errors, bins=50, alpha=0.75)
                plt.title('Absolute Error Distribution')
                plt.xlabel('Absolute Error')
                plt.ylabel('Frequency')
                plt.grid(True)

                # Save the figure
                plt.savefig(os.path.join(plot_dir, f'abs_error_hist_epoch_{self.epoch}.png'))
                plt.close()

                # Log to TensorBoard
                self.sw.add_figure('abs_error_hist', plt.gcf(), self.epoch)

                # Compute and plot relative errors for non-zero targets only
                non_zero_targets = (all_targets != 0)
                if np.sum(non_zero_targets) > 0:
                    rel_errors = abs_errors[non_zero_targets] / np.abs(all_targets[non_zero_targets])

                    # Filter valid values
                    valid_rel_idx = ~(np.isnan(rel_errors) | np.isinf(rel_errors))
                    if np.sum(valid_rel_idx) > 0:
                        filtered_rel_errors = rel_errors[valid_rel_idx]

                        # Clip extreme values (e.g., values > 2)
                        filtered_rel_errors = np.clip(filtered_rel_errors, 0, 2)

                        # Plot histogram of relative errors
                        plt.figure(figsize=(10, 6))
                        plt.hist(filtered_rel_errors, bins=50, alpha=0.75)
                        plt.title('Relative Error Distribution')
                        plt.xlabel('Relative Error')
                        plt.ylabel('Frequency')
                        plt.grid(True)

                        # Save the figure
                        plt.savefig(os.path.join(plot_dir, f'rel_error_hist_epoch_{self.epoch}.png'))
                        plt.close()

                        # Log to TensorBoard
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
    from config.config import Config  # Ensure that the path to the config configuration file is correct

    cnf = Config(conf_file_path=r"D:\PythonProject\chatbot\config\config\CSE-CIC-IDS2018.yaml", exp_name="CSE-CIC-IDS2018_1s")
    cnf.device = "cuda"
    # cnf = Config(conf_file_path=r"D:\PythonProject\chatbot\config\config\synthetic_data.yaml", exp_name="synthetic_data")
    trainer = Trainer(cnf)  # Creating a Trainer Instance
    trainer.run()  # Operational training
