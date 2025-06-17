# -*- coding: utf-8 -*-
# ---------------------

import json
import os
from datetime import datetime
from enum import Enum
from typing import *
from typing import Callable, List, TypeVar

import PIL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL.Image import Image
from matplotlib import cm
from matplotlib import figure
from pathlib import Path
from torch import Tensor
from torch import nn
from torchvision.transforms import ToTensor


class QuantileLoss(nn.Module):
    """
    Quantile loss module for multiple quantiles.
    Includes optional normalized quantile loss for evaluation,
    and high-value penalty for extreme under-predictions.
    """

    def __init__(self, quantiles):
        """
        Args:
            quantiles (list of float): Target quantiles (e.g., [0.1, 0.5, 0.9])
        """
        super().__init__()
        self.quantiles = quantiles
        self.weights = [1, 1, 1]  # Optional: different weights per quantile

    def numpy_normalised_quantile_loss(self, y_pred, y, quantile):
        """
        Compute the normalized quantile loss as in the TFT paper (q-Risk).
        Args:
            y_pred (np.ndarray or torch.Tensor): Predictions, shape [batch, time, quantiles] or [batch, time]
            y (np.ndarray or torch.Tensor): Ground truth targets
            quantile (float): The quantile level (between 0 and 1)
        Returns:
            float: Normalized quantile loss
        """
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if len(y_pred.shape) == 3:
            ix = self.quantiles.index(quantile)
            y_pred = y_pred[..., ix]
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        prediction_underflow = y - y_pred
        weighted_errors = quantile * np.maximum(prediction_underflow, 0.) \
                          + (1. - quantile) * np.maximum(-prediction_underflow, 0.)

        quantile_loss = weighted_errors.mean()
        normaliser = np.abs(y).mean()

        return 2 * quantile_loss / normaliser

    def forward(self, preds, target, ret_losses=True):
        """
        Compute multi-quantile loss, optionally with high-value penalties.
        Args:
            preds: [batch, time, num_quantiles]
            target: [batch, time]
            ret_losses: Whether to return individual losses per quantile
        Returns:
            loss: Combined scalar loss
            losses (optional): List of quantile-wise losses
        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i]  # prediction error for quantile q
            quantile_loss = torch.max((q - 1) * errors, q * errors).unsqueeze(1)
            losses.append(self.weights[i] * quantile_loss)

        # base_loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        # Add extra penalty for underestimating high target values
        high_values_mask = (target > target.mean() + target.std()).float()
        high_value_errors = torch.abs(target - preds[:, :, 2]) * high_values_mask  # p90 predictions
        additional_high_value_loss = high_value_errors.mean() * 3.0  # penalty scale

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1)) + additional_high_value_loss

        if ret_losses:
            return loss, losses
        return loss


class ScaleAwareQuantileLoss(nn.Module):
    """
    Enhanced quantile loss that:
    - Penalizes under-predictions more strongly via a configurable multiplier.
    - Optionally includes a trend alignment penalty between predicted and true differences over time.
    """

    def __init__(self, quantiles, low_penalty=20.0):

        super().__init__()
        self.quantiles = quantiles
        self.low_penalty = low_penalty

    def forward(self, preds, target):

        losses = []

        # Compute quantile loss for each quantile
        for i, q in enumerate(self.quantiles):
            current_pred = preds[..., i] if preds.dim() > target.dim() else preds
            errors = target - current_pred

            # Apply asymmetric penalty: stronger for underestimation
            weighted_errors = torch.where(
                errors > 0,
                errors * q * self.low_penalty,  # amplify under-prediction penalty
                errors.abs() * (1 - q)
            )
            losses.append(weighted_errors.mean())

        # Try adding trend alignment loss using p50 predictions
        try:
            if preds.dim() > 2 and target.dim() > 1:
                if preds.shape[-1] >= 2:
                    p50_pred = preds[..., 1]  # Use quantile at index 1 (assumed to be 0.5)
                else:
                    p50_pred = preds.squeeze(-1)  # fallback if only one quantile

                # Compute time differences (delta_t)
                if p50_pred.shape[1] > 1 and target.shape[1] > 1:
                    pred_diff = p50_pred[:, 1:] - p50_pred[:, :-1]
                    target_diff = target[:, 1:] - target[:, :-1]

                    # Align and compare trend deltas
                    min_len = min(pred_diff.shape[1], target_diff.shape[1])
                    trend_loss = torch.mean(torch.abs(
                        pred_diff[:, :min_len] - target_diff[:, :min_len]
                    )) * 0.5  # trend penalty scale

                    base_loss = torch.stack(losses).mean()
                    return base_loss + trend_loss, losses
        except Exception as e:
            print(f" Warning: Trend loss computation failed: {e}")
            # Ignore trend loss if computation fails

        # If trend loss was not added, return base quantile loss
        return torch.stack(losses).mean(), losses



class StrongTrendQuantileLoss(nn.Module):
    """
    A customized quantile loss function with:
    - Amplified penalty for under-prediction
    - Trend alignment loss (difference between predicted and actual trend)
    - Mean offset loss (encourages predictions to align with the mean target level)
    """

    def __init__(self, quantiles, low_penalty=15.0, trend_weight=3.0, offset_weight=0.3):
        """
        Args:
            quantiles (list of float): Target quantile levels (e.g., [0.1, 0.5, 0.9])
            low_penalty (float): Penalty multiplier for under-prediction
            trend_weight (float): Weight for trend loss term
            offset_weight (float): Weight for mean alignment penalty
        """
        super().__init__()
        self.quantiles = quantiles
        self.low_penalty = low_penalty
        self.trend_weight = trend_weight
        self.offset_weight = offset_weight

    def forward(self, preds, target):
        """
        Compute the full quantile loss.
        Args:
            preds: Tensor of shape [batch, time, num_quantiles]
            target: Tensor of shape [batch, time]
        Returns:
            total_loss: Scalar loss combining quantile, trend, and offset components
            losses: List of individual quantile losses
        """
        losses = []

        # 1. Base quantile loss with underestimation penalty
        for i, q in enumerate(self.quantiles):
            pred_q = preds[..., i] if preds.dim() > target.dim() else preds
            error = target - pred_q
            weighted_error = torch.where(
                error > 0,
                error * q * self.low_penalty,    # amplified penalty for under-prediction
                error.abs() * (1 - q)
            )
            losses.append(weighted_error.mean())

        base_loss = torch.stack(losses).mean()

        # 2. Trend alignment loss between p50 prediction and target
        if preds.shape[-1] >= 2:
            p50 = preds[..., 1]  # assumes index 1 corresponds to median (0.5)
        else:
            p50 = preds.squeeze(-1)

        if p50.shape[1] > 1 and target.shape[1] > 1:
            pred_diff = p50[:, 1:] - p50[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            trend_loss = torch.mean(torch.abs(pred_diff - target_diff))
        else:
            trend_loss = torch.tensor(0.0).to(preds.device)

        # 3. Mean offset loss: encourage predictions to match target scale
        mean_diff = torch.abs(p50.mean() - target.mean())

        # Final combined loss
        total_loss = base_loss + self.trend_weight * trend_loss + self.offset_weight * mean_diff

        return total_loss, losses

    def numpy_normalised_quantile_loss(self, predictions, targets, quantile):
        """
        Compute normalized quantile loss using NumPy (used for evaluation)
        Args:
            predictions: np.ndarray, shape [..., time]
            targets: np.ndarray, same shape as predictions
            quantile: float between 0 and 1
        Returns:
            float: Normalized quantile loss
        """
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)

        # Remove NaNs safely
        predictions = np.nan_to_num(predictions)
        targets = np.nan_to_num(targets)

        # Return 0 if all targets are zero (to avoid divide by zero)
        if np.all(targets == 0):
            return 0.0

        errors = targets - predictions

        # Apply quantile loss
        weighted_errors = np.where(
            errors >= 0,
            quantile * errors * self.low_penalty,  # amplified under-prediction penalty
            (quantile - 1) * np.abs(errors)
        )

        # Normalize by mean absolute target value
        normaliser = np.mean(np.abs(targets))
        if normaliser > 0:
            return np.mean(weighted_errors) / normaliser
        else:
            return np.mean(weighted_errors)


def unnormalize_tensor(formatter, data, identifier):

    if not hasattr(formatter, "format_predictions"):
        raise AttributeError("The formatter must have a `format_predictions()` method.")

    data = pd.DataFrame(
        data.detach().cpu().numpy(),
        columns=[f't+{i}' for i in range(data.shape[1])]
    )
    data["identifier"] = np.array(identifier)

    return formatter.format_predictions(data).drop(columns=["identifier"]).values








def symmetric_mean_absolute_percentage_error(forecast, actual, epsilon=1e-3):

    numerator = 2 * np.abs(forecast - actual)
    denominator = np.abs(forecast) + np.abs(actual) + epsilon
    return np.mean(numerator / denominator)



def plot_temporal_serie(y_pred, y_true):
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    if isinstance(y_true, Tensor):
        y_true = y_true.detach().cpu().numpy()

    ind = np.random.choice(y_pred.shape[0])
    plt.plot(y_pred[ind, :, 0], label='pred_1')
    plt.plot(y_pred[ind, :, 1], label='pred_5')
    plt.plot(y_pred[ind, :, 2], label='pred_9')

    plt.plot(y_true[ind, :, 0], label='true')
    plt.legend()
    plt.show()


def imread(path):
    # type: (Union[Path, str]) -> Image

    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')


def pyplot_to_numpy(pyplot_figure):
    # type: (figure.Figure) -> np.ndarray

    pyplot_figure.canvas.draw()
    x = np.fromstring(pyplot_figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    x = x.reshape(pyplot_figure.canvas.get_width_height()[::-1] + (3,))
    return x


def pyplot_to_tensor(pyplot_figure):
    # type: (figure.Figure) -> Tensor
    """
    Converts a PyPlot figure into a PyTorch tensor
    :param pyplot_figure: figure you want to convert
    :return: converted PyTorch tensor
    """
    x = pyplot_to_numpy(pyplot_figure=pyplot_figure)
    x = ToTensor()(x)
    return x


def apply_colormap_to_tensor(x, cmap='jet', range=(None, None)):
    # type: (Tensor, str, Optional[Tuple[float, float]]) -> Tensor
    """
    :param x: Tensor with shape (1, H, W)
    :param cmap: name of the color map you want to apply
    :param range: tuple of (minimum possible value in x, maximum possible value in x)
    :return: Tensor with shape (3, H, W)
    """
    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(vmin=range[0], vmax=range[1])
    x = x.detatch().cpu().numpy()
    x = x.squeeze()
    x = cmap.to_rgba(x)[:, :, :-1]
    return ToTensor()(x)
