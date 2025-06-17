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

    def __init__(self, quantiles):
        # takes a list of quantiles
        super().__init__()
        self.quantiles = quantiles
        self.weights = [1, 1, 1]

    def numpy_normalised_quantile_loss(self, y_pred, y, quantile):
        """Computes normalised quantile loss for numpy arrays.
        Uses the q-Risk metric as defined in the "Training Procedure" section of the
        main TFT paper.
        Args:
          y: Targets
          y_pred: Predictions
          quantile: Quantile to use for loss calculations (between 0 & 1)
        Returns:
          Float for normalised quantile loss.
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
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        # print(f" preds.shape: {preds.shape}")
        # print(f" target.shape: {target.shape}")

        # for i, q in enumerate(self.quantiles):
        #     errors = target - preds[:, :, i]
        #     losses.append(
        #         torch.max(
        #             (q - 1) * errors,
        #             q * errors
        #         ).unsqueeze(1))

        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i]
            quantile_loss = torch.max((q - 1) * errors, q * errors).unsqueeze(1)
            losses.append(self.weights[i] * quantile_loss)
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))

        high_values_mask = (target > target.mean() + target.std()).float()
        high_value_errors = torch.abs(target - preds[:, :, 2]) * high_values_mask
        additional_high_value_loss = high_value_errors.mean() * 3.0

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1)) + additional_high_value_loss

        if ret_losses:
            return loss, losses

        return loss


class ScaleAwareQuantileLoss(nn.Module):

    def __init__(self, quantiles, low_penalty=20.0):
        super().__init__()
        self.quantiles = quantiles
        self.low_penalty = low_penalty

    def forward(self, preds, target):
        losses = []

        for i, q in enumerate(self.quantiles):
            if preds.dim() > target.dim():
                current_pred = preds[..., i]
            else:
                current_pred = preds

            errors = target - current_pred
            weighted_errors = torch.where(
                errors > 0,
                errors * q * self.low_penalty,
                errors.abs() * (1 - q)
            )
            losses.append(weighted_errors.mean())

        try:

            if preds.dim() > 2 and target.dim() > 1:

                if preds.shape[-1] >= 2:
                    p50_pred = preds[..., 1]
                else:
                    p50_pred = preds.squeeze(-1)

                if p50_pred.shape[1] > 1 and target.shape[1] > 1:
                    pred_diff = p50_pred[:, 1:] - p50_pred[:, :-1]
                    target_diff = target[:, 1:] - target[:, :-1]

                    min_len = min(pred_diff.shape[1], target_diff.shape[1])
                    trend_loss = torch.mean(torch.abs(
                        pred_diff[:, :min_len] - target_diff[:, :min_len]
                    )) * 0.5

                    base_loss = torch.stack(losses).mean()
                    return base_loss + trend_loss, losses
        except Exception as e:
            print(f"️ Warning: Failed to compute trend loss: {e}")

        return torch.stack(losses).mean(), losses

    def numpy_normalised_quantile_loss(self, predictions, targets, quantile):

        predictions = np.asarray(predictions)
        targets = np.asarray(targets)

        predictions = np.nan_to_num(predictions)
        targets = np.nan_to_num(targets)

        if np.all(targets == 0):
            return 0.0

        errors = targets - predictions

        weighted_errors = np.where(
            errors >= 0,
            quantile * errors * self.low_penalty,
            (quantile - 1) * np.abs(errors)
        )

        normaliser = np.mean(np.abs(targets))
        if normaliser > 0:
            return np.mean(weighted_errors) / normaliser
        else:
            return np.mean(weighted_errors)


# def unnormalize_tensor(data_formatter, data, identifier):
#     data = pd.DataFrame(
#         data.detach().cpu().numpy(),
#         columns=[
#             't+{}'.format(i)
#             for i in range(data.shape[1])
#         ])
#
#     data['identifier'] = np.array(identifier)
#     data = data_formatter.format_predictions(data)
#
#     return data.drop(columns=['identifier']).values

def unnormalize_tensor(formatter, data, identifier):
    if not hasattr(formatter, "format_predictions"):
        raise AttributeError("formatter must have format_predictions() !")

    data = pd.DataFrame(
        data.detach().cpu().numpy(),
        columns=[f't+{i}' for i in range(data.shape[1])]
    )
    data["identifier"] = np.array(identifier)

    return formatter.format_predictions(data).drop(columns=["identifier"]).values


def symmetric_mean_absolute_percentage_error(forecast, actual, epsilon=1e-3):
    # # Symmetric Mean Absolute Percentage Error (SMAPE)
    # sequence_length = forecast.shape[1]
    # sumf = np.sum(np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)), axis=1)
    # return np.mean((2 * sumf) / sequence_length)

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
    """
    Reads the image located in `path`
    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        with PIL.Image.open(f) as img:
            return img.convert('RGB')


def pyplot_to_numpy(pyplot_figure):
    # type: (figure.Figure) -> np.ndarray
    """
    Converts a PyPlot figure into a NumPy array
    :param pyplot_figure: figure you want to convert
    :return: converted NumPy array
    """
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


def combined_quantile_trend_mae_loss(output, target, quantiles=[0.1, 0.5, 0.9],
                                     alpha=1.0, beta=0.3, gamma=0.2,
                                     low_penalty=5.0, epsilon=1e-8):
    quantile_losses = []
    for i, q in enumerate(quantiles):
        errors = target - output[..., i]
        # Penalize underestimation more heavily
        weighted_errors = torch.where(
            errors > 0,  # underestimation
            q * errors * low_penalty,  # increased penalty
            (1 - q) * torch.abs(errors)  # normal penalty for overestimation
        )
        quantile_losses.append(torch.mean(weighted_errors))

    quantile_loss = sum(quantile_losses) / len(quantile_losses)

    # Trend loss using cosine similarity with safety checks
    pred_p50 = output[..., 1]  # Middle quantile (p50)

    # Center the values
    target_centered = target - target.mean(dim=1, keepdim=True)
    pred_centered = pred_p50 - pred_p50.mean(dim=1, keepdim=True)

    # Calculate norms with epsilon for stability
    target_norm = torch.sqrt(torch.sum(target_centered ** 2, dim=1) + epsilon)
    pred_norm = torch.sqrt(torch.sum(pred_centered ** 2, dim=1) + epsilon)

    # Safe cosine similarity calculation
    safe_cosine_sim = torch.sum(target_centered * pred_centered, dim=1) / (target_norm * pred_norm)
    # Clip to valid range [-1, 1]
    safe_cosine_sim = torch.clamp(safe_cosine_sim, -1.0, 1.0)
    trend_loss = torch.mean(1 - safe_cosine_sim)

    # MAE loss
    mae_loss = torch.mean(torch.abs(pred_p50 - target))

    # Combined loss
    total_loss = alpha * quantile_loss + beta * trend_loss + gamma * mae_loss

    return total_loss
