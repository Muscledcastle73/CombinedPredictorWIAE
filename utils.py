import torch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def calculate_gradient_penalty(discriminator, real_output, fake_output):
    epsilon = torch.rand((real_output.shape[0], 1, 1)).to(real_output.device)
    interpolates = (epsilon * real_output + (1 - epsilon) * fake_output).requires_grad_(
        True
    )

    interpolate_output = discriminator(interpolates)
    # gradients = torch.autograd.grad(outputs=interpolate_output,inputs=interpolates)

    grad_outputs = torch.ones(interpolate_output.size(), requires_grad=False).to(interpolates.device)
    gradients = torch.autograd.grad(
        outputs=interpolate_output,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean((gradients.norm(p=2, dim=1) - 1) ** 2)
    return gradient_penalty

def fill_nan_inf_2d(arr_2d):
    arr_filled = np.copy(arr_2d)
    for i in range(arr_2d.shape[1]):
        s = pd.Series(arr_2d[:, i])
        s.replace([np.inf, -np.inf], np.nan, inplace=True)
        s = s.fillna(method='ffill').fillna(method='bfill')
        arr_filled[:, i] = s.values
    return arr_filled

def fill_nan_inf_3d(arr_3d):
    arr_filled = np.copy(arr_3d)
    T, S, F = arr_3d.shape
    for i in range(F):
        for j in range(S):
            s = pd.Series(arr_3d[:, j, i])
            s.replace([np.inf, -np.inf], np.nan, inplace=True)
            s = s.fillna(method='ffill').fillna(method='bfill')
            arr_filled[:, j, i] = s.values
    return arr_filled

def extract_percentiles(pred_all):
    percentiles = [5, 25, 50, 75, 95]
    keys = ['P5', 'P25', 'P50', 'P75', 'P95']
    results = {
        key: np.percentile(pred_all, q=p, axis=1)  # result shape: [T, F]
        for key, p in zip(keys, percentiles)
    }

    return results

def metrics(true, pred_mean, pred_median, pred_all, pred_step):
    true = fill_nan_inf_2d(true)
    pred_mean = fill_nan_inf_2d(pred_mean)
    pred_median = fill_nan_inf_2d(pred_median)
    pred_all = fill_nan_inf_3d(pred_all)
    pred_mape = np.zeros_like(true)
    mse = np.zeros(true.shape[1])
    mae = np.zeros(true.shape[1])
    accuracy_mean = np.zeros(true.shape[1])
    accuracy_median = np.zeros(true.shape[1])
    accuracy_mape = np.zeros(true.shape[1])

    for i in range(true.shape[1]):
        y_true = true[:, i]
        y_mean = pred_mean[:, i]
        y_median = pred_median[:, i]

        # Normalization factors
        std = np.std(y_true)
        mean = np.mean(y_true)

        # Mask
        mask = np.abs(y_true - mean) <= 3 * std

        if not np.any(mask):
            print("  All values filtered by mask — skipping computation")
            mse[i] = mae[i] = accuracy_mean[i] = accuracy_median[i] = accuracy_mape[i] = np.nan
            pred_mape[:, i] = np.nan
            continue

        mse[i] = np.mean((y_true[mask] - y_mean[mask]) ** 2)   # For each timestep, take compare the mean to the true value. Take the average of all these calculations
        mae[i] = np.mean(np.abs(y_true[mask] - y_median[mask]))

        if i == 1:
            correct_count = 0
            total_count = 0
            for j in range(len(y_true)):
                preds = pred_all[j, :, i]

                count_low = np.sum(preds < 59.95)
                count_mid = np.sum((preds >= 59.95) & (preds <= 60.05))
                count_high = np.sum(preds > 60.05)
                counts = np.array([count_low, count_mid, count_high])
                max_index = np.argmax(counts)

                true_val = y_true[j]
                if max_index == 0 and true_val < 59.95:
                    correct_count += 1
                elif max_index == 1 and 59.95 <= true_val <= 60.05:
                    correct_count += 1
                elif max_index == 2 and true_val > 60.05:
                    correct_count += 1

                total_count += 1
            accuracy_median[i] = correct_count / total_count if total_count > 0 else np.nan
        else:
            accuracy_median[i] = np.mean(np.sign(y_true[mask]) == np.sign(y_median[mask]))

    percentiles = extract_percentiles(pred_all)

    return mse, mae, pred_mape, accuracy_mean, accuracy_median, accuracy_mape, percentiles

def estimate_coef(x, y):
    n = x.size
    m_x, m_y = x.mean(), y.mean()
    SS_xy = np.sum(y * x) - n*m_y*m_x
    SS_xx = np.sum(x * x) - n*m_x*m_x
    b1 = SS_xy / SS_xx
    b0 = m_y - b1*m_x
    return b0, b1