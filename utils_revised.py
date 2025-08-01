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

import numpy as np

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
    #
    # print(true)
    # print(true.shape)
    # print(pred_mean)
    # print(pred_mean.shape)
    # print(pred_median)
    # print(pred_median.shape)
    # print(pred_all)
    # print(pred_all.shape)

    for i in range(true.shape[1]):
        y_true = true[:, i]
        y_mean = pred_mean[:, i]
        y_median = pred_median[:, i]
        y_ens = pred_all[:, :, i]

        # Normalization factors
        std = np.std(y_true)
        mean = np.mean(y_true)

        # Mask
        mask = np.abs(y_true-mean) <= 3 * std

        if not np.any(mask):
            print("  All values filtered by mask — skipping computation")
            mse[i] = mae[i] = accuracy_mean[i] = accuracy_median[i] = accuracy_mape[i] = np.nan
            pred_mape[:, i] = np.nan
            continue

        mse[i] = np.mean((y_true[mask] - y_mean[mask]) ** 2)
        mae[i] = np.mean(np.abs(y_true[mask] - y_median[mask]))
        mae_values = np.abs(y_true[mask] - y_median[mask])

        # ... existing arrays above ...
        nmse = np.zeros(true.shape[1])  # ← add
        nmae = np.zeros(true.shape[1])  # ← add
        crps_scores = np.zeros(true.shape[1])

        # ── normalised errors (paper definition) ───────────────────
        denom_mse = np.mean(y_true[mask] ** 2)  # ⟨x_t²⟩
        denom_mae = np.mean(np.abs(y_true[mask]))  # ⟨|x_t|⟩
        nmse[i] = mse[i] / denom_mse  # NMSE
        nmae[i] = mae[i] / denom_mae  # NMAE
        # ───────────────────────────────────────────────────────────
        print("NMSE and NMAE")
        print(nmse)
        print(nmae)
        crps_scores[i] = crps(y_true[mask], y_ens[mask, :])
        print("CRPS")
        print(crps_scores)

        if i == 1:
            correct_count = 0
            total_count = 0
            correct_count_narrow = 0
            for j in range(len(y_true)):
                if not mask[j]:
                    continue

                preds = pred_all[j, :, i]

                count_low = np.sum(preds < 59.95)
                count_mid = np.sum((preds >= 59.95) & (preds <= 60.05))
                count_high = np.sum(preds > 60.05)
                counts = np.array([count_low, count_mid, count_high])
                max_index = np.argmax(counts)

                count_low_narrow = np.sum(preds < 59.964)
                count_mid_narrow = np.sum((preds >= 59.964) & (preds <= 60.036))
                count_high_narrow = np.sum(preds > 60.036)
                counts_narrow = np.array([count_low_narrow, count_mid_narrow, count_high_narrow])
                max_index_narrow = np.argmax(counts_narrow)

                true_val = y_true[j]
                if max_index == 0 and true_val < 59.95:
                    correct_count += 1
                elif max_index == 1 and 59.95 <= true_val <= 60.05:
                    correct_count += 1
                elif max_index == 2 and true_val > 60.05:
                    correct_count += 1

                if max_index_narrow == 0 and true_val < 59.964:
                    correct_count_narrow += 1
                elif max_index_narrow == 1 and 59.964 <= true_val <= 60.036:
                    correct_count_narrow += 1
                elif max_index_narrow == 2 and true_val > 60.036:
                    correct_count_narrow += 1

                total_count += 1
            accuracy_median[i] = correct_count / total_count if total_count > 0 else np.nan
            # print("index: " + str(i) + "Correct Count: " + str(correct_count) + "Total Count: " + str(total_count))

            acc_w = correct_count / total_count * 100
            acc_n = correct_count_narrow / total_count * 100

            print("Frequency "
                  f"Interval I Accuracy: {acc_w:.2f}%   "
                  f"Interval II Accuracy: {acc_n:.2f}%")

        else:
            accuracy_median[i] = np.mean(np.sign(y_true[mask]) == np.sign(y_median[mask]))
            if i == 0:
                correct_count = 0
                total_count = 0
                correct_count_narrow = 0
                for j in range(len(y_true)):
                    if not mask[j]:
                        continue
                    preds = pred_all[j, :, i]

                    count_low = np.sum(preds <= 0)
                    count_high = np.sum(preds > 0)
                    counts = np.array([count_low, count_high])
                    max_index = np.argmax(counts)

                    count_low_narrow = np.sum(preds <= -200)
                    count_mid_narrow = np.sum((preds > -200) & (preds <= 200))
                    count_high_narrow = np.sum(preds > 200)
                    counts_narrow = np.array([count_low_narrow, count_mid_narrow, count_high_narrow])
                    max_index_narrow = np.argmax(counts_narrow)

                    true_val = y_true[j]
                    if max_index == 0 and true_val <= 0:
                        correct_count += 1
                    elif max_index == 1 and true_val > 0:
                        correct_count += 1

                    if max_index_narrow == 0 and true_val <= -200:
                        correct_count_narrow += 1
                    elif max_index_narrow == 1 and -200 < true_val <= 200:
                        correct_count_narrow += 1
                    elif max_index_narrow == 2 and true_val > 200:
                        correct_count_narrow += 1

                    total_count += 1
                accuracy_median[i] = correct_count / total_count if total_count > 0 else np.nan
                # print("index: " + str(i) + "Correct Count: " + str(correct_count) + "Total Count: " + str(total_count))

                acc_w = correct_count / total_count * 100
                acc_n = correct_count_narrow / total_count * 100

                print("ACE"
                      f"Interval I Accuracy: {acc_w:.2f}%   "
                      f"Interval II Accuracy: {acc_n:.2f}%")

    percentiles = extract_percentiles(pred_all)

    return mse, mae, pred_mape, accuracy_mean, accuracy_median, accuracy_mape, percentiles


def crps(y_true_all, y_pred_all, sample_weight=None):
    num_samples = y_pred_all.shape[1]
    total_crps = []
    y_pred = np.transpose(y_pred_all, (1, 0))
    absolute_error = np.mean(np.abs(y_pred - y_true_all), axis=0)

    if num_samples == 1:
        return np.average(absolute_error, weights=sample_weight)

    y_pred = np.sort(y_pred, axis=0)
    b0 = y_pred.mean(axis=0)
    b1_values = y_pred * np.arange(num_samples).reshape((num_samples, 1))
    b1 = b1_values.mean(axis=0) / num_samples

    per_obs_crps = absolute_error + b0 - 2 * b1
    crps = np.average(per_obs_crps, weights=sample_weight)
    total_crps.append(crps)
    return sum(total_crps) / len(total_crps)


def add_linear_means(pred_residual_samples: np.ndarray,
                     date: str,
                     seq_len: int,
                     csv_dir: str = "data/residuals"):
    """
    Aligns rows automatically; returns (min_len , 1000 , 2).
    """
    means = pd.read_csv(f"{csv_dir}/{date}_linear_preds.csv").values   # (M_full, 2)

    # print("means")
    # print(means.shape)
    # print(means)
    #
    # print("pred_residual_samples")
    # print(pred_residual_samples.shape)
    # print(pred_residual_samples)

    # print("after means")
    # print(means)
    # print(means.shape)
    #
    # print("after resid")
    # print(pred_residual_samples)
    # print(pred_residual_samples.shape)

    M_pred   = pred_residual_samples.shape[0]
    M_means  = means.shape[0]
    min_len  = min(M_pred, M_means)

    # print(seq_len)
    offset = M_means - M_pred
    means = means[offset:, :]

    M_pred = pred_residual_samples.shape[0]
    M_means = means.shape[0]
    min_len = min(M_pred, M_means)

    if M_pred != M_means:
        print(f"[WARN] size mismatch: residuals={M_pred}, means={M_means} ⇒ "
              f"cropping to {min_len}")
        print("Change the hardcoded mean slice value in utils_revised line 217")
        pred_residual_samples = pred_residual_samples[:min_len, :, :]
        means                 = means[:min_len, :]

    # print("Combined results")
    # print(pred_residual_samples + means[:, None, :])

    return pred_residual_samples + means[:, None, :]        # broadcast OK
