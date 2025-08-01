import torch
import numpy as np
import pandas as pd
from math import erf, sqrt

# def linear_distribution_terms(training_data, seq_len, pred_step):
#     """
#     Parameters
#     ----------
#     training_data : Tensor (D, N)
#         Two rows: ACE and Frequency for one full training day.
#     seq_len       : int
#         Length n of the history window (120 if you want 8 min).
#     pred_step     : int
#         Forecast horizon T in timesteps (15 for 1 min ahead, etc.).
#
#     Returns
#     -------
#     mu_x  : (D * seq_len,)
#     mu_y  : (D,)
#     cov_x : (D * seq_len, D * seq_len)
#     cov_y : (D, D)
#     cov_xy: (D * seq_len, D)
#     cov_yx: (D, D * seq_len)
#     """
#     X, Y = gather_samples(training_data, seq_len, pred_step)   # (M, D·n), (M, D)
#
#     # print("X")
#     # print(X)
#     # print(X.shape)
#     # print("Y")
#     # print(Y)
#     # print(Y.shape)
#     # print("training data")
#     # print(training_data)
#     # print(training_data.shape)
#
#     # means
#     mu_x = X.mean(dim=0)
#     mu_y = Y.mean(dim=0)
#
#     # print(mu_x)
#     # print(mu_y)
#
#     # centred
#     Xc = X - mu_x
#     Yc = Y - mu_y
#     M   = X.size(0)                      # number of windows
#
#     # print(Xc)
#     # print(Xc.shape)
#     # print(Yc)
#     # print(Yc.shape)
#     # print(M)
#
#     # covariances (unbiased: divide by M-1)
#     cov_x  = Xc.t().mm(Xc) / (M - 1)
#     cov_y  = Yc.t().mm(Yc) / (M - 1)
#     cov_xy = Xc.t().mm(Yc) / (M - 1)
#     cov_yx = cov_xy.t()
#
#     # print(cov_x)
#     # print(cov_x.shape)
#     # print(cov_y)
#     # print(cov_y.shape)
#     # print(cov_xy)
#     # print(cov_xy.shape)
#     # print(cov_yx)
#     # print(cov_yx.shape)
#
#     cov_x_inv = torch.linalg.inv(cov_x)  # (240,240)
#     K = cov_yx.mm(cov_x_inv)  # (2,240)     # precalculating term needed for calculating mean of normal distributions
#     Sigma_cond = cov_y - K.mm(cov_xy)  # (2,2)  optional but handy
#
#     """
#     K          = Σ_yx  Σ_x^{-1}
#     Sigma_cond = Σ_y  - Σ_yx Σ_x^{-1} Σ_xy   (optional)
#     """
#
#     return mu_x, mu_y, K, Sigma_cond

import torch

def linear_distribution_terms(training_data: torch.Tensor,
                              seq_len: int,
                              pred_step: int,
                              z_thresh: float = 3.0,
                              eps: float = 1e-6):
    """
    Parameters
    ----------
    training_data : Tensor  (D , N)
        D signals (here D=1 for NYISO prices or D=2 for ACE & Freq)
    seq_len       : int     history window length  n
    pred_step     : int     forecast horizon       T
    z_thresh      : float   |z| above which a point is discarded
    eps           : float   diagonal jitter added to Σ_x  (regularisation)

    Returns
    -------
    mu_x, mu_y, K, Sigma_cond   (see original docstring)
    """

    D, N = training_data.shape
    M    = N - seq_len - pred_step + 1         # max. number of windows

    # ------------------------------------------------------------------
    # 1.  Pre-compute mean & std on the ENTIRE training day  (robust     )
    # ------------------------------------------------------------------
    mean_all = training_data.mean(dim=1, keepdim=True)          # (D,1)
    std_all  = training_data.std (dim=1, keepdim=True).clamp_min(1e-9)

    # ------------------------------------------------------------------
    # 2.  Build X / Y but skip windows that contain a ≥3 σ spike
    # ------------------------------------------------------------------
    X_rows, Y_rows = [], []

    for t0 in range(M):                           # window start
        t_hist   = slice(t0, t0 + seq_len)        # history indices
        t_target = t0 + seq_len + pred_step - 1   # scalar index

        window   = training_data[:, t_hist]                      # (D,n)
        target   = training_data[:, t_target]                    # (D,)

        # ---- spike check ------------------------------------------------
        z_hist   = (window - mean_all)  / std_all
        z_target = (target - mean_all.squeeze(1)) / std_all.squeeze(1)

        if (z_hist.abs()   > z_thresh).any() or \
           (z_target.abs() > z_thresh).any():
            continue                           # skip this sample  🔥

        # -----------------------------------------------------------------
        X_rows.append(window.reshape(-1))      # (D·n,)
        Y_rows.append(target)                  # (D,)
    # ------------------------------------------------------------------

    if len(X_rows) == 0:
        raise ValueError("All windows were removed by the 3σ filter.")

    X = torch.stack(X_rows)         # (M_clean , D·n)
    Y = torch.stack(Y_rows)         # (M_clean , D)

    # ------------------------------------------------------------------
    # 3.  Empirical moments  (unbiased, ε-regularised)
    # ------------------------------------------------------------------
    mu_x = X.mean(dim=0)
    mu_y = Y.mean(dim=0)

    Xc   = X - mu_x
    Yc   = Y - mu_y
    Mc   = Xc.size(0)

    Σ_x  = Xc.t().mm(Xc) / (Mc - 1) + eps * torch.eye(D * seq_len)
    Σ_y  = Yc.t().mm(Yc) / (Mc - 1)
    Σ_xy = Xc.t().mm(Yc) / (Mc - 1)
    Σ_yx = Σ_xy.t()

    # ------------------------------------------------------------------
    # 4.  Linear-Gaussian coefficients
    # ------------------------------------------------------------------
    K          = Σ_yx.mm(torch.linalg.inv(Σ_x))           # (D , D·n)
    Sigma_cond = Σ_y - K.mm(Σ_xy)                         # (D , D)

    return mu_x, mu_y, K, Sigma_cond


def gather_samples(training_data, seq_len, pred_step):
    """
    Build design matrix X and target matrix Y from one day of ACE/Freq data.

    Returns
    -------
    X : (M, D * seq_len)   flattened history windows
    Y : (M, D)             ACE & Freq at t+T
    """
    D, N = training_data.shape          # D=2 signals, N = number of time steps (21600 for 1 day)

    M = N - seq_len - pred_step + 1     # number of sliding windows

    X_list = []
    Y_list = []

    for idx in range(M):
        hist   = training_data[:, idx : idx + seq_len]          # (D, n). If idx = 0, then you'd want to take values 0-119
        target = training_data[:, idx + seq_len + pred_step - 1]# (D,). If idx = 0, then you'd want to take index 134

        X_list.append(hist.reshape(-1))   # flatten to (D·n,)
        Y_list.append(target)

    X = torch.stack(X_list)              # (M, D·n)
    Y = torch.stack(Y_list)              # (M, D)

    return X, Y


# def generate_linear_forecast(
#         test_data, seq_len, pred_step, mu_x, mu_y, K,
#         csv_path="linear_predictions_4-4_1Min.csv"):
#
#     D, N = test_data.shape
#     M = N - seq_len - pred_step + 1
#     linear_predictions = torch.empty((M, D), dtype=test_data.dtype,
#                                      device=test_data.device)
#
#     for idx in range(M):
#         hist = test_data[:, idx: idx + seq_len].reshape(-1)
#         diff = hist - mu_x
#         mu_cond = mu_y + K.mm(diff.unsqueeze(1)).squeeze(1)
#         linear_predictions[idx] = mu_cond
#
#     # 1) move tensors to CPU → numpy
#     preds_np = linear_predictions.detach().cpu().numpy()
#     true_np  = test_data.detach().cpu().numpy()
#
#     # 2) drop the first (seq_len + pred_step − 1) rows of truth
#     offset   = seq_len + pred_step - 1          # rows that have no forecast
#     true_np  = true_np[:, offset : offset + M].T   # shape (M, 2)
#
#     # 3) build dataframe
#     df = pd.DataFrame(
#         np.hstack([true_np, preds_np]),
#         columns=["ACE_true", "Freq_true", "ACE_pred", "Freq_pred"],
#     )
#
#     df.to_csv(csv_path, index=False)
#     print(f"[Linear] wrote {csv_path}  ({len(df)} rows)")
#
#     linear_metrics(test_data, linear_predictions, seq_len, pred_step, "Linear")
#     # return linear_predictions

# def generate_linear_forecast(test_data, seq_len, pred_step, mu_x, mu_y, K, Sigma_cond):
#     # using the mean of the conditional normal distribution as predicted value (MMSE approach)
#     # print("Test data")
#     # print(test_data)
#     # print(test_data.shape)
#     D, N = test_data.shape  # D=2 signals, N = number of time steps (21600 for 1 day)
#     M = N - seq_len - pred_step + 1  # number of sliding windows
#     # print("D")
#     # print(D)
#     #
#     # print("N, seq_len, pred_step")
#     # print((str(N), str(seq_len), str(pred_step)))
#     # print("M")
#     # print(M)
#
#     linear_predictions = torch.empty((M, D), dtype = test_data.dtype, device=test_data.device)
#
#     for idx in range(M):
#         hist = test_data[:, idx: idx + seq_len].reshape(-1)  # (D*n,)
#         diff = hist - mu_x
#         mu_cond = mu_y + K.mm(diff.unsqueeze(1)).squeeze(1)  # (D,)
#         linear_predictions[idx] = mu_cond
#
#     print("linear_predictions")
#     print(linear_predictions)
#     print(linear_predictions.shape)
#
#     variance_vector = Sigma_cond.diag().cpu().numpy()
#     # print("variance vector")
#     # print(variance_vector)
#     linear_metrics(test_data, linear_predictions, seq_len, pred_step, "Linear", variance_vector)
#     # point_metrics(test_data, linear_predictions, seq_len, pred_step, "Linear")
#     # test_buckets(test_data, linear_predictions, seq_len, pred_step)
#
#     return linear_predictions

import torch
import numpy as np

def generate_linear_forecast(test_data      : torch.Tensor,
                             seq_len        : int,
                             pred_step      : int,
                             mu_x           : torch.Tensor,
                             mu_y           : torch.Tensor,
                             K              : torch.Tensor,
                             Sigma_cond     : torch.Tensor,
                             z_thresh       : float = 3.0):
    """
    Forecast with a Gaussian linear predictor while masking test-time spikes.

    Any sample whose absolute z-score (vs. the *whole* test-day mean/σ)
    exceeds `z_thresh` is replaced by the last valid value in that channel.
    """

    # ------------------------------------------------------------
    # 1)  Forward–fill spike points in a copy of the test tensor
    # ------------------------------------------------------------
    D, N        = test_data.shape               # (channels , time)
    clean       = test_data.clone()             # will be modified in-place

    mean_d      = test_data.mean(dim=1, keepdim=True)          # (D,1)
    std_d       = test_data.std (dim=1, keepdim=True).clamp_min(1e-9)

    z_scores    = (test_data - mean_d) / std_d                 # (D,N)
    spike_mask  = z_scores.abs() > z_thresh                    # boolean

    # forward-fill channel-wise
    for d in range(D):
        last_good = clean[d, 0]
        for t in range(N):
            if spike_mask[d, t]:
                clean[d, t] = last_good          # overwrite spike
            else:
                last_good = clean[d, t]

    # ------------------------------------------------------------
    # 2)  Standard conditional-Gaussian prediction
    # ------------------------------------------------------------
    M = N - seq_len - pred_step + 1             # #windows
    preds = torch.empty((M, D),
                        dtype=test_data.dtype,
                        device=test_data.device)

    for idx in range(M):
        hist = clean[:, idx : idx + seq_len].reshape(-1)       # (D·n,)
        diff = hist - mu_x                                     # centred
        mu_c = mu_y + K.mm(diff.unsqueeze(1)).squeeze(1)       # (D,)
        preds[idx] = mu_c

    # ------------------------------------------------------------
    # 3)  Evaluate
    # ------------------------------------------------------------
    var_vec = Sigma_cond.diag().cpu().numpy()      # (D,)
    linear_metrics(test_data, preds,
                   seq_len, pred_step,
                   pred_type="Linear",
                   variance_vector=var_vec)

    return preds


def naive_prediction(test_data, seq_len, pred_step):
    naive = test_data[:, seq_len-1:-pred_step]
    # print("naive")
    # print(naive)
    # print(naive.shape)
    naive_preds = naive.t().contiguous()

    # print("naive_preds")
    # print(naive_preds)
    # print(naive_preds.shape)

    point_metrics(test_data, naive_preds, seq_len, pred_step, "Naive")
    # return naive_preds

# def naive_prediction(test_data: torch.Tensor,
#                      seq_len   : int,
#                      pred_step : int,
#                      z_thresh  : float = 3.0):
#     """
#     Persistence forecast with spike suppression.
#
#     If the candidate persisted value is more than `z_thresh` standard
#     deviations away from that channel’s test-day mean, replace it by
#     the mean itself.
#     """
#     # ------------------------------------------------------------
#     # 1) build the usual (D , M) matrix of persisted values
#     # ------------------------------------------------------------
#     naive      = test_data[:, seq_len-1:-pred_step]            # (D , M)
#     D, M       = naive.shape
#     preds      = naive.clone()                                 # will be edited
#
#     # channel-wise statistics over the *entire* test day
#     mean_d     = test_data.mean(dim=1, keepdim=True)           # (D,1)
#     std_d      = test_data.std (dim=1, keepdim=True).clamp_min(1e-9)
#
#     # ------------------------------------------------------------
#     # 2) suppress outliers (> z_thresh σ) in the prediction matrix
#     # ------------------------------------------------------------
#     z_scores   = (preds - mean_d) / std_d                      # broadcast OK
#     mask_spike = z_scores.abs() > z_thresh                     # boolean
#
#     # replace spikes with mean
#     preds[mask_spike] = mean_d.expand_as(preds)[mask_spike]
#
#     # ------------------------------------------------------------
#     # 3) convert to (M , D) and evaluate
#     # ------------------------------------------------------------
#     naive_preds = preds.t().contiguous()                       # (M , D)
#
#     point_metrics(test_data, naive_preds,
#                   seq_len, pred_step,
#                   pred_type="Naive-clipped")
#     # return naive_preds


# def naive_prediction(
#     test_data, seq_len, pred_step,
#     csv_path="naive_predictions_4-4_1Min.csv"):
#
#     naive      = test_data[:, seq_len-1:-pred_step]      # (2, M)
#     naive_preds = naive.t().contiguous()                 # (M, 2) rows = forecasts
#
#     preds_np = naive_preds.detach().cpu().numpy()
#     true_np  = test_data.detach().cpu().numpy()
#
#     offset   = seq_len + pred_step - 1
#     M        = preds_np.shape[0]
#     true_np  = true_np[:, offset : offset + M].T
#
#     df = pd.DataFrame(
#         np.hstack([true_np, preds_np]),
#         columns=["ACE_true", "Freq_true", "ACE_pred", "Freq_pred"],
#     )
#     df.to_csv(csv_path, index=False)
#     print(f"[Naive] wrote {csv_path}  ({len(df)} rows)")
#
#     point_metrics(test_data, naive_preds, seq_len, pred_step, "Naive")
#     # return naive_preds


def norm_cdf(z_score):
    return 0.5 * (1.0 + erf(z_score / sqrt(2.0)))


def linear_metrics(true, predictions, seq_len, pred_step, pred_type, variance_vector):
    # ---------- convert to numpy -------------------------------
    true        = true.detach().cpu().numpy() if torch.is_tensor(true) else np.asarray(true)
    predictions = predictions.detach().cpu().numpy() if torch.is_tensor(predictions) else np.asarray(predictions)

    # print(true)
    # print(true.shape)

    # ---------- align shapes -----------------------------------
    offset = seq_len + pred_step - 1
    M = predictions.shape[0]  # same number of rows as preds
    true = true[:, offset: offset + M].T  # shape (M, 2)  ✅


    mse = np.zeros(true.shape[1])
    mae = np.zeros(true.shape[1])

    # print("true shape:", true.shape)
    # print("pred shape:", predictions.shape)

    for i in range(true.shape[1]):           # i = 0 (ACE) , 1 (Freq)
        y_true = true[:, i]
        y_pred = predictions[:, i]

        # σ-clip mask
        std  = np.std(y_true)
        mean = np.mean(y_true)
        mask = np.abs(y_true - mean) <= 3 * std


        # MSE & MAE on masked rows
        mse[i] = np.mean((y_true[mask] - y_pred[mask]) ** 2)
        mae[i] = np.mean(np.abs(y_true[mask] - y_pred[mask]))

        print("\n----------- {} Stats -----------".format("ACE" if i == 0 else "Frequency"))
        print("MSE:", mse[i])
        print("MAE:", mae[i])

        # --------------------------------------------------------
        # Interval-bucket accuracy
        # --------------------------------------------------------
        correct_wide  = 0      # “Interval I” in your printout
        correct_narrow= 0      # “Interval II”
        total_count   = 0

        sigma = sqrt(variance_vector[i])

        for j in range(len(y_true)):
            if not mask[j]:        # skip outliers
                continue

            t_val = y_true[j]
            mu = y_pred[j]

            # ---- choose bucket with highest probability --------------
            if i == 1:  # Frequency
                # wide
                p_low = norm_cdf((59.95 - mu) / sigma)
                p_mid = norm_cdf((60.05 - mu) / sigma) - p_low
                p_high = 1.0 - norm_cdf((60.05 - mu) / sigma)
                bucket = np.argmax([p_low, p_mid, p_high])

                # narrow
                p_low_n = norm_cdf((59.964 - mu) / sigma)
                p_mid_n = norm_cdf((60.036 - mu) / sigma) - p_low_n
                p_high_n = 1.0 - norm_cdf((60.036 - mu) / sigma)
                bucket_n = np.argmax([p_low_n, p_mid_n, p_high_n])

                if bucket == 0 and t_val < 59.95:
                    correct_wide += 1
                elif bucket == 1 and 59.95 <= t_val <= 60.05:
                    correct_wide += 1
                elif bucket == 2 and t_val > 60.05:
                    correct_wide += 1

                if bucket_n == 0 and t_val < 59.964:
                    correct_narrow += 1
                elif bucket_n == 1 and 59.964 <= t_val <= 60.036:
                    correct_narrow += 1
                elif bucket_n == 2 and t_val > 60.036:
                    correct_narrow += 1

            else:  # ACE
                p_neg = norm_cdf((-0.0 - mu) / sigma)
                p_pos = 1.0 - p_neg
                bucket = np.argmax([p_neg, p_pos])

                p_low_n = norm_cdf((-200.0 - mu) / sigma)
                p_mid_n = norm_cdf((200.0 - mu) / sigma) - p_low_n
                p_high_n = 1.0 - norm_cdf((200.0 - mu) / sigma)
                bucket_n = np.argmax([p_low_n, p_mid_n, p_high_n])

                if bucket == 0 and t_val <= 0:
                    correct_wide += 1
                elif bucket == 1 and t_val > 0:
                    correct_wide += 1

                if bucket_n == 0 and t_val <= -200:
                    correct_narrow += 1
                elif bucket_n == 1 and -200 < t_val <= 200:
                    correct_narrow += 1
                elif bucket_n == 2 and t_val > 200:
                    correct_narrow += 1

            total_count += 1

        if total_count > 0:
            acc_w = 100 * correct_wide   / total_count
            acc_n = 100 * correct_narrow / total_count
        else:
            acc_w = acc_n = np.nan

        label = "ACE" if i == 0 else "Frequency"
        print(f"{label} prediction type: {pred_type} "
              f"Interval I Accuracy: {acc_w:.2f}%   "
              f"Interval II Accuracy: {acc_n:.2f}%")
        print("this is the most likely bucket")
    return mse, mae


def point_metrics(true, predictions, seq_len, pred_step, pred_type):
    # ---------- convert to numpy -------------------------------
    true        = true.detach().cpu().numpy() if torch.is_tensor(true) else np.asarray(true)
    predictions = predictions.detach().cpu().numpy() if torch.is_tensor(predictions) else np.asarray(predictions)

    # print(true)
    # print(true.shape)

    # ---------- align shapes -----------------------------------
    offset = seq_len + pred_step - 1
    M = predictions.shape[0]  # same number of rows as preds
    true = true[:, offset: offset + M].T  # shape (M, 2)  ✅


    mse = np.zeros(true.shape[1])
    mae = np.zeros(true.shape[1])

    # print("true shape:", true.shape)
    # print("pred shape:", predictions.shape)

    for i in range(true.shape[1]):           # i = 0 (ACE) , 1 (Freq)
        y_true = true[:, i]
        y_pred = predictions[:, i]

        # σ-clip mask
        std  = np.std(y_true)
        mean = np.mean(y_true)
        mask = np.abs(y_true - mean) <= 3 * std


        # MSE & MAE on masked rows
        mse[i] = np.mean((y_true[mask] - y_pred[mask]) ** 2)
        mae[i] = np.mean(np.abs(y_true[mask] - y_pred[mask]))

        print("\n----------- {} Stats -----------".format("ACE" if i == 0 else "Frequency"))
        print("MSE:", mse[i])
        print("MAE:", mae[i])

        # --------------------------------------------------------
        # Interval-bucket accuracy
        # --------------------------------------------------------
        correct_wide  = 0      # “Interval I” in your printout
        correct_narrow= 0      # “Interval II”
        total_count   = 0

        for j in range(len(y_true)):
            if not mask[j]:        # skip outliers
                continue

            t_val = y_true[j]
            p_val = y_pred[j]

            if i == 1:  # ---------------- frequency buckets -----------------
                # wide buckets
                if (t_val <= 59.95 and p_val <= 59.95) \
                   or (59.95 < t_val <= 60.05 and 59.95 < p_val <= 60.05) \
                   or (t_val > 60.05 and p_val > 60.05):
                    correct_wide += 1

                # narrow buckets
                if (t_val <= 59.964 and p_val <= 59.964) \
                   or (59.964 < t_val <= 60.036 and 59.964 < p_val <= 60.036) \
                   or (t_val > 60.036 and p_val > 60.036):
                    correct_narrow += 1

            elif i == 0:  # --------------- ACE buckets -----------------------
                # wide buckets
                if (t_val <= 0 and p_val <= 0) or (t_val > 0 and p_val > 0):
                    correct_wide += 1

                # narrow buckets
                if (t_val <= -200 and p_val <= -200) \
                   or (-200 < t_val <= 200 and -200 < p_val <= 200) \
                   or (t_val > 200 and p_val > 200):
                    correct_narrow += 1

            total_count += 1

        if total_count > 0:
            acc_w = 100 * correct_wide   / total_count
            acc_n = 100 * correct_narrow / total_count
        else:
            acc_w = acc_n = np.nan

        label = "ACE" if i == 0 else "Frequency"
        print(f"{label} prediction type: {pred_type} "
              f"Interval I Accuracy: {acc_w:.2f}%   "
              f"Interval II Accuracy: {acc_n:.2f}%")

    return mse, mae
