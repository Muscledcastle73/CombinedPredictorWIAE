import numpy as np
import pandas as pd
import torch
import Linear_Predictor as lp
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import math

def output_linear_data(train_data: torch.Tensor,
                       test_data: torch.Tensor,
                       seq_len: int,
                       pred_step: int,
                       save_prefix: str = "2025-04-02"):
    """Generate linear‑model forecasts for *test_data*, compute residuals
    (true − forecast), and save both arrays to CSV files.

    Two files are created:
        <save_prefix>_linear_preds.csv   – columns: ACE_pred, Freq_pred
        <save_prefix>_linear_resid.csv   – columns: ACE_resid, Freq_resid

    Parameters
    ----------
    train_data : torch.Tensor (2, N_train)
        One full training day (ACE, Freq)
    test_data  : torch.Tensor (2, N_test)
        The following day used for evaluation.
    seq_len    : int
        History‑window length in samples (120 for 8 min @4 s).
    pred_step  : int
        Forecast horizon in samples (120 for 8 min ahead).
    save_prefix: str, optional
        Prefix used in the output filenames; typically the **test** date.
    """

    # 1. build linear predictor on *training* day
    mu_x, mu_y, K, sigma_cond = lp.linear_distribution_terms(train_data, seq_len, pred_step)

    # 2. forecast *test* day – we expect Linear_Predictor.generate_linear_forecast
    #    to RETURN the predictions; if your local copy only prints metrics,
    #    add a `return linear_predictions` at its end.
    linear_preds = lp.generate_linear_forecast(
        test_data, seq_len, pred_step, mu_x, mu_y, K, sigma_cond)

    if linear_preds is None:
        raise RuntimeError("generate_linear_forecast() must return the prediction array.\n"
                           "Add `return linear_predictions` at the end of that function.")

    # 3. Align shapes: drop first (seq_len + pred_step − 1) rows of truth
    offset = seq_len + pred_step - 1
    M      = linear_preds.shape[0]
    true   = test_data[:, offset : offset + M].T        # shape (M,2)

    # 4. Residuals (innovations)
    residuals = true - linear_preds.detach().cpu().numpy()

    # 5. Save to CSV
    preds_np = linear_preds.detach().cpu().numpy()
    # pd.DataFrame(preds_np, columns=["ACE_pred", "Freq_pred"]).to_csv(
    #     f"{save_prefix}_linear_preds.csv", index=False)
    # pd.DataFrame(residuals, columns=["ACE_resid", "Freq_resid"]).to_csv(
    #     f"{save_prefix}_linear_resid.csv", index=False)

    pd.DataFrame(preds_np, columns=["real_time_price", "day_ahead_price", "load"]).to_csv(
        f"{save_prefix}_linear_preds.csv", index=False)
    pd.DataFrame(residuals, columns=["RT_price_resid", "DA_price_resid", "load_resid"]).to_csv(
        f"{save_prefix}_linear_resid.csv", index=False)

    print(f"[combined_prediction] wrote {save_prefix}_linear_preds.csv and ..._linear_resid.csv  ({M} rows)")


# ---------------------------------------------------------------------------
# Example usage inside main.main (pseudo‑code):
# ---------------------------------------------------------------------------
# train_tensor = train_data.train_data     # (2,N_train)
# test_tensor  = test_data.test_data       # (2,N_test)
# output_linear_data(train_tensor, test_tensor,
#                    seq_len=120, pred_step=120,
#                    save_prefix="2025-04-03")


def load_caiso_truth(csv_path: str,
                     date: str,
                     seq_len: int,
                     pred_step: int) -> np.ndarray:
    """
    Returns a numpy array (M,2) of *ground-truth* ACE & Frequency
    that align with the residual forecast windows of the same day.

      M = 21600 - seq_len - pred_step + 1   (≈ 21304 for 1-min case)

    Parameters
    ----------
    csv_path  : str   full path to CAISO_both.csv
    date      : str   test day, e.g. "2025-04-03"
    seq_len   : int   history window length n
    pred_step : int   forecast horizon T
    """
    cur   = datetime.strptime(date, "%Y-%m-%d")
    start = cur.strftime("%Y-%m-%d 00:00:00")
    end   = cur.strftime("%Y-%m-%d 23:59:56")
    # print("date")
    # print(cur)

    # read the whole one-day slice
    df = (pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
            .loc[start:end]
            .astype(np.float32))

    # align with first available forecast :
    offset = seq_len * 2 + 41 + pred_step - 1          # rows to skip
    truth  = df.values[offset:, :]            # (M_full - offset , 2)

    print("truth")
    print(truth)
    print(truth.shape)
    return truth      # shape (M_truth , 2)


def add_pricing_linear_means(all_preds_all, seq_len: int, pred_step: int, decoder_in_len: int, filter_size: int, csv_path) -> np.ndarray:
    """
            Add the linear‑model mean values (price, day‑ahead price, load) from
            `csv_path` to every sample in `all_preds_all`.

            Parameters
            ----------
            all_preds_all : np.ndarray
                Shape (T, N, 3) – T timesteps (1610) × N samples (1000) × 3 features.
            seq_len : int, optional
                Number of leading rows in the CSV that belong to earlier history and
                should be ignored (default = 46 → start at row 47).
            csv_path : str, optional
                Path to the CSV with columns
                ["real_time_price", "day_ahead_price", "load"].

            Returns
            -------
            np.ndarray
                Same shape as `all_preds_all`, but shifted by the row‑wise means.
            """

    offset = (seq_len - decoder_in_len) + pred_step - 1
    print("offset = " + str(offset))
    print("all_preds_all")
    print(all_preds_all.shape)

    # 1. Load the CSV and drop the first `seq_len` rows
    df = (pd.read_csv(
        csv_path,
        usecols=["real_time_price", "day_ahead_price", "load"],
        dtype=np.float32)
          .iloc[offset:])  # → (T, 3)
          # .iloc[offset:offset + all_preds_all.shape[0]])  # → (T, 3)
    print("df")
    print(df.shape)
    # 2. Sanity‑check length
    if df.shape[0] != all_preds_all.shape[0]:
        raise ValueError(
            f"CSV after dropping {offset} rows has {df.shape[0]} rows, "
            f"but all_preds_all has {all_preds_all.shape[0]} timesteps."
        )

    # 3. Convert to ndarray and broadcast over the sample axis
    linear_means = df.to_numpy()  # (T, 3)
    adjusted = all_preds_all + linear_means[:, None, :]  # (T, N, 3)
    #
    print("df")
    print(df.shape)

    print("adjusted")
    print(adjusted.shape)
    # print(adjusted)
    return adjusted

def load_price_truth(
    all_preds_all,
    csv_path: str = "data/NYISO_Jul_RTDA_Load.csv",
) -> np.ndarray:
    """
    Load Real-Time, Day-Ahead, and Load values between start_ts and end_ts (inclusive)
    from the NYISO July dataset and return as a NumPy array of shape (M, 3), dtype float32.

    Parameters
    ----------
    csv_path : str
        Path to the CSV (default: "data/NYISO_Jul_RTDA_Load.csv").
    start_ts : str
        Inclusive start timestamp (e.g., "2023-07-26 07:55:00").
    end_ts   : str
        Inclusive end timestamp (e.g., "2023-07-31 22:00:00").
    expected_len : int | None
        If provided, validate that the slice has this many rows.

    Returns
    -------
    np.ndarray
        Array with columns [Real_Time, Day_Ahead, Load], shape (M, 3), dtype float32.
    """
    num_timesteps = all_preds_all.shape[0] - 1
    hour_difference = math.ceil(num_timesteps/12) % 24
    minutes_difference = (num_timesteps % 24) % 12

    if hour_difference <= 2:
        hour = 22 + hour_difference
    else:
        hour = 22 - hour_difference

    minute = 5 * (12 - minutes_difference)
    if minute == 60:
        minute = 0
    if hour < 10:
        hour = "0" + str(hour)
    if minute < 10:
        minute = "0" + str(minute)
    start_ts = "2023-07-26 " + str(hour) + ":" + str(minute) + ":00"
    print(start_ts)
    # start_ts: str = "2023-07-26 07:55:00"
    end_ts: str = "2023-07-31 22:00:00"
    cols = ["Real_Time", "Day_Ahead", "Load"]

    # Read, index by timestamp, and sort (just in case)
    df = (pd.read_csv(csv_path, parse_dates=["Time Stamp"])
            .set_index("Time Stamp")
            .sort_index())

    # Validate required columns
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Inclusive time slice
    sl = df.loc[start_ts:end_ts, cols].astype(np.float32)

    if sl.empty:
        raise ValueError(
            f"No rows found between {start_ts!r} and {end_ts!r}. "
            "Check timestamps and CSV contents."
        )

    true_values = sl.to_numpy()
    # print("true values")
    # print(true_values.shape)
    # print(true_values)

    return true_values

def export_csv(
    all_true:  np.ndarray,
    all_pred:  np.ndarray,
    out_path:  str | Path = "pricing_forecasts_vs_truth.csv",
    round_to:  int | None = 4,
) -> None:
    """
    Export ground‑truth and forecast arrays to a single CSV file.

    Parameters
    ----------
    all_true : np.ndarray
        Array of true values, shape (M, 3) → [Real_Time, Day_Ahead, Load].
    all_pred : np.ndarray
        Array of forecast values (e.g., median of samples), shape (M, 3).
    out_path : str or pathlib.Path, optional
        Destination filename (default "pricing_forecasts_vs_truth.csv").
    round_to : int | None, optional
        If given, round values to this many decimals before writing.

    The CSV columns will be:
        ["rt_true", "da_true", "load_true",
         "rt_pred", "da_pred", "load_pred"]
    """
    if all_true.shape != all_pred.shape:
        raise ValueError(
            f"Shape mismatch: all_true{all_true.shape} vs all_pred{all_pred.shape}"
        )
    if all_true.shape[1] != 3:
        raise ValueError(
            f"Expected 3 features per row, got {all_true.shape[1]}"
        )

    # Stack and label
    data = np.hstack((all_true, all_pred))
    cols = [
        "rt_true", "da_true", "load_true",
        "rt_pred", "da_pred", "load_pred"
    ]

    df = pd.DataFrame(data, columns=cols)
    if round_to is not None:
        df = df.round(round_to)

    df.to_csv(out_path, index=False)
    print(f"[export_csv] wrote {len(df)} rows → {out_path}")