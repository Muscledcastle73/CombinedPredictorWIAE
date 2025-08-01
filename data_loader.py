import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datetime import datetime, timedelta


class Custom_Dataset(Dataset):
    def __init__(
        self,
        input_size: int,
        data_path: str,
        dataset: str,
        flag: str,
        input_dim: int,
        filter_size=1,
        date: str = None,
    ):
        self.input_size = input_size
        self.train_data, self.test_data = eval("prepare_" + dataset)(data_path, date)
        self.flag = flag
        self.input_dim = input_dim
        self.filter_size = filter_size
        self.pred_step = input_dim - 2 * (self.filter_size - 1)

    def __len__(self):
        if self.flag == "train":
            return self.train_data.shape[1] - self.input_size
        elif self.flag == "test":
            return (
                self.test_data.shape[1] - self.input_size - self.pred_step
            )  # Calculate the number of windows

    def __getitem__(self, index):
        if self.flag == "train":
            y_input = self.train_data[:, index : index + self.input_size].clone()

            if torch.all(y_input.std(dim=1) > 0):
                mean = y_input.mean(dim=1)
                std = y_input.std(dim=1)
                # std = torch.clamp(std, min=1e-2)
                y_input = (y_input - mean.unsqueeze(1)) / std.unsqueeze(1)

            return y_input.squeeze(1)

        if self.flag == "test":
            y_input = self.test_data[:, index : index + self.input_size].clone()

            if torch.all(y_input.std(dim=1) > 0):
                mean = y_input.mean(dim=1)
                std = y_input.std(dim=1)
                # std = torch.clamp(std, min=1e-2)
                y_input = (y_input - mean.unsqueeze(1)) / std.unsqueeze(1)
            else:
                std = torch.ones(y_input.std(dim=1).shape)
                mean = torch.zeros(y_input.mean(dim=1).shape)

            y_true = self.test_data[
                :, index + self.input_size
            ]  # Only return the channel that needs prediction, which is always placed as the first channel

            return y_input.squeeze(1), y_true, mean, std


def prepare_PJM(csv_path: str):
    train_start = "2022-01-01 00:00:00"
    train_end = "2022-09-01 23:00:00"
    test_start = "2022-09-01 00:00:00"
    test_end = "2022-12-31 00:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_PJM_spread(csv_path: str):
    train_start = "2022-01-01 00:00:00"
    train_end = "2022-09-01 23:00:00"
    test_start = "2022-09-01 00:00:00"
    test_end = "2022-12-31 00:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_NYISO_spread(csv_path: str):
    train_start = "2023-07-01 00:00:00"
    train_end = "2023-07-25 23:00:00"
    test_start = "2023-07-25 23:00:05"
    test_end = "2023-07-31 23:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_NYISO_RT(csv_path: str):
    train_start = "2023-07-01 00:00:05"
    train_end = "2023-07-25 23:00:00"
    test_start = "2023-07-25 23:00:05"
    test_end = "2023-07-31 22:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_NYISO_spread_2D(csv_path: str):
    train_start = "2023-07-01 00:00:05"
    train_end = "2023-07-25 23:00:00"
    test_start = "2023-07-25 23:00:05"
    test_end = "2023-07-31 22:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_NYISO_RTDA_load(csv_path: str, date: str):
    # train_start = "2023-07-01 00:00:05"
    # train_end = "2023-07-25 23:00:00"
    # test_start = "2023-07-25 23:00:05"
    # test_end = "2023-07-31 22:00:00"
    train_start = "2023-07-01 00:00:05"
    train_end = "2023-07-11 00:00:00"
    test_start = "2023-07-11 00:00:05"
    test_end = "2023-07-26 00:00:00"

    # train_start = "2023-07-01 00:00:05"
    # train_end = "2023-07-26 00:00:00"
    # test_start = "2023-07-26 00:00:05"
    # test_end = "2023-07-31 22:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_NYISO_RTDA_load_2D(csv_path: str):
    train_start = "2023-07-01 00:00:05"
    train_end = "2023-07-25 23:00:00"
    test_start = "2023-07-25 23:00:05"
    test_end = "2023-07-31 21:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_PJM_ACE(csv_path: str):
    train_start = "2024-01-24 05:40:00"
    train_end = "2024-01-25 16:40:00"
    test_start = "2024-01-25 16:40:15"
    test_end = "2024-01-25 20:40:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_CTS(csv_path: str):
    train_start = "2024-02-08 00:00:00"
    train_end = "2024-02-18 00:00:00"
    test_start = "2024-02-18 00:00:15"
    test_end = "2024-02-20 00:10:15"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)


def prepare_CTS_2D(csv_path: str):
    train_start = "2024-02-08 00:00:00"
    train_end = "2024-02-18 00:00:00"
    test_start = "2024-02-18 00:00:15"
    test_end = "2024-02-20 00:10:15"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)

def prepare_CAISO_ACE(csv_path: str, date: str):
    current_date = datetime.strptime(date, "%Y-%m-%d")
    next_date = current_date + timedelta(days=1)
    train_start = current_date.strftime("%Y-%m-%d 00:00:00")
    train_end = current_date.strftime("%Y-%m-%d 23:59:56")
    # test_start = next_date.strftime("%Y-%m-%d 00:00:00")
    # test_end = next_date.strftime("%Y-%m-%d 04:00:00")
    # train_start = current_date.strftime("%Y-%m-%d 12:00:00")
    # train_end = next_date.strftime("%Y-%m-%d 11:59:56")
    test_start = next_date.strftime("%Y-%m-%d 12:00:00")
    test_end = next_date.strftime("%Y-%m-%d 16:00:00")
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)

def prepare_CAISO_Frequency(csv_path: str, date: str):
    current_date = datetime.strptime(date, "%Y-%m-%d")
    next_date = current_date + timedelta(days=1)
    train_start = current_date.strftime("%Y-%m-%d 12:00:00")
    train_end = next_date.strftime("%Y-%m-%d 11:59:56")
    test_start = next_date.strftime("%Y-%m-%d 12:00:00")
    test_end = next_date.strftime("%Y-%m-%d 16:00:00")
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)
    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)

def prepare_CAISO_ACE_2D(csv_path: str, date: str):
    current_date = datetime.strptime(date, "%Y-%m-%d")
    next_date = current_date + timedelta(days=1)
    train_start = current_date.strftime("%Y-%m-%d 00:00:00")
    train_end = current_date.strftime("%Y-%m-%d 23:59:56")
    test_start = next_date.strftime("%Y-%m-%d 00:00:00")
    test_end = next_date.strftime("%Y-%m-%d 23:59:56")
    # train_start = "2025-04-01 00:00:00"
    # train_end = "2025-04-01 12:00:00"
    # test_start = "2025-04-01 12:00:04"
    # test_end = "2025-04-01 15:00:00"
    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    data_frame.fillna(method="bfill", inplace=True)
    # data_frame.set_index("datetime_beginning_ept", inplace=True)

    # data_frame = data_frame.apply(pd.to_numeric, errors="coerce")
    # data_frame["Actual Frequency"] -= 60.0

    training_data = torch.Tensor(
        data_frame[train_start:train_end].astype(np.float32).values
    )
    testing_data = torch.Tensor(
        data_frame[test_start:test_end].astype(np.float32).values
    )
    return training_data.transpose(0, 1), testing_data.transpose(0, 1)

# ------------------------------------------------------------------
# Residual loader for the hybrid linear + WIAE model
# ------------------------------------------------------------------
def prepare_CAISO_RESIDUALS(csv_dir: str, date: str):
    """
    Train on residuals from *previous* day, test on residuals of <date>.
    Files are expected at
        <csv_dir>/<YYYY-MM-DD>_linear_resid.csv
    """
    test_day   = pd.to_datetime(date)
    train_day  = test_day - pd.Timedelta(days=1)

    train_path = f"{csv_dir}/{train_day.date()}_linear_resid.csv"
    test_path  = f"{csv_dir}/{test_day.date()}_linear_resid.csv"

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # convert to torch (2, N) shape
    train_tensor = torch.Tensor(train_df.values.T).contiguous()
    test_tensor  = torch.Tensor(test_df.values.T).contiguous()
    return train_tensor, test_tensor


def prepare_pricing_residuals(csv_dir: str, date: str):
    train_path = "2023-07-11-" + csv_dir + "_linear_resid.csv"
    test_path = "2023-07-26-" + csv_dir + "_linear_resid.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # convert to torch (2, N) shape
    train_tensor = torch.Tensor(train_df.values.T).contiguous()
    test_tensor = torch.Tensor(test_df.values.T).contiguous()

    return train_tensor, test_tensor

