import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class Custom_Dataset(Dataset):
    def __init__(
        self,
        input_size: int,
        data_path: str,
        dataset: str,
        flag: str,
        input_dim: int,
        filter_size=1,
    ):
        self.input_size = input_size
        self.train_data, self.test_data = eval("prepare_" + dataset)(data_path)
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
                y_input = (y_input - mean.unsqueeze(1)) / std.unsqueeze(1)

            return y_input.squeeze(1)

        if self.flag == "test":
            y_input = self.test_data[:, index : index + self.input_size].clone()

            if torch.all(y_input.std(dim=1)) > 0:
                mean = y_input.mean(dim=1)
                std = y_input.std(dim=1)
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


def prepare_NYISO_RTDA_load(csv_path: str):
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


def prepare_NYISO_RTDA_load_2D(csv_path: str):
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


def prepare_NYISO_RTDA_Summer_Week1(csv_path: str):
    train_start = "2023-06-01 00:05:00"
    train_end = "2023-06-30 23:55:00"
    test_start = "2023-07-01 00:00:00"
    test_end = "2023-07-07 23:55:00"

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


def prepare_NYISO_RTDA_Summer_Week2(csv_path: str):
    train_start = "2023-06-08 00:00:00"
    train_end = "2023-07-07 23:55:00"
    test_start = "2023-07-08 00:00:00"
    test_end = "2023-07-14 23:55:00"

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


def prepare_NYISO_RTDA_Summer_Week3(csv_path: str):
    train_start = "2023-06-15 00:00:00"
    train_end = "2023-07-14 23:55:00"
    test_start = "2023-07-15 00:00:00"
    test_end = "2023-07-21 23:55:00"

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


def prepare_NYISO_RTDA_Summer_Week4(csv_path: str):
    train_start = "2023-06-22 00:00:00"
    train_end = "2023-07-21 23:55:00"
    test_start = "2023-07-22 00:00:00"
    test_end = "2023-07-28 23:55:00"

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


def prepare_NYISO_RTDA_Fall_Week1(csv_path: str):
    train_start = "2023-09-01 00:05:00"
    train_end = "2023-09-30 23:55:00"
    test_start = "2023-10-01 00:00:00"
    test_end = "2023-10-07 23:55:00"

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


def prepare_NYISO_RTDA_Fall_Week2(csv_path: str):
    train_start = "2023-09-08 00:00:00"
    train_end = "2023-10-07 23:55:00"
    test_start = "2023-10-08 00:00:00"
    test_end = "2023-10-14 23:55:00"

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


def prepare_NYISO_RTDA_Fall_Week3(csv_path: str):
    train_start = "2023-09-15 00:00:00"
    train_end = "2023-10-14 23:55:00"
    test_start = "2023-10-15 00:00:00"
    test_end = "2023-10-21 23:55:00"

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


def prepare_NYISO_RTDA_Fall_Week4(csv_path: str):
    train_start = "2023-09-22 00:00:00"
    train_end = "2023-10-21 23:55:00"
    test_start = "2023-10-22 00:00:00"
    test_end = "2023-10-28 23:55:00"

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


def prepare_NYISO_RTDA_Winter_Week1(csv_path: str):
    train_start = "2023-12-01 00:05:00"
    train_end = "2023-12-31 23:55:00"
    test_start = "2024-01-01 00:00:00"
    test_end = "2024-01-07 23:55:00"

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


def prepare_NYISO_RTDA_Winter_Week2(csv_path: str):
    train_start = "2023-12-08 00:00:00"
    train_end = "2024-01-07 23:55:00"
    test_start = "2024-01-08 00:00:00"
    test_end = "2024-01-14 23:55:00"

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


def prepare_NYISO_RTDA_Winter_Week3(csv_path: str):
    train_start = "2023-12-15 00:00:00"
    train_end = "2024-01-14 23:55:00"
    test_start = "2024-01-15 00:00:00"
    test_end = "2024-01-21 23:55:00"

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


def prepare_NYISO_RTDA_Winter_Week4(csv_path: str):
    train_start = "2023-12-22 00:00:00"
    train_end = "2024-01-21 23:55:00"
    test_start = "2024-01-22 00:00:00"
    test_end = "2024-01-28 23:55:00"

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


def prepare_NYISO_RTDA_Spring_Week1(csv_path: str):
    train_start = "2024-03-01 00:05:00"
    train_end = "2024-03-31 23:55:00"
    test_start = "2024-04-01 00:00:00"
    test_end = "2024-04-07 23:55:00"

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


def prepare_NYISO_RTDA_Spring_Week2(csv_path: str):
    train_start = "2024-03-08 00:00:00"
    train_end = "2024-04-07 23:55:00"
    test_start = "2024-04-08 00:00:00"
    test_end = "2024-04-14 23:55:00"

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


def prepare_NYISO_RTDA_Spring_Week3(csv_path: str):
    train_start = "2024-03-15 00:00:00"
    train_end = "2024-04-14 23:55:00"
    test_start = "2024-04-15 00:00:00"
    test_end = "2024-04-21 23:55:00"

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


def prepare_NYISO_RTDA_Spring_Week4(csv_path: str):
    train_start = "2024-03-22 00:00:00"
    train_end = "2024-04-21 23:55:00"
    test_start = "2024-04-22 00:00:00"
    test_end = "2024-04-28 23:55:00"

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

# Loads CAISO ACE data
def prepare_CAISO_ACE(csv_path: str):
    train_start = "2025-04-02 05:40:00"
    train_end = "2025-04-02 20:20:00" # only uses roughly 1.5 days of the data to train on
    test_start = "2025-04-02 20:20:15" # tests on roughly 4 hours of data
    test_end = "2025-04-02 21:00:00"
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


# Loads CAISO Frequency data
def prepare_CAISO_FREQUENCY(csv_path: str):
    train_start = "2025-04-02 05:40:00"
    train_end = "2025-04-02 16:40:00" # only uses roughly 1.5 days of the data to train on
    test_start = "2025-04-02 16:40:15" # tests on roughly 4 hours of data
    test_end = "2025-04-02 20:40:00"
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

# # Loads CAISO Frequency data
# def prepare_CAISO_both(csv_path: str):
#     train_start = "2025-04-02 05:40:00"
#     train_end = "2025-04-02 16:40:00" # only uses roughly 1.5 days of the data to train on
#     test_start = "2025-04-02 16:40:15" # tests on roughly 4 hours of data
#     test_end = "2025-04-02 20:40:00"
#     data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
#     data_frame.fillna(method="bfill", inplace=True)
#     # data_frame.set_index("datetime_beginning_ept", inplace=True)
#     training_data = torch.Tensor(
#         data_frame[train_start:train_end].astype(np.float32).values
#     )
#     testing_data = torch.Tensor(
#         data_frame[test_start:test_end].astype(np.float32).values
#     )
#     return training_data.transpose(0, 1), testing_data.transpose(0, 1)

def prepare_CAISO_both(csv_path: str):
    train_start = "2025-04-04 12:00:00"
    train_end = "2025-04-05 12:00:00"
    test_start = "2025-04-05 12:00:15"
    test_end = "2025-04-05 16:00:00"

    data_frame = pd.read_csv(csv_path, index_col=0, parse_dates=True, decimal=",")
    print("Full data shape:", data_frame.shape)
    print("Full index range:", data_frame.index.min(), "to", data_frame.index.max())

    data_frame.fillna(method="bfill", inplace=True)

    train_df = data_frame[train_start:train_end]
    test_df = data_frame[test_start:test_end]

    print("Train data shape after slicing:", train_df.shape)
    print("Test data shape after slicing:", test_df.shape)

    training_data = torch.Tensor(train_df.astype(np.float32).values)
    testing_data = torch.Tensor(test_df.astype(np.float32).values)

    return training_data.transpose(0, 1), testing_data.transpose(0, 1)
