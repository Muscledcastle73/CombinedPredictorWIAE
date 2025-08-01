import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from model_TV import Generator, Discriminator
from data_loader_TV import Custom_Dataset
from utils import calculate_gradient_penalty, metrics, estimate_coef

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

b0 = 0
b1 = 0


def arguement():
    parser = argparse.ArgumentParser()

    parser.add_argument("-data_path", type=str, required=True)
    parser.add_argument("-dataset", type=str, required=True)
    parser.add_argument("-data_bad", type=str, required=False)
    parser.add_argument("-pred_step", type=int, default=1)

    parser.add_argument("-degree", type=int, default=4)
    parser.add_argument("-block", type=int, default=100)
    parser.add_argument("-stride", type=int, default=100)

    parser.add_argument("-output_dim", type=int, default=1)
    parser.add_argument("-hidden_dim", type=int, default=100)
    parser.add_argument("-seq_len", type=int, default=50)
    parser.add_argument("-num_feature", type=int, default=2)
    parser.add_argument("-filter_size", type=int, default=20)

    parser.add_argument("-batch_size", type=int, default=60)
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-lrD", type=float, default=1e-5)
    parser.add_argument("-lrG", type=float, default=1e-5)
    parser.add_argument("-num_critic", type=int, default=10)

    parser.add_argument("-gp_coef_inn", type=float, default=0.1)
    parser.add_argument("-coef_recons", type=float, default=0.1)
    parser.add_argument("-gp_coef_recons", type=float, default=0.1)
    parser.add_argument("-seed", type=int, default=1)
    parser.add_argument("-sample_size", type=int, default=1000)

    parser.add_argument("--univariate", action="store_true", default=False)

    opt = parser.parse_args()

    return opt


def train_epoch(
    train_dataloader,
    encoder,
    decoder,
    inn_discriminator,
    recons_discriminator,
    optimizer_generator,
    optimizer_discriminator,
    opt,
):
    encoder.train()
    decoder.train()
    inn_discriminator.train()
    recons_discriminator.train()
    loss_D = []
    loss_G = []

    for x_input in tqdm(train_dataloader):
        x_input = x_input.to(device)

        start_index = 2 * (opt.filter_size - 1)
        end_index = -1 * opt.pred_step
        for i in range(opt.num_critic):
            optimizer_discriminator.zero_grad()

            inn = encoder(x_input)
            x_recons = decoder(inn)
            x_recons[:, :, :end_index] = x_input[:, :, start_index:end_index]
            inn_real = (2 * torch.rand(inn.shape) - 1).to(device)

            inn_fake_output = inn_discriminator(inn)
            inn_real_output = inn_discriminator(inn_real)
            inn_score_real = inn_real_output.mean()
            inn_score_fake = inn_fake_output.mean()

            recons_fake_output = recons_discriminator(x_recons)
            remaining_length = x_recons.shape[2]
            if opt.univariate:
                recons_real_output = recons_discriminator(
                    x_input[:, 0, -remaining_length:].unsqueeze(1)
                )
            else:
                recons_real_output = recons_discriminator(
                    x_input[:, :, -remaining_length:]
                )
            recons_score_real = recons_real_output.mean()
            recons_score_fake = recons_fake_output.mean()

            inn_gradient_penalty = calculate_gradient_penalty(
                inn_discriminator, inn_real, inn
            )
            if opt.univariate:
                recons_gradient_penalty = calculate_gradient_penalty(
                    recons_discriminator,
                    x_input[:, 0, -remaining_length:].unsqueeze(1),
                    x_recons,
                )
            else:
                recons_gradient_penalty = calculate_gradient_penalty(
                    recons_discriminator, x_input[:, :, -remaining_length:], x_recons
                )

            loss_discriminator = (
                inn_score_fake
                - inn_score_real
                + opt.gp_coef_inn * inn_gradient_penalty
                + opt.coef_recons
                * (
                    recons_score_fake
                    - recons_score_real
                    + opt.gp_coef_recons * recons_gradient_penalty
                )
            )

            loss_D.append(loss_discriminator.item())

            loss_discriminator.backward()
            optimizer_discriminator.step()

        # Train Generators
        optimizer_generator.zero_grad()

        inn = encoder(x_input)
        x_recons = decoder(inn)

        inn_fake_output = inn_discriminator(inn)
        recons_fake_output = recons_discriminator(x_recons)

        loss_generator = (
            -inn_fake_output.mean() - opt.coef_recons * recons_fake_output.mean()
        )

        # loss_generator = -torch.std(inn,dim=2).mean() - torch.std(x_recons,dim=2).mean()
        loss_generator.backward()
        optimizer_generator.step()

        loss_G.append(loss_generator.item())

    return sum(loss_G) / len(loss_G), sum(loss_D) / len(loss_D)


# def eval_epoch(
#     test_dataloader,
#     encoder,
#     decoder,
#     opt,
#     save_predict=False,
# ):
#     """
#     Evaluate one epoch: generate predictions and compute metrics using revised utils.metrics.
#     Returns:
#       all_mse: float            # Mean squared error (ACE)
#       all_mae: float            # Mean absolute error (ACE)
#       all_pred_mape: np.ndarray # Pointwise MAPE (ACE) shape (T,1)
#       all_acc_mean: float       # Direction accuracy of mean forecast (ACE)
#       all_acc_median: float     # Direction accuracy of median forecast (ACE)
#       all_acc_mape: float       # Interval accuracy MAPE (unused)
#       percentiles: dict         # P5, P25, P50, P75, P95 arrays of shape (T,1)
#       all_true: np.ndarray      # True values shape (T,1)
#       all_pred_mean: np.ndarray # Mean forecasts shape (T,1)
#       all_pred_median: np.ndarray # Median forecasts shape (T,1)
#     """
#     encoder.eval()
#     decoder.eval()
#     num_feature = 1 if opt.univariate else opt.num_feature
#
#     # Initialize with a dummy first row to simplify append logic
#     all_pred_mean = np.empty((1, num_feature))
#     all_pred_median = np.empty((1, num_feature))
#     all_true = np.empty((1, num_feature))
#     all_pred_all = np.empty((1, opt.sample_size, num_feature))
#
#     for x_input, x_true, x_mean, x_std in test_dataloader:
#         x_input = x_input.to(device)
#
#         # Unbatch true/statistics for univariate
#         if opt.univariate:
#             x_true = x_true[:, 0, :].unsqueeze(1)
#             x_mean = x_mean[:, 0].unsqueeze(1)
#             x_std = x_std[:, 0].unsqueeze(1)
#
#         # Encode to latent, convert to NumPy for sampling
#         inn = encoder(x_input).detach().cpu().numpy()
#         step = opt.pred_step
#
#         batch = inn.shape[0]
#         # Preallocate per-batch arrays
#         x_pred_mean = np.empty((batch, num_feature))
#         x_pred_median = np.empty((batch, num_feature))
#         x_pred_all = np.empty((batch, opt.sample_size, num_feature))
#
#         # Monte Carlo sampling loop
#         for row in range(batch):
#             inn_rep = np.tile(inn[row], (opt.sample_size, 1, 1))
#             inn_rep[:, :, -step:] = np.random.uniform(
#                 -1.0, 1.0,
#                 size=(opt.sample_size, num_feature, step)
#             )
#             dec_out = decoder(
#                 torch.tensor(inn_rep, dtype=torch.float32).to(device)
#             ).detach().cpu().numpy()
#             last = dec_out[:, :, -1]
#             x_pred_mean[row] = np.mean(last, axis=0)
#             x_pred_median[row] = np.median(last, axis=0)
#             x_pred_all[row] = last
#
#         # De-normalize predictions
#         mean_arr = x_mean.detach().cpu().numpy()
#         std_arr = x_std.detach().cpu().numpy()
#         x_pred_mean = x_pred_mean * std_arr + mean_arr
#         x_pred_median = x_pred_median * std_arr + mean_arr
#         x_pred_all = x_pred_all * std_arr[:, None, :] + mean_arr[:, None, :]
#
#         # Append to full arrays
#         all_pred_mean = np.append(all_pred_mean, x_pred_mean, axis=0)
#         all_pred_median = np.append(all_pred_median, x_pred_median, axis=0)
#         all_true = np.append(all_true, x_true, axis=0)
#         all_pred_all = np.append(all_pred_all, x_pred_all, axis=0)
#
#     # Drop dummy first row
#     all_pred_mean = all_pred_mean[1:]
#     all_pred_median = all_pred_median[1:]
#     all_true = all_true[1:]
#     all_pred_all = all_pred_all[1:]
#
#     # Compute metrics for ACE (feature 0)
#     mse, mae, pred_mape, acc_mean, acc_median, acc_mape, percentiles = metrics(
#         all_true[:, 0:1],
#         all_pred_mean[:, 0:1],
#         all_pred_median[:, 0:1],
#         all_pred_all[:, :, 0:1],
#         opt.pred_step
#     )
#
#     # Round and extract results
#     all_mse = round(float(mse), 4)
#     all_mae = round(float(mae), 4)
#     all_pred_mape = np.round(pred_mape, 4)
#     all_acc_mean = round(float(acc_mean), 4)
#     all_acc_median = round(float(acc_median), 4)
#     all_acc_mape = round(float(acc_mape), 4)
#
#     if save_predict:
#         if opt.univariate:
#             median_fig_name = "Revised_{}_{}_univariate/Median_lrG_{}gp_coef_inn_{}gp_coef_recons_{}coef_recons_{}seed_{}UC90_{}UC50_{}UC10_{}W90_{}W50_{}W10_{}.jpg".format(
#                 opt.dataset,
#                 opt.pred_step,
#                 opt.lrG,
#                 opt.gp_coef_inn,
#                 opt.gp_coef_recons,
#                 opt.coef_recons,
#                 opt.seed,
#                 all_uc_90,
#                 all_uc_50,
#                 all_uc_10,
#                 all_pinaw_90,
#                 all_pinaw_50,
#                 all_pinaw_10,
#             )
#
#             mean_fig_name = "Revised_{}_{}_univariate/Mean_lrG_{}gp_coef_inn_{}gp_coef_recons_{}coef_recons_{}seed_{}MSE_{}MAE_{}MAPE_{}MASE_{}CRPS_{}CD_{}.jpg".format(
#                 opt.dataset,
#                 opt.pred_step,
#                 opt.lrG,
#                 opt.gp_coef_inn,
#                 opt.gp_coef_recons,
#                 opt.coef_recons,
#                 opt.seed,
#                 all_mse,
#                 all_mae,
#                 all_mape,
#                 all_mase,
#                 all_crps,
#                 all_cd,
#             )
#             path = "{}_{}_univariate".format(opt.dataset, opt.pred_step)
#         else:
#             median_fig_name = "Revised_{}_{}/Median_lrG_{}gp_coef_inn_{}gp_coef_recons_{}coef_recons_{}seed_{}UC90_{}UC50_{}UC10_{}W90_{}W50_{}W10_{}.jpg".format(
#                 opt.dataset,
#                 opt.pred_step,
#                 opt.lrG,
#                 opt.gp_coef_inn,
#                 opt.gp_coef_recons,
#                 opt.coef_recons,
#                 opt.seed,
#                 all_uc_90,
#                 all_uc_50,
#                 all_uc_10,
#                 all_pinaw_90,
#                 all_pinaw_50,
#                 all_pinaw_10,
#             )
#
#             mean_fig_name = "Revised_{}_{}/Mean_lrG_{}gp_coef_inn_{}gp_coef_recons_{}coef_recons_{}seed_{}MSE_{}MAE_{}MAPE_{}MASE_{}CRPS_{}CD_{}.jpg".format(
#                 opt.dataset,
#                 opt.pred_step,
#                 opt.lrG,
#                 opt.gp_coef_inn,
#                 opt.gp_coef_recons,
#                 opt.coef_recons,
#                 opt.seed,
#                 all_mse,
#                 all_mae,
#                 all_mape,
#                 all_mase,
#                 all_crps,
#                 all_cd,
#             )
#             path = "Revised_{}_{}".format(opt.dataset, opt.pred_step)
#         if not os.path.exists(path):
#             os.mkdir(path)
#
#
#         all_true = all_true[1:, :]
#         all_pred_mean = all_pred_mean[1:, :]
#         all_pred_median = all_pred_median[1:, :]
#         all_pred_all = all_pred_all[1:, :, :]
#
#         if opt.dataset != "CAISO_both":
#
#             plt.figure()
#             plt.plot(all_true[:, 0], label="Ground Truth")
#             plt.plot(all_pred_mean[:, 0], label="Mean Estimation")
#             plt.legend()
#             plt.savefig(mean_fig_name)
#             plt.close()
#
#             plt.figure()
#             plt.plot(all_true[:, 0], label="Ground Truth")
#             plt.plot(all_pred_median[:, 0], label="Mean Estimation")
#             plt.legend()
#             plt.savefig(median_fig_name)
#             plt.close()
#
#         if opt.dataset == "CAISO_both":
#             plt.figure()
#             plt.plot(all_true[1:, 0], label="Ground Truth")
#             plt.plot(all_pred_median[1:, 0], label="Median Estimation")
#             plt.xlabel("Time Step")
#             plt.ylabel("ACE (MW)")
#             plt.legend()
#             plt.savefig(mean_fig_name)
#             plt.close()
#
#             ace = all_pred_all[:, :, 0]
#             lin_freq = b0 + b1 * ace
#             median_lin_freq = np.median(lin_freq, axis=1)  # shape: (T,)
#
#             plt.figure()
#             plt.plot(all_true[1:, 1], label="Ground Truth")
#             plt.plot(all_pred_median[:, 1], label="Median Estimation")
#             plt.plot(median_lin_freq[1:],    label="Linear Estimation")
#             plt.xlabel("Time Step")
#             plt.ylabel("Frequency (Hz)")
#             plt.legend()
#             plt.savefig(median_fig_name)
#             plt.close()
#
#         # # generate histogram
#         # single_timestep_forecast = all_pred_all[1, :, 0]
#         # actual_value = all_true[1, 0]
#         # plt.hist(single_timestep_forecast, bins=30, color='skyblue', edgecolor='black')
#         # plt.xlabel('ACE Values (MW)')
#         # plt.ylabel('Count')
#         # plt.title(f'Histogram of Decoder Output (Actual={actual_value:.2f} MW)')
#         #
#         # plt.savefig(mean_fig_name.replace("Mean", "Histogram"))
#         # plt.close()
#         # print(all_pred_all)
#         # print(all_pred_all.shape)
#         # print(all_pred_all[1, :, 1]) # take 1D slice from 3D array. [1 because first entry (entry 0) is random, all 1000 samples, 0 = ACE, 1 = Frequency]
#
#         # print(all_true)
#         # print(all_true.shape)
#
#         if opt.dataset == "CAISO_ACE":
#             single_timestep_forecast = all_pred_all[0, :, 0]
#             actual_value = all_true[15, 0]
#
#             print("all_true")
#             print(all_true[15, 0])
#             print("all_pred")
#             print(all_pred_median[0, 0])
#             print(all_pred_mean[0, 0])
#
#             plt.hist(single_timestep_forecast, bins=30, color='skyblue', edgecolor='black')
#             plt.xlabel('ACE Values (MW)')
#             plt.ylabel('Count')
#             plt.title(f'Histogram of Decoder Output (Actual={actual_value:.2f} MW)')
#
#             plt.savefig(mean_fig_name.replace("Mean", "Histogram"))
#             plt.close()
#
#         if opt.dataset == "CAISO_FREQUENCY":
#             single_timestep_forecast = all_pred_all[151, :, 0]
#             num_below = np.sum(single_timestep_forecast < 59.95)
#             num_above = np.sum(single_timestep_forecast > 60.05)
#             num_within = single_timestep_forecast.size - num_below - num_above
#
#             percent_below = num_below / single_timestep_forecast.size * 100
#             percent_above = num_above / single_timestep_forecast.size * 100
#             percent_within = num_within / single_timestep_forecast.size * 100
#
#             actual_value = all_true[151, 0]
#             plt.hist(single_timestep_forecast, bins=30, color='skyblue', edgecolor='black')
#             plt.xlabel(f'Frequency (HZ), Actual = {actual_value: .3f} (HZ)')
#             plt.ylabel('Count')
#             plt.title(f'<59.95 = {percent_below:.2f}%, 59.95-60.05 = {percent_within:.2f} %, >60.05 = {percent_above:.2f}%')
#
#             plt.savefig(mean_fig_name.replace("Mean", "Histogram"))
#             plt.close()
#
#     return (
#         all_mse,  # float
#         all_mae,  # float
#         all_pred_mape,  # array (T×1)
#         all_acc_mean,  # float
#         all_acc_median,  # float
#         all_acc_mape,  # float (unused)
#         percentiles,  # dict of P5…P95 arrays
#         all_true,  # array (T×1)
#         all_pred_mean,  # array (T×1)
#         all_pred_median  # array (T×1)
#     )

def eval_epoch(
    test_dataloader,
    encoder,
    decoder,
    opt,
    save_predict=False,
):
    """
    Evaluate one epoch: generate predictions and compute metrics using revised utils.metrics.
    Returns:
      all_mse: float            # Mean squared error (ACE)
      all_mae: float            # Mean absolute error (ACE)
      all_pred_mape: np.ndarray # Pointwise MAPE (ACE) shape (T,1)
      all_acc_mean: float       # Direction accuracy of mean forecast (ACE)
      all_acc_median: float     # Direction accuracy of median forecast (ACE)
      all_acc_mape: float       # Interval accuracy MAPE (unused)
      percentiles: dict         # P5, P25, P50, P75, P95 arrays of shape (T,1)
      all_true: np.ndarray      # True values shape (T,1)
      all_pred_mean: np.ndarray # Mean forecasts shape (T,1)
      all_pred_median: np.ndarray # Median forecasts shape (T,1)
    """
    encoder.eval()
    decoder.eval()
    num_feature = 1 if opt.univariate else opt.num_feature

    # Initialize with a dummy first row to simplify append logic
    all_pred_mean = np.empty((1, num_feature))
    all_pred_median = np.empty((1, num_feature))
    all_true = np.empty((1, num_feature))
    all_pred_all = np.empty((1, opt.sample_size, num_feature))

    for x_input, x_true, x_mean, x_std in test_dataloader:
        x_input = x_input.to(device)

        # Unbatch true/statistics for univariate
        if opt.univariate:
            x_true = x_true[:, 0, :].unsqueeze(1)
            x_mean = x_mean[:, 0].unsqueeze(1)
            x_std = x_std[:, 0].unsqueeze(1)

        # Encode to latent, convert to NumPy for sampling
        inn = encoder(x_input).detach().cpu().numpy()
        step = opt.pred_step

        batch = inn.shape[0]
        # Preallocate per-batch arrays
        x_pred_mean = np.empty((batch, num_feature))
        x_pred_median = np.empty((batch, num_feature))
        x_pred_all = np.empty((batch, opt.sample_size, num_feature))

        # Monte Carlo sampling loop
        for row in range(batch):
            inn_rep = np.tile(inn[row], (opt.sample_size, 1, 1))
            inn_rep[:, :, -step:] = np.random.uniform(
                -1.0, 1.0,
                size=(opt.sample_size, num_feature, step)
            )
            dec_out = decoder(
                torch.tensor(inn_rep, dtype=torch.float32).to(device)
            ).detach().cpu().numpy()
            last = dec_out[:, :, -1]
            x_pred_mean[row] = np.mean(last, axis=0)
            x_pred_median[row] = np.median(last, axis=0)
            x_pred_all[row] = last

        # De-normalize predictions
        mean_arr = x_mean.detach().cpu().numpy()
        std_arr = x_std.detach().cpu().numpy()
        x_pred_mean = x_pred_mean * std_arr + mean_arr
        x_pred_median = x_pred_median * std_arr + mean_arr
        x_pred_all = x_pred_all * std_arr[:, None, :] + mean_arr[:, None, :]

        # Append to full arrays
        all_pred_mean = np.append(all_pred_mean, x_pred_mean, axis=0)
        all_pred_median = np.append(all_pred_median, x_pred_median, axis=0)
        all_true = np.append(all_true, x_true, axis=0)
        all_pred_all = np.append(all_pred_all, x_pred_all, axis=0)

    # Drop dummy first row
    all_pred_mean = all_pred_mean[1:]
    all_pred_median = all_pred_median[1:]
    all_true = all_true[1:]
    all_pred_all = all_pred_all[1:]

    # Compute metrics for ACE (feature 0)
    mse, mae, pred_mape, acc_mean, acc_median, acc_mape, percentiles = metrics(
        all_true[:, 0:1],
        all_pred_mean[:, 0:1],
        all_pred_median[:, 0:1],
        all_pred_all[:, :, 0:1],
        opt.pred_step
    )

    # Round and extract results
    all_mse = round(float(mse), 4)
    all_mae = round(float(mae), 4)
    all_pred_mape = np.round(pred_mape, 4)
    all_acc_mean = round(float(acc_mean), 4)
    all_acc_median = round(float(acc_median), 4)
    all_acc_mape = round(float(acc_mape), 4)

    # Optional: save plots if requested
    if save_predict:
        # (keep existing plotting code)
        pass

    return (
        all_mse,
        all_mae,
        all_pred_mape,
        all_acc_mean,
        all_acc_median,
        all_acc_mape,
        percentiles,
        all_true,
        all_pred_mean,
        all_pred_median,
    )


def main(opt):
    if opt.univariate:
        encoder = Generator(opt.num_feature, 1, opt.filter_size, opt.seq_len, "encoder").to(device)
        decoder = Generator(1, 1, opt.filter_size, opt.seq_len, "decoder").to(device)
        inn_discriminator = Discriminator((opt.seq_len - opt.filter_size + 1), opt.hidden_dim).to(device)
        recons_discriminator = Discriminator((opt.seq_len - 2*(opt.filter_size - 1)), opt.hidden_dim).to(device)
    else:
        encoder = Generator(opt.num_feature, opt.num_feature, opt.filter_size, opt.seq_len, "encoder").to(device)
        decoder = Generator(opt.num_feature, opt.num_feature, opt.filter_size, opt.seq_len, "decoder").to(device)
        inn_discriminator = Discriminator((opt.seq_len - opt.filter_size + 1)*opt.num_feature, opt.hidden_dim).to(device)
        recons_discriminator = Discriminator((opt.seq_len - 2*(opt.filter_size - 1))*opt.num_feature, opt.hidden_dim).to(device)

    optimizer_generator = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=opt.lrG
    )
    optimizer_discriminator = torch.optim.Adam(
        list(recons_discriminator.parameters()) + list(inn_discriminator.parameters()), lr=opt.lrD
    )

    train_data = Custom_Dataset(opt.seq_len, opt.data_path, opt.dataset, "train", opt.seq_len)
    test_data  = Custom_Dataset(opt.seq_len, opt.data_path, opt.dataset, "test", opt.seq_len, opt.filter_size)

    # optional linear benchmark
    global b0, b1
    if opt.dataset == "CAISO_both":
        raw = test_data.test_data.cpu().numpy()
        ace, freq = raw[0,:], raw[1,:]
        b0, b1 = estimate_coef(ace, freq)
        print(f"[Linear benchmark] Frequency = {b1:.6f} * ACE + {b0:.6f}")

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=opt.batch_size, shuffle=False)

    iter_best_mse = float("inf")
    iter_best_mae = float("inf")
    iter_best_mape = float("inf")
    iter_best_acc_mean = 0.0
    iter_best_acc_median = 0.0

    for epoch in range(1, opt.epochs + 1):
        loss_G, loss_D = train_epoch(
            train_loader, encoder, decoder,
            inn_discriminator, recons_discriminator,
            optimizer_generator, optimizer_discriminator,
            opt
        )
        print(f"Epoch {epoch}: Gen Loss={loss_G:.6f}, Disc Loss={loss_D:.6f}")

        mse, mae, pointwise_mape, acc_mean, acc_median, acc_mape, percentiles, \
        all_true, all_pred_mean, all_pred_median = eval_epoch(
            test_loader, encoder, decoder, opt, False
        )

        mean_pointwise_mape = float(np.nanmean(pointwise_mape))

        print(
            f"Test - MSE: {mse:.4f}, "
            f"MAE: {mae:.4f}, "
            f"PtMAPE: {mean_pointwise_mape:.4f}, "
            f"DirAcc(mean): {acc_mean:.2%}, "
            f"DirAcc(med): {acc_median:.2%}"
        )

        p5, p95 = percentiles["P5"], percentiles["P95"]
        avg_width = float(np.nanmean(p95 - p5))
        print(f"Avg 90% PI width: {avg_width:.4f}")

        if mse < iter_best_mse:
            eval_epoch(test_loader, encoder, decoder, opt, save_predict=True)

        iter_best_mse = min(iter_best_mse, mse)
        iter_best_mae = min(iter_best_mae, mae)
        iter_best_mape = min(iter_best_mape, mean_pointwise_mape)
        iter_best_acc_mean = max(iter_best_acc_mean, acc_mean)
        iter_best_acc_median = max(iter_best_acc_median, acc_median)

        print(
            f"Best so far (epoch {epoch}) - "
            f"MSE: {iter_best_mse:.4f}, "
            f"MAE: {iter_best_mae:.4f}, "
            f"PtMAPE: {iter_best_mape:.4f}, "
            f"DirAcc(mean): {iter_best_acc_mean:.2%}, "
            f"DirAcc(med): {iter_best_acc_median:.2%}"
        )

    return iter_best_mse, iter_best_mae, iter_best_mape, iter_best_acc_mean, iter_best_acc_median



# def main(opt):
#     if opt.univariate:
#         encoder = Generator(opt.num_feature, 1, opt.filter_size, opt.seq_len, "encoder").to(device)
#         decoder = Generator(1, 1, opt.filter_size, opt.seq_len, "decoder").to(device)
#         inn_discriminator = Discriminator(
#             (opt.seq_len - opt.filter_size + 1), opt.hidden_dim
#         ).to(device)
#         recons_discriminator = Discriminator(
#             (opt.seq_len - 2 * (opt.filter_size - 1)), opt.hidden_dim
#         ).to(device)
#     else:
#         encoder = Generator(
#             opt.num_feature, opt.num_feature, opt.filter_size, opt.seq_len, "encoder"
#         ).to(device)
#         decoder = Generator(
#             opt.num_feature, opt.num_feature, opt.filter_size, opt.seq_len, "decoder"
#         ).to(device)
#         inn_discriminator = Discriminator(
#             (opt.seq_len - opt.filter_size + 1) * opt.num_feature, opt.hidden_dim
#         ).to(device)
#         recons_discriminator = Discriminator(
#             (opt.seq_len - 2 * (opt.filter_size - 1)) * opt.num_feature, opt.hidden_dim
#         ).to(device)
#     optimizer_generator = torch.optim.Adam(
#         list(encoder.parameters()) + list(decoder.parameters()),
#         lr=opt.lrG,
#     )
#     optimizer_discriminator = torch.optim.Adam(
#         list(recons_discriminator.parameters()) + list(inn_discriminator.parameters()),
#         lr=opt.lrD,
#     )
#     train_data = Custom_Dataset(
#         opt.seq_len, opt.data_path, opt.dataset, "train", opt.seq_len
#     )
#     test_data = Custom_Dataset(
#         opt.seq_len, opt.data_path, opt.dataset, "test", opt.seq_len, opt.filter_size
#     )
#     # linear‐regression baseline on raw (ACE, Frequency)
#     global b0, b1
#     if opt.dataset == "CAISO_both":
#         # test_data.test_data is a torch.Tensor of shape [num_features, T]
#         raw = test_data.test_data.cpu().numpy()
#         ace = raw[0, :]  # ACE is channel 0
#         freq = raw[1, :]  # Frequency is channel 1
#         b0, b1 = estimate_coef(ace, freq)
#         print(f"[Linear benchmark] Frequency = {b1:.6f} * ACE + {b0:.6f}")
#
#
#     train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
#     test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
#     iter_best_mse = float("inf")
#     iter_best_mae = float("inf")
#     iter_best_median_se = float("inf")
#     iter_best_median_ae = float("inf")
#     iter_best_mape = float("inf")
#     iter_best_mase = float("inf")
#     iter_best_crps = float("inf")
#     for i in range(opt.epochs):
#         loss_G, loss_D = train_epoch(
#             train_dataloader,
#             encoder,
#             decoder,
#             inn_discriminator,
#             recons_discriminator,
#             optimizer_generator,
#             optimizer_discriminator,
#             opt,
#         )
#         print(
#             "Epoch {}: Generator Loss: {}, Discriminator Loss:{}".format(
#                 i + 1, loss_G, loss_D
#             )
#         )
#         if (i + 1) % 1 == 0:
#             mse, mae, pred_mape, acc_mean, acc_median, acc_mape, percentiles, all_true, all_pred_mean, all_pred_median = eval_epoch(
#                 test_dataloader,
#                 encoder,
#                 decoder,
#                 opt,
#                 False,
#             )
#
#             mean_pointwise_mape = float(np.nanmean(pred_mape))
#
#             # print(
#             #     "Test result-MSE:{}, MAE:{}, Median SE:{}, Median AE:{}, MAPE:{}, MASE:{},CRPS:{}".format(
#             #         mse,
#             #         mae,
#             #         median_se,
#             #         median_ae,
#             #         mape,
#             #         mase,
#             #         crps_score,
#             #     )
#
#             print(
#                 f"Test result — MSE: {mse:.4f}, MAE: {mae:.4f}, "
#                 f"Mean point-wise MAPE: {mean_pointwise_mape:.4f}, "
#                 f"DirAcc (mean): {acc_mean:.2%}, "
#                 f"DirAcc (median): {acc_median:.2%}"
#             )
#             p5 = percentiles['P5']  # shape (T,1)
#             p95 = percentiles['P95']
#             avg_width = float(np.nanmean(p95 - p5))
#             print(f"Avg 90% PI width: {avg_width:.4f}")
#
#             if opt.dataset == "CAISO_FREQUENCY":
#                 true_vals = all_true[:, 0]  # 1D array of non-normalized values
#                 mean_vals = all_pred_mean[:, 0]
#                 median_vals = all_pred_median[:, 0]
#
#                 true_dir = true_vals > 60.0
#                 mean_dir = mean_vals > 60.0
#                 median_dir = median_vals > 60.0
#
#                 correct_mean = (mean_dir == true_dir)
#                 correct_median = (median_dir == true_dir)
#
#                 acc_mean = correct_mean.mean()
#                 acc_median = correct_median.mean()
#
#                 print(f"Direction accuracy (mean):   {acc_mean:.1%}")
#                 print(f"Direction accuracy (median): {acc_median:.1%}")
#
#                 # print("true_vals" + str(true_vals))
#                 # print("all_true" + str(all_true))
#
#             if opt.dataset == "CAISO_both":
#                 # all_true = all_true[1:, :]
#                 # all_pred_median = all_pred_median[1:, :]
#
#                 # print(all_true)
#                 # print(all_true.shape)
#                 # print(all_pred_median.shape)
#
#                 true_ACE = all_true[:, 0]
#                 true_frequency = all_true[:, 1]
#
#                 median_ACE = all_pred_median[:, 0]
#                 median_frequency = all_pred_median[:, 1]
#
#                 true_ACE_direction = true_ACE > 0.0
#                 forecast_ACE_direction = median_ACE > 0.0 # Array where values > 0 are true and <= 0 are false
#
#                 true_frequency_direction = true_frequency > 60.0
#                 forecast_frequency_direction = median_frequency > 60.0
#
#                 correct_ACE_direction = (forecast_ACE_direction == true_ACE_direction)
#                 correct_frequency_direction = (forecast_frequency_direction == true_frequency_direction)
#
#                 accuracy_ACE = correct_ACE_direction.mean()
#                 accuracy_frequency = correct_frequency_direction.mean()
#
#                 print(f"Directional Accuracy ACE = {accuracy_ACE: .1%}")
#                 print(f"Directional Accuracy Frequency = {accuracy_frequency: .1%}")
#
#                 lin_freq_samples = b0 + b1 * median_ACE
#                 lin_freq_direction = lin_freq_samples > 60.0
#
#                 correct_lin_direction = (lin_freq_direction == true_frequency_direction)
#                 accuracy_lin = correct_lin_direction.mean()
#                 print(f"Directional Accuracy Linear Frequency = {accuracy_lin: .1%}")
#
#                 # print(lin_freq_samples)
#                 # print(lin_freq_samples.shape)
#                 # median_lin_freq = np.median(lin_freq_samples, axis=1)  # (T,)
#
#                 # align with true_frequency_direction (which you computed from all_true[1:,1])
#                 # you did `all_true = all_true[1:,:]` and then `true_frequency = all_true[1:,1]`,
#                 # so drop the first two rows of median_lin_freq to match:
#                 # median_lin_freq = median_lin_freq[2:]  # now length matches true_frequency_direction
#                 #
#                 # lin_freq_direction = median_lin_freq > 60.0  # boolean array
#                 # correct_lin = lin_freq_direction == true_frequency_direction
#                 # accuracy_lin = correct_lin.mean()
#
#
#             if opt.dataset == "CAISO_ACE":
#                 true_vals = all_true[:, 0]  # 1D array of non-normalized values
#                 mean_vals = all_pred_mean[:, 0]
#                 median_vals = all_pred_median[:, 0]
#
#                 true_dir = true_vals > 0.0
#                 mean_dir = mean_vals > 0.0
#                 median_dir = median_vals > 0.0
#
#                 correct_mean = (mean_dir == true_dir)
#                 correct_median = (median_dir == true_dir)
#
#                 acc_mean = correct_mean.mean()
#                 acc_median = correct_median.mean()
#
#                 print(f"Direction accuracy (mean):   {acc_mean:.1%}")
#                 print(f"Direction accuracy (median): {acc_median:.1%}")
#
#                 # print("true_vals" + str(true_vals))
#                 # print("all_true" + str(all_true))
#
#
#             if mse < iter_best_mse:
#                 epoch = i + 1
#                 eval_epoch(
#                     test_dataloader,
#                     encoder,
#                     decoder,
#                     opt,
#                     save_predict=True,
#                 )
#             iter_best_mse = min(iter_best_mse, mse)
#             iter_best_mae = min(iter_best_mae, mae)
#             iter_best_median_se = min(iter_best_median_se, median_se)
#             iter_best_median_ae = min(iter_best_median_ae, median_ae)
#             iter_best_mape = min(iter_best_mape, mape)
#             iter_best_mase = min(iter_best_mase, mase)
#             iter_best_crps = min(iter_best_crps, crps_score)
#
#             print(
#                 "Best Testing Results for this iteration at epoch {}, with MSE:{}, MAE:{}, Median SE:{}, Median AE:{}, MAPE:{}, MASE:{},CRPS:{}".format(
#                     epoch,
#                     iter_best_mse,
#                     iter_best_mae,
#                     iter_best_median_se,
#                     iter_best_median_ae,
#                     iter_best_mape,
#                     iter_best_mase,
#                     iter_best_crps,
#                 )
#             )
#
#     return iter_best_mse, iter_best_mae, iter_best_median_se, iter_best_median_ae


if __name__ == "__main__":
    opt = arguement()
    torch.manual_seed(opt.seed)

    print(
        "---------------------------------------------New Parameter Run----------------------------------------------"
    )
    print("[Info]-Dataset:{}, Prediction Step:{}".format(opt.dataset, opt.pred_step))
    main(opt)
