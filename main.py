import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import Generator, Discriminator
from data_loader import Custom_Dataset
# from utils import calculate_gradient_penalty, metrics
from utils_revised import calculate_gradient_penalty, metrics, add_linear_means
import Linear_Predictor as lp
import combined_prediction as comb_pred

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

def arguement():
    parser = argparse.ArgumentParser()

    parser.add_argument("-data_path", type=str, required=True)
    parser.add_argument("-dataset", type=str, required=True)
    parser.add_argument("-data_bad", type=str, required=False)
    parser.add_argument("--dates", type=str, nargs="+", required=False)
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


def eval_epoch(
    test_dataloader,
    encoder,
    decoder,
    opt,
    epoch_num,
    save_predict=False,
):
    encoder.eval()
    decoder.eval()
    if opt.univariate:
        num_feature = 1
    else:
        num_feature = opt.num_feature
    all_pred_mean = np.empty((1, num_feature))
    all_pred_median = np.empty((1, num_feature))
    all_true = np.empty((1, num_feature))
    all_pred_all = np.empty((1, opt.sample_size, num_feature))
    for x_input, x_true, x_mean, x_std in test_dataloader:
        x_input = x_input.to(device)    # move input to GPU

        if opt.univariate:
            x_true = x_true[:, 0, :].unsqueeze(1)
            x_mean = x_mean[:, 0].unsqueeze(1)
            x_std = x_std[:, 0].unsqueeze(1)
        inn = encoder(x_input)
        inn = inn.detach().cpu().numpy()      # convert to NumPy for sampling
        step = opt.pred_step
        decoder_in_len = opt.seq_len - 2 * opt.filter_size + 2  # 12

        # x_pred_mean = decoder(torch.tensor(inn)).detach().numpy()
        # x_pred_median = decoder(torch.tensor(inn)).detach().numpy()

        x_pred_median = np.empty((inn.shape[0], num_feature))
        x_pred_mean = np.empty((inn.shape[0], num_feature))
        x_pred_all = np.empty((inn.shape[0], opt.sample_size, num_feature))
        for row in range(inn.shape[0]):
            inn_test_temp = np.tile(inn[row, :, :].copy(), (opt.sample_size, 1, 1))
            inn_test_temp[:, :, -1 * step :] = np.random.uniform(
                low=-1.0, high=1.0, size=(opt.sample_size, num_feature, step)
            )
            decoder_out = decoder(torch.tensor(inn_test_temp, dtype=torch.float32).to(device))  # On GPU
            decoder_out = decoder_out.detach().cpu().numpy()  # back to CPU
            x_pred_median[row, :] = np.median(decoder_out[:, :, -1], axis=0)
            x_pred_mean[row, :] = np.mean(decoder_out[:, :, -1], axis=0)
            x_pred_all[row, :, :] = decoder_out[:, :, -1]
        # Take only some rows for evaluation

        x_pred_mean = x_pred_mean * x_std.detach().numpy() + x_mean.detach().numpy()
        x_pred_median = x_pred_median * x_std.detach().numpy() + x_mean.detach().numpy()
        x_pred_all = x_pred_all * np.expand_dims(
            x_std.detach().numpy(), axis=(1,)
        ) + np.expand_dims(x_mean.detach().numpy(), axis=(1,))

        all_pred_mean = np.append(
            all_pred_mean,
            x_pred_mean,
            axis=0,
        )
        all_pred_median = np.append(
            all_pred_median,
            x_pred_median,
            axis=0,
        )
        all_true = np.append(all_true, x_true, axis=0)
        all_pred_all = np.append(all_pred_all, x_pred_all, axis=0)

    MAX_V = np.max(all_true, axis=0)[0]
    MIN_V = np.min(all_true, axis=0)[0]

    # print("\n► WIAE raw samples  shape:", all_pred_all.shape)
    # print("  first sample, first 3 timesteps:\n",
    #       all_pred_all[:3, 0, :])  # 3×2
    # print("► WIAE mean  shape:", all_pred_mean.shape,
    #       "  first 3 rows:\n", all_pred_mean[:3])

    # ------------------------------------------------------------
    # HYBRID STEP: add linear means back to every residual sample
    # ------------------------------------------------------------
    if opt.dataset == "CAISO_RESIDUALS":        # only for the hybrid run
        full_samples = add_linear_means(all_pred_all, opt.date, opt.seq_len)     # (M,1000,2)
        all_pred_mean    = full_samples.mean(axis=1)                # (M,2)
        all_pred_median  = np.median(full_samples, axis=1)          # (M,2)
        all_pred_all     = full_samples                             # overwrite
        all_true = comb_pred.load_caiso_truth(csv_path="data/CAISO_both.csv",
                                    date=opt.date,
                                    seq_len=opt.seq_len,
                                    pred_step=opt.pred_step)
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # HYBRID STEP: add linear means back to every residual sample
    # ------------------------------------------------------------
    if opt.dataset == "pricing_residuals":  # only for the hybrid run
        full_samples = comb_pred.add_pricing_linear_means(all_pred_all, opt.seq_len, opt.pred_step, decoder_in_len, opt.filter_size, csv_path="2023-07-26-filtered-price-linear_linear_preds.csv")  # (M,1000,2)
        all_pred_mean = full_samples.mean(axis=1)  # (M,2)
        all_pred_median = np.median(full_samples, axis=1)  # (M,2)
        all_pred_all = full_samples  # overwrite
        all_true = comb_pred.load_price_truth(all_pred_all, csv_path="data/NYISO_Jul_RTDA_Load.csv")
        # comb_pred.export_csv(all_true, all_pred_median, "median_vs_true_data_seq_len_36.csv")

        all_pred_mean = all_pred_mean[1:, :]
        all_pred_median = all_pred_median[1:, :]
        all_pred_all = all_pred_all[1:, :, :]
        all_true = all_true[1:, :]
    # ------------------------------------------------------------

    (mse, mae, pred_mape, accuracy_mean, accuracy_median, accuracy_mape,percentiles_dict) = metrics(all_true, all_pred_mean, all_pred_median, all_pred_all, step)

    all_mse = np.round(mse, 4)
    all_mae = np.round(mae, 4)
    all_accuracy_mean = np.round(accuracy_mean, 4)
    all_accuracy_median = np.round(accuracy_median, 4)
    all_accuracy_mape = np.round(accuracy_mape, 4)

    if save_predict:
        ACE_median_fig_name = "{}_{}_{}/ACE_Median_epoch{}_lrG_{}gp_coef_inn_{}gp_coef_recons_{}coef_recons_{}seed_{}MAE_{}Accuracy_{}.jpg".format(
            opt.date,
            opt.dataset,
            opt.pred_step,
            epoch_num,
            opt.lrG,
            opt.gp_coef_inn,
            opt.gp_coef_recons,
            opt.coef_recons,
            opt.seed,
            all_mae[0],
            all_accuracy_median[0],
        )

        Fre_median_fig_name = "{}_{}_{}/Fre_Median_epoch{}_lrG_{}gp_coef_inn_{}gp_coef_recons_{}coef_recons_{}seed_{}MAE_{}Accuracy_{}.jpg".format(
            opt.date,
            opt.dataset,
            opt.pred_step,
            epoch_num,
            opt.lrG,
            opt.gp_coef_inn,
            opt.gp_coef_recons,
            opt.coef_recons,
            opt.seed,
            all_mae[1],
            all_accuracy_median[1],
        )

        ACE_mean_fig_name = "{}_{}_{}/ACE_Mean_epoch_{}lrG_{}gp_coef_inn_{}gp_coef_recons_{}coef_recons_{}seed_{}MSE_{}Accuracy_{}.jpg".format(
            opt.date,
            opt.dataset,
            opt.pred_step,
            epoch_num,
            opt.lrG,
            opt.gp_coef_inn,
            opt.gp_coef_recons,
            opt.coef_recons,
            opt.seed,
            all_mse[0],
            all_accuracy_mean[0],
        )

        Fre_mean_fig_name = "{}_{}_{}/Fre_ACE_Mean_epoch_{}lrG_{}gp_coef_inn_{}gp_coef_recons_{}coef_recons_{}seed_{}MSE_{}Accuracy_{}.jpg".format(
            opt.date,
            opt.dataset,
            opt.pred_step,
            epoch_num,
            opt.lrG,
            opt.gp_coef_inn,
            opt.gp_coef_recons,
            opt.coef_recons,
            opt.seed,
            all_mse[1],
            all_accuracy_mean[1],
        )

        # mape_fig_name = "{}_{}_{}/MAPE_lrG_{}gp_coef_inn_{}gp_coef_recons_{}coef_recons_{}seed_{}Accuracy_{}.jpg".format(
        #     opt.date,
        #     opt.dataset,
        #     opt.pred_step,
        #     opt.lrG,
        #     opt.gp_coef_inn,
        #     opt.gp_coef_recons,
        #     opt.coef_recons,
        #     opt.seed,
        #     all_accuracy_mape,
        # )
        path = "{}_{}_{}".format(opt.date, opt.dataset, opt.pred_step)
        if not os.path.exists(path):
            os.mkdir(path)

        all_true = all_true[1:, :]
        all_pred_mean = all_pred_mean[1:, :]
        all_pred_median = all_pred_median[1:, :]
        # all_pred_mape = pred_mape[1:, :]
        # print(all_true)
        # print(all_true.shape)
        # print("all true")

        plt.figure()
        plt.plot(all_true[:, 0], label="Ground Truth")
        plt.plot(all_pred_mean[:, 0], label="Mean Estimation")
        plt.legend()
        plt.savefig(ACE_mean_fig_name)
        plt.close()

        plt.figure()
        plt.plot(all_true[:, 0], label="Ground Truth")
        plt.plot(all_pred_median[:, 0], label="Median Estimation")
        plt.legend()
        plt.savefig(ACE_median_fig_name)
        plt.close()

        plt.figure()
        plt.plot(all_true[:, 1], label="Ground Truth")
        plt.plot(all_pred_mean[:, 1], label="Mean Estimation")
        plt.legend()
        plt.savefig(Fre_mean_fig_name)
        plt.close()

        plt.figure()
        plt.plot(all_true[:, 1], label="Ground Truth")
        plt.plot(all_pred_median[:, 1], label="Median Estimation")
        plt.legend()
        plt.savefig(Fre_median_fig_name)
        plt.close()

        # plt.figure()
        # plt.plot(all_true[:, 0], label="Ground Truth")
        # plt.plot(all_pred_mape[:, 0], label="MAPE Estimation")
        # plt.legend()
        # plt.savefig(mape_fig_name)
        # plt.close()

        filename_data = "{}_{}_{}/epoch_{}MAE_{}.csv".format(
            opt.date,
            opt.dataset,
            opt.pred_step,
            epoch_num,
            all_mae,
        )

        df = pd.DataFrame({
            'Ground Truth ACE': all_true[:, 0],
            'Median Estimation ACE': all_pred_median[:, 0],
            'Mean Estimation ACE': all_pred_mean[:, 0],
            'P5 ACE': percentiles_dict['P5'][1:, 0],
            'P25 ACE': percentiles_dict['P25'][1:, 0],
            'P50 ACE': percentiles_dict['P50'][1:, 0],
            'P75 ACE': percentiles_dict['P75'][1:, 0],
            'P95 ACE': percentiles_dict['P95'][1:, 0],
            'Ground Truth Fre': all_true[:, 1],
            'Median Estimation Fre': all_pred_median[:, 1],
            'Mean Estimation Fre': all_pred_mean[:, 1],
            'P5 Fre': percentiles_dict['P5'][1:, 1],
            'P25 Fre': percentiles_dict['P25'][1:, 1],
            'P50 Fre': percentiles_dict['P50'][1:, 1],
            'P75 Fre': percentiles_dict['P75'][1:, 1],
            'P95 Fre': percentiles_dict['P95'][1:, 1],
        })
        df.to_csv(filename_data, index=False)

    return (
        all_mse,
        all_mae,
        all_accuracy_mean,
        all_accuracy_median,
        all_accuracy_mape
    )


def main(opt):
    if opt.univariate:
        encoder = Generator(opt.num_feature, 1, opt.filter_size, opt.seq_len, "encoder").to(device)
        decoder = Generator(1, 1, opt.filter_size, opt.seq_len, "decoder").to(device)
        inn_discriminator = Discriminator(
            (opt.seq_len - opt.filter_size + 1), opt.hidden_dim
        ).to(device)
        recons_discriminator = Discriminator(
            (opt.seq_len - 2 * (opt.filter_size - 1)), opt.hidden_dim
        ).to(device)
    else:
        encoder = Generator(
            opt.num_feature, opt.num_feature, opt.filter_size, opt.seq_len, "encoder"
        ).to(device)
        decoder = Generator(
            opt.num_feature, opt.num_feature, opt.filter_size, opt.seq_len, "decoder"
        ).to(device)
        inn_discriminator = Discriminator(
            (opt.seq_len - opt.filter_size + 1) * opt.num_feature, opt.hidden_dim
        ).to(device)
        recons_discriminator = Discriminator(
            (opt.seq_len - 2 * (opt.filter_size - 1)) * opt.num_feature, opt.hidden_dim
        ).to(device)
    optimizer_generator = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=opt.lrG,
    )
    optimizer_discriminator = torch.optim.Adam(
        list(recons_discriminator.parameters()) + list(inn_discriminator.parameters()),
        lr=opt.lrD,
    )
    train_data = Custom_Dataset(
        opt.seq_len, opt.data_path, opt.dataset, "train", opt.seq_len, date=opt.date
    )
    test_data = Custom_Dataset(
        opt.seq_len, opt.data_path, opt.dataset, "test", opt.seq_len, opt.filter_size, date=opt.date
    )

    # mu_x, mu_y, K, sigma_cond = lp.linear_distribution_terms(train_data.train_data, opt.seq_len, opt.pred_step)
    # lp.generate_linear_forecast(test_data.test_data, opt.seq_len, opt.pred_step, mu_x, mu_y, K, sigma_cond)
    # lp.naive_prediction(test_data.test_data, opt.seq_len, opt.pred_step)
    # comb_pred.output_linear_data(train_data.train_data, test_data.test_data, opt.seq_len, opt.pred_step, "2023-07-11-filtered-price-linear")

    # print("\n► TRAIN tensor")
    # print("  shape:", train_data.train_data.shape)
    # print("  first 5 rows:\n", train_data.train_data[:, :5])

    # print("\n► TEST tensor")
    # print("  shape:", test_data.test_data.shape)
    # print("  first 5 rows:\n", test_data.test_data[:, :5])

    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False)
    iter_best_mse = float("inf") * np.ones(opt.num_feature)
    iter_best_mae = float("inf") * np.ones(opt.num_feature)
    iter_best_accuracy_mean = float("-inf") * np.ones(opt.num_feature)
    iter_best_accuracy_median = float("-inf") * np.ones(opt.num_feature)
    for i in range(opt.epochs):
        loss_G, loss_D = train_epoch(
            train_dataloader,
            encoder,
            decoder,
            inn_discriminator,
            recons_discriminator,
            optimizer_generator,
            optimizer_discriminator,
            opt,
        )
        print(
            "Epoch {}: Generator Loss: {}, Discriminator Loss:{}".format(
                i + 1, loss_G, loss_D
            )
        )
        if (i + 1) % 1 == 0:
            mse, mae, accuracy_mean, accuracy_median, all_accuracy_mape = eval_epoch(
                test_dataloader,
                encoder,
                decoder,
                opt,
                i+1,
                False,
            )
            print(
                "Test result-MSE:{}, MAE:{}, Accuracy Mean:{}, Accuracy Median:{}, Accuracy MAPE:{}".format(
                    mse,
                    mae,
                    accuracy_mean,
                    accuracy_median,
                    all_accuracy_mape,
                )
            )

            if mse[0] < iter_best_mse[0]:
                epoch = i + 1
                eval_epoch(
                    test_dataloader,
                    encoder,
                    decoder,
                    opt,
                    i+1,
                    save_predict=True,
                )
            iter_best_mse = np.minimum(iter_best_mse, mse)
            # iter_best_mae = np.minimum(iter_best_mae, mae)
            # iter_best_median_se = min(iter_best_median_se, median_se)
            # iter_best_median_ae = min(iter_best_median_ae, median_ae)
            # iter_best_mape = min(iter_best_mape, mape)
            # iter_best_mase = min(iter_best_mase, mase)
            # iter_best_crps = min(iter_best_crps, crps_score)
            #
            # print(
            #     "Best Testing Results for this iteration at epoch {}, with MSE:{}, MAE:{}, Median SE:{}, Median AE:{}, MAPE:{}, MASE:{},CRPS:{}".format(
            #         epoch,
            #         iter_best_mse,
            #         iter_best_mae,
            #         iter_best_median_se,
            #         iter_best_median_ae,
            #         iter_best_mape,
            #         iter_best_mase,
            #         iter_best_crps,
            #     )
            # )

    return iter_best_mse, iter_best_mae


if __name__ == "__main__":
    opt = arguement()
    torch.manual_seed(opt.seed)

    print(
        "---------------------------------------------New Parameter Run----------------------------------------------"
    )
    print("[Info]-Dataset:{}, Prediction Step:{}".format(opt.dataset, opt.pred_step))
    main(opt)