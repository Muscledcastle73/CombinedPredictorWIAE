import main
from types import SimpleNamespace
from datetime import datetime, timedelta
import torch
import numpy as np

start_date = datetime.strptime("2023-01-02", "%Y-%m-%d")
end_date = datetime.strptime("2023-01-19", "%Y-%m-%d")

for i in range(0, (end_date - start_date).days + 1, 1):
    current_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
    print("Day: " + str(i))

# start_date = datetime.strptime("2025-04-05", "%Y-%m-%d")
# end_date = datetime.strptime("2025-04-02", "%Y-%m-%d")
# dates = [start_date, end_date]
# for current_date in dates:
#     current_date = current_date.strftime("%Y-%m-%d")

    opt = SimpleNamespace(
        data_path="data/NYISO_Jan_prices_cleaned.csv",
        dataset="CAISO_ACE_2D",
        # data_path="data/CAISO_both.csv",
        # dataset="CAISO_ACE_2D",
        data_bad=None,
        pred_step=12,
        degree=4,
        block=100,
        stride=100,
        output_dim=1,
        hidden_dim=100,
        seq_len=36,
        num_feature=1,
        filter_size=40,
        batch_size=60,
        epochs=100,
        lrD=0.00001,
        lrG=0.000001,
        num_critic=10,
        gp_coef_inn=0.1,
        coef_recons=0.1,
        gp_coef_recons=0.1,
        seed=1,
        sample_size=1000,
        univariate=False,
        date=current_date,
    )

    torch.manual_seed(opt.seed)
    # np.random.seed(opt.seed)
    main.main(opt)

