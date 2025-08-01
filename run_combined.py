import main
from types import SimpleNamespace
from datetime import datetime, timedelta
import torch

start = datetime.strptime("2025-04-03", "%Y-%m-%d")   # first day that *has* residuals
end   = datetime.strptime("2025-04-10", "%Y-%m-%d")

for d in range((end - start).days + 1):
    test_date = (start + timedelta(days=d)).strftime("%Y-%m-%d")
    print("Running for", test_date)

    opt = SimpleNamespace(
        data_path       = "data/filtered-price-linear",        # directory, not file
        dataset         = "pricing_residuals",       # <-- our new prepare_*
        pred_step       = 12,
        seq_len         = 80,
        # --- everything else unchanged ---
        filter_size     = 40,
        batch_size      = 60,
        epochs          = 100,
        lrD             = 1e-5,
        lrG             = 1e-6,
        num_critic      = 10,
        gp_coef_inn     = 0.1,
        coef_recons     = 0.1,
        gp_coef_recons  = 0.1,
        seed            = 1,
        sample_size     = 1000,
        univariate      = False,
        date            = test_date,

        data_bad=None,
        degree=4,
        block=100,
        stride=100,
        output_dim=1,
        hidden_dim=100,
        num_feature=3,

    )

    torch.manual_seed(opt.seed)
    main.main(opt)
