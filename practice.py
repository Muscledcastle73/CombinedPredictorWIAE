# import pandas as pd
#
# # ————————————————————————————————
# # 1) Load the CSV
# # If you run this from Multivariate_WIAE-main/,
# # your data file lives in ./data/CAISO_ACE.csv
# csv_path = 'data/CAISO_ACE.csv'
# df = pd.read_csv(csv_path, parse_dates=['time'])
# ace = df['ACE'].tolist()
#
# # ————————————————————————————————
# # 2) Define or input your x‐values
#
# # Option A) Hard‐code them here:
# x_vals = [132, 125, 114, 88, 68]
#
#
# # ————————————————————————————————
# # 3) Scan for a contiguous match
# n = len(x_vals)
# found = False
#
# for i in range(len(ace) - n + 1):
#     if ace[i:i+n] == x_vals:
#         print("Found! First value occurs at time:", df.loc[i, 'time'])
#         found = True
#         break
#
# if not found:
#     print("Sequence not found in CAISO_ACE.csv.")

#
# import numpy as np
#
# matrix = np.array([[1, 3],
#                    [4, 6],
#                    [7, 8]])
#
#
# # print(np.zeros(matrix.shape[0]))
#
# print("hello")
#
# ACE_mean = np.mean(matrix[:, 0])
# print(ACE_mean)
#
# array = np.array([[1, 2, 3],
#                   [4, 5, 6],
#                   [7, 18, 9]])
#
#
# # Calculate the mean of the second column (index 1)
# column_mean = np.mean(array[:, 1])
#
# print("Mean of the second column:", column_mean)
#
#
# from math import erf, sqrt
# z_score = -1
# prob = 0.5 * (1.0 + erf(z_score / sqrt(2.0)))
#
# print(prob)

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# 1. file paths
# ------------------------------------------------------------------
DATA_DIR = Path(r"C:\Users\muscl\OneDrive\Documents\Multivariate_WIAE-main\data")
RAW_CSV  = DATA_DIR / "ISONE_price.csv"
CLEAN_CSV = DATA_DIR / "ISONE_price_cleaned.csv"

# ------------------------------------------------------------------
# 2. load & parse
# ------------------------------------------------------------------
df = (
    pd.read_csv(RAW_CSV)
      .rename(columns=lambda c: c.strip())              # trim stray spaces
)

# Parse the timestamp column and set it as the index
df["time"] = pd.to_datetime(df["time"])
df = (df
      .set_index("time")
      .sort_index()
)

# ------------------------------------------------------------------
# 3. drop exact duplicates (keep first)
# ------------------------------------------------------------------
df = df[~df.index.duplicated(keep="first")]

# ------------------------------------------------------------------
# 4. resample to exact 5-minute grid
#    • If you prefer forward-fill, replace .interpolate("time") with .ffill()
# ------------------------------------------------------------------
df_resampled = (
    df
    .resample("5min")            # produces a regular 5-minute DateTimeIndex
    .mean()                      # price columns: take mean if duplicates existed
    .interpolate("time")         # linearly fill any gaps
)

# ------------------------------------------------------------------
# 5. save
# ------------------------------------------------------------------
df_resampled.to_csv(CLEAN_CSV, float_format="%.4f")
print(f"Cleaned file written to:\n  {CLEAN_CSV.resolve()}")
print(f"Rows after cleaning: {len(df_resampled)}")
