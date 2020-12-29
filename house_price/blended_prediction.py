import os
import pandas as pd
import numpy as np

lasso = pd.read_csv(os.path.dirname(__file__) + "/submission_lasso_v1.csv")
lgbm = pd.read_csv(os.path.dirname(__file__) + "/submission_lgbm_v1.csv")
rf = pd.read_csv(os.path.dirname(__file__) + "/submission_rf_v1.csv")
xgb = pd.read_csv(os.path.dirname(__file__) + "/submission_xgb_v1.csv")
svr = pd.read_csv(os.path.dirname(__file__) + "/submission_svr_v1.csv")
kernel_ridge = pd.read_csv(os.path.dirname(__file__) + "/submission_kernel_ridge_v1.csv")

combined_df = pd.concat([lasso, lgbm, rf, xgb, svr, kernel_ridge])
pred = combined_df.groupby("id")['SalePrice'].median().to_frame()
pred["id"] = pred.index
pred[["id", "SalePrice"]].to_csv("./house_price/submission_blended_v2.csv", index=False)
