import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(np.abs(y_pred - y_true) / denominator) * 100


def multi_model_chain_predict(df, path, frac=0.3, max_depth=5):
    predictions = {}
    error_metrics = {}

    x_pred = df[path[0]].values

    for i in range(1, len(path)):
        y = df[path[i]].values
        x = x_pred.reshape(-1, 1)

        # LOESS
        sort_idx = np.argsort(x.ravel())
        x_sorted = x.ravel()[sort_idx]
        y_sorted = y[sort_idx]
        loess_result = lowess(y_sorted, x_sorted, frac=frac, return_sorted=False)
        loess_preds = np.empty_like(loess_result)
        loess_preds[sort_idx] = loess_result

        # SVR
        svr = SVR(kernel='rbf')
        svr.fit(x, y)
        svr_preds = svr.predict(x)

        # CART
        cart = DecisionTreeRegressor(max_depth=max_depth)
        cart.fit(x, y)
        cart_preds = cart.predict(x)

        # Save predictions
        col_name = path[i]
        predictions[f"{col_name}_loess"] = loess_preds
        predictions[f"{col_name}_svr"] = svr_preds
        predictions[f"{col_name}_cart"] = cart_preds

        # Compute metrics
        edge = (path[i - 1], path[i])
        # # Pointwise sMAPE
        # smape_series_loess = pointwise_smape(y, loess_preds)
        # smape_series_svr = pointwise_smape(y, svr_preds)
        # smape_series_cart = pointwise_smape(y, cart_preds)

        error_metrics[edge] = {
            "rmse": [
                np.sqrt(np.mean((y - loess_preds) ** 2)),
                np.sqrt(np.mean((y - svr_preds) ** 2)),
                np.sqrt(np.mean((y - cart_preds) ** 2)),
            ],
            "mae": [
                np.mean(np.abs(y - loess_preds)),
                np.mean(np.abs(y - svr_preds)),
                np.mean(np.abs(y - cart_preds)),
            ],
            "smape": [
                smape(y, loess_preds),
                smape(y, svr_preds),
                smape(y, cart_preds),
            ],
        }

        x_pred = loess_preds

    df_result = df.copy()
    for key, val in predictions.items():
        df_result[key] = val

    return df_result, error_metrics
