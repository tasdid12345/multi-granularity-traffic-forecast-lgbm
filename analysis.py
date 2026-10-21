"""Next-hour traffic volume on UCI Metro Interstate: ridge lag baseline vs LightGBM multi-granularity model."""
import json, socket, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")  # headless plotting on GitHub Actions
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from ucimlrepo import fetch_ucirepo

SEED, SEEDS, TEST_FRAC = 42, [42, 43, 44], 0.2
GRID = {"num_leaves": [15, 31, 63], "learning_rate": [0.05, 0.1]}  # small CV grid
socket.setdefaulttimeout(60)  # 60 s timeout on network calls (ucimlrepo uses requests internally)


def load_data(retries=5):
    """Fetch UCI dataset 492 via ucimlrepo, retrying with exponential backoff on failures."""
    for attempt in range(retries):
        try:
            ds = fetch_ucirepo(id=492)
            df = ds.data.features.copy()
            df["traffic_volume"] = ds.data.targets.iloc[:, 0].values
            return df
        except Exception as e:  # HTTP 429/5xx and network errors end up here
            wait = 2 ** attempt
            print(f"download failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    sys.exit("could not download dataset")


def make_hourly(raw):
    """Dedupe timestamps, build a regular hourly series, forward-fill missing hours."""
    df = raw.assign(ts=pd.to_datetime(raw["date_time"])).drop_duplicates("ts", keep="first")
    df = df.set_index("ts").sort_index()
    df["is_holiday"] = (df["holiday"].notna() & (df["holiday"] != "None")).astype(float)
    df["weather_code"] = df["weather_main"].astype("category").cat.codes.astype(float)
    cols = ["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all", "is_holiday", "weather_code"]
    return df[cols].astype(float).asfreq("h").ffill()


def build_features(df):
    """Features known at hour t; target = traffic volume at hour t+1."""
    y = df["traffic_volume"]
    F = pd.DataFrame(index=df.index)
    for k in range(1, 25):                        # 24 hourly lags: baseline inputs
        F[f"y_lag{k}"] = y.shift(k - 1)           # k hours before the target hour
    base_cols = list(F.columns)
    F["y_lag168"] = y.shift(167)                  # same hour one week earlier
    F["roll24_mean"] = y.rolling(24).mean()       # daily level (includes hour t)
    F["roll168_mean"] = y.rolling(168).mean()     # weekly level
    F["hour"] = df.index.hour
    F["dayofweek"] = df.index.dayofweek
    for c in ["temp", "rain_1h", "snow_1h", "clouds_all", "is_holiday", "weather_code"]:
        F[c] = df[c]
    data = F.join(y.shift(-1).rename("target")).dropna()
    return data, base_cols


def evaluate(y_true, y_pred):
    """Main metric MAE; secondary RMSE and R-squared."""
    return {"mae": mean_absolute_error(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": r2_score(y_true, y_pred)}


def lgb_model(params, seed):
    """LightGBM regressor; subsampling makes the seed matter."""
    return lgb.LGBMRegressor(n_estimators=400, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=seed, verbose=-1, **params)


def cv_mae(params, X, y):
    """Mean MAE over time-ordered CV folds inside the training set only."""
    scores = []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=3).split(X):
        m = lgb_model(params, SEED).fit(X.iloc[tr_idx], y.iloc[tr_idx])
        scores.append(mean_absolute_error(y.iloc[va_idx], m.predict(X.iloc[va_idx])))
    return float(np.mean(scores))


def main():
    raw = load_data()
    print(f"loaded {len(raw)} raw rows")
    data, base_cols = build_features(make_hourly(raw))
    feat_all = [c for c in data.columns if c != "target"]
    n_train = int(len(data) * (1 - TEST_FRAC))  # chronological split, no shuffling
    tr, te = data.iloc[:n_train], data.iloc[n_train:]
    print(f"modelling rows: train={len(tr)} test={len(te)}")

    # Tune LightGBM with time-series CV on the training set only
    print("tuning LightGBM by CV on training set...")
    best = min(ParameterGrid(GRID), key=lambda p: cv_mae(p, tr[feat_all], tr["target"]))
    print(f"best params: {best}")

    # Fit both models on the same training data; evaluate once on the test set
    base_res, imp_res = [], []
    for seed in SEEDS:
        print(f"seed {seed}: fitting baseline and improved models")
        p_base = Ridge(alpha=1.0).fit(tr[base_cols], tr["target"]).predict(te[base_cols])
        p_imp = lgb_model(best, seed).fit(tr[feat_all], tr["target"]).predict(te[feat_all])
        base_res.append(evaluate(te["target"], p_base))
        imp_res.append(evaluate(te["target"], p_imp))

    def agg(res, key):
        vals = np.array([r[key] for r in res])
        return float(vals.mean()), float(vals.std())

    b_mae, b_mae_sd = agg(base_res, "mae")
    i_mae, i_mae_sd = agg(imp_res, "mae")
    b_rmse, i_rmse = agg(base_res, "rmse")[0], agg(imp_res, "rmse")[0]
    b_r2, i_r2 = agg(base_res, "r2")[0], agg(imp_res, "r2")[0]
    metrics = {
        "baseline_score": b_mae, "improved_score": i_mae,
        "improvement": b_mae - i_mae,  # MAE is lower-is-better
        "baseline_score_std": b_mae_sd, "improved_score_std": i_mae_sd,
        "baseline_rmse": b_rmse, "improved_rmse": i_rmse,
        "baseline_r2": b_r2, "improved_r2": i_r2,
    }
    if not all(np.isfinite(v) for v in metrics.values()):
        sys.exit("non-finite metrics; failing")

    # Optional chart: MAE comparison across seeds
    import os
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["ridge baseline", "LightGBM"], [b_mae, i_mae],
           yerr=[b_mae_sd, i_mae_sd], capsize=6, color=["grey", "tab:blue"])
    ax.set_ylabel("test MAE (vehicles/hour)")
    ax.set_title("Next-hour traffic: mean over seeds 42-44")
    fig.tight_layout()
    fig.savefig("figures/mae_comparison.png", dpi=120)
    plt.close(fig)

    results = {
        "question": "Can a LightGBM model with multi-granularity lag and calendar features beat a standard ridge lag model at forecasting next-hour traffic volume on the public UCI Metro Interstate dataset?",
        "method": "Chronological 80/20 split; ridge on 24 hourly lags vs LightGBM on multi-granularity lag, rolling-mean, calendar and weather features; LightGBM grid tuned by time-series CV on training data; test set used once.",
        "dataset": {
            "name": "UCI Metro Interstate Traffic Volume",
            "source_url": "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume",
            "n_rows": int(len(raw)),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "MAE (mean absolute error, vehicles per hour)",
        "higher_is_better": False,
        "baseline": "Ridge regression (alpha=1.0) on the previous 24 hourly traffic counts.",
        "improved": "LightGBM regressor with 24 hourly lags, lag-168, 24h and 168h rolling means, hour, day-of-week, holiday and weather features; small CV-tuned grid over num_leaves and learning_rate.",
        "metrics": metrics,
        "notes": ("Single sensor used as a stand-in for car activity; the paper's graph diffusion model is not "
                  "reproduced and no paper numbers are compared. Gaps in the hourly series are forward-filled. "
                  "Ridge is deterministic, so its seed std is 0. Results are from one chronological split on one dataset."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"baseline MAE={b_mae:.2f}  improved MAE={i_mae:.2f}  improvement={b_mae - i_mae:.2f}")
    print("wrote results.json")


if __name__ == "__main__":
    main()
