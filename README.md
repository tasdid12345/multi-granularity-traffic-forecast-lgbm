# Multi-granularity LightGBM vs ridge lag model for next-hour traffic

Motivation: Forecasting private car activity matters for traffic management, and recent work such as "A multi-granularity self-guiding graph diffusion model for predicting private car activity" (2026) models traffic across several time granularities and spatial graphs. Graph data and method details for that model are not public, so this experiment asks a simpler question with a simpler stand-in.

Question: Can a LightGBM model with multi-granularity lag and calendar features beat a standard ridge lag model at forecasting next-hour traffic volume on the public UCI Metro Interstate dataset?

Dataset: UCI Metro Interstate Traffic Volume (hourly vehicle counts on one Minneapolis interstate segment, with weather), loaded via the `ucimlrepo` package (dataset id 492). It is a single sensor, not the paper's graph, and is not the paper's own dataset.

Baseline vs improved: The baseline is ridge regression on the previous 24 hourly counts. The improved model is a LightGBM regressor that adds longer lags (same hour last week), 24-hour and 168-hour rolling means, hour-of-day, day-of-week, holiday and weather columns, with a small grid tuned by time-series cross-validation on the training set. The split is chronological (first 80% train, last 20% test).

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results: results are written to `results.json` (and an optional chart to `figures/`).

## Results (latest run)

On the public UCI Metro Interstate Traffic Volume dataset, a ridge regression on the previous 24 hourly counts reached a test MAE of 307.96 vehicles per hour, while my LightGBM model with multi-granularity lag and calendar features reached 166.79 on the same held-out split.

| Metric | Value |
|---|---|
| baseline_score | 308 |
| improved_score | 166.8 |
| improvement | 141.2 |
| baseline_score_std | 0 |
| improved_score_std | 0.05859 |
| baseline_rmse | 459.5 |
| improved_rmse | 249.7 |
| baseline_r2 | 0.9464 |
| improved_r2 | 0.9842 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Single sensor used as a stand-in for car activity; the paper's graph diffusion model is not reproduced and no paper numbers are compared. Gaps in the hourly series are forward-filled. Ridge is deterministic, so its seed std is 0. Results are from one chronological split on one dataset.

## Reference

This project is motivated by the following paper:

> B Liu, T Li, Z Xiao, W Chen, Y Hei (2026). *A multi-granularity self-guiding graph diffusion model for predicting private car activity*. IEEE Transactions on. https://ieeexplore.ieee.org/abstract/document/11373778/

Paper author: Prof. Guo Chen, Hunan University (guochen@hnu.edu.cn)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
