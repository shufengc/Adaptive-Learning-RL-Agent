import os
import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_XLSX = os.path.join(DATA_DIR, "train_valid_sequences_quelevel_clean_time.xlsx")
OUT_PATH = os.path.join(DATA_DIR, "fitted_params.json")



# Load train_valid_sequences_quelevel_clean_time.xlsx
def load_train_dataframe() -> pd.DataFrame:
    if os.path.exists(TRAIN_XLSX):
        return pd.read_excel(TRAIN_XLSX)
    raise FileNotFoundError("train_valid_sequences_quelevel_clean_time.xlsx not found.")


def build_step_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        uid = row["uid"]
        qs = list(map(int, str(row["questions"]).split(",")))
        rs = list(map(int, str(row["responses"]).split(",")))
        dts = list(map(int, str(row["timedelta"]).split(",")))
        n = min(len(qs), len(rs), len(dts))
        for i in range(n):
            rows.append((uid, i, qs[i], rs[i], dts[i]))
    return pd.DataFrame(rows, columns=["uid", "position", "question", "correct", "delta_t_ms"])

# Fit β: relationship between step index and accuracy
def fit_beta(step_df: pd.DataFrame, max_pos: int = 150) -> float:
    subset = step_df[step_df["position"] < max_pos]
    agg = subset.groupby("position")["correct"].mean()

    xs = agg.index.to_numpy(float)
    ys = agg.values.astype(float)
    if len(xs) < 3:
        return 0.01

    m, b = np.polyfit(xs, ys, 1)
    beta = max(0.0, -m)
    return float(np.clip(beta, 1e-4, 0.1))

# Fit γ and λ: effect of time gap (Δt) on accuracy
def fit_gamma_lambda(step_df: pd.DataFrame) -> Tuple[float, float]:
    df = step_df[step_df["delta_t_ms"] > 0].copy()
    df["delta_t_hours"] = df["delta_t_ms"] / 3_600_000

    q25, q75 = df["delta_t_hours"].quantile([0.25, 0.75])
    short = df[df["delta_t_hours"] <= q25]
    long = df[df["delta_t_hours"] >= q75]

    if len(short) < 50 or len(long) < 50:
        return 0.001, 0.001

    acc_s = short["correct"].mean()
    acc_l = long["correct"].mean()
    dt_s = short["delta_t_hours"].mean()
    dt_l = long["delta_t_hours"].mean()

    eps = 1e-6
    ratio = acc_l / max(acc_s, eps)
    if ratio <= 0:
        lam = 0.001
    else:
        lam = -np.log(ratio) / max(dt_l - dt_s, eps)

    lam = float(np.clip(lam, 1e-5, 1.0))
    gamma = lam  # 简单假设 attention decay ≈ forgetting decay

    return gamma, lam

# Fit learning rate (lr): repeated attempts on the same question
def fit_lr(step_df: pd.DataFrame) -> float:
    df = step_df.sort_values(["uid", "question", "position"])
    groups = df.groupby(["uid", "question"])

    first_acc = []
    last_acc = []
    repeats = []

    for (_, _), g in groups:
        if len(g) >= 2:
            first_acc.append(g["correct"].iloc[0])
            last_acc.append(g["correct"].iloc[-1])
            repeats.append(len(g))

    if not first_acc:
        return 0.05

    m0 = np.mean(first_acc)
    mN = np.mean(last_acc)
    n_avg = np.mean(repeats)

    eps = 1e-6
    if mN <= m0:
        lr = 0.05
    else:
        lr = (mN - m0) / max((1.0 - m0) * n_avg, eps)

    return float(np.clip(lr, 0.001, 0.5))

# Main execution function
def main():
    df = load_train_dataframe()
    step_df = build_step_df(df)

    beta = fit_beta(step_df)
    gamma, lam = fit_gamma_lambda(step_df)
    lr = fit_lr(step_df)

    params = {"beta": beta, "gamma": gamma, "lam": lam, "lr": lr}

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print("Fitted params:")
    for k, v in params.items():
        print(f"{k} = {v:.6f}")
    print("Saved to:", OUT_PATH)


if __name__ == "__main__":
    main()
