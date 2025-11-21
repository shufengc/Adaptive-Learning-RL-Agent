import os
import json
import numpy as np
import pandas as pd

from env.student_state import init_student_state, StudentState
from env.student_model import prob_correct, update_state
from env.environment import load_question_difficulty


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PARAM_PATH = os.path.join(DATA_DIR, "fitted_params.json")

TEST_FILE = os.path.join(DATA_DIR, "test_quelevel_cleaned_time.csv")



# Load fitted model parameters
def load_params():
    if not os.path.exists(PARAM_PATH):
        return {"lr": 0.05, "beta": 0.01, "gamma": 0.001, "lam": 0.001}
    with open(PARAM_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def roc_auc_score_manual(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    order = np.argsort(y_score)
    y_sorted = y_true[order]
    ranks = np.arange(1, len(y_score) + 1)
    pos_ranks = ranks[y_sorted == 1]
    n_pos = np.sum(y_sorted == 1)
    n_neg = np.sum(y_sorted == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)

# Load test dataset
def load_test_df():
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(f"{TEST_FILE} not found")
    return pd.read_csv(TEST_FILE)


def parse_row(row):
    qs = list(map(int, row["questions"].split(",")))
    rs = list(map(int, row["responses"].split(",")))
    dts = list(map(int, row["timedelta"].split(",")))
    n = min(len(qs), len(rs), len(dts))
    return qs[:n], rs[:n], dts[:n]


# AUC evaluation
def evaluate_auc():
    df = load_test_df()
    q2d = load_question_difficulty()
    params = load_params()

    lr = params["lr"]
    beta = params["beta"]
    gamma = params["gamma"]
    lam = params["lam"]

    y_true, y_pred = [], []

    for _, row in df.iterrows():
        state = init_student_state()

        qs, rs, dts = parse_row(row)
        for q_id, correct, dt in zip(qs, rs, dts):
            diff = q2d.get(q_id, 0.5)
            p = prob_correct(state, diff)

            y_true.append(correct)
            y_pred.append(p)

            delta_t_hours = dt / 3_600_000
            state = update_state(
                state=state,
                correct=correct,
                difficulty=diff,
                delta_t=delta_t_hours,
                lr=lr,
                lam=lam,
                beta=beta,
                gamma=gamma,
            )

    auc = roc_auc_score_manual(y_true, y_pred)
    print("AUC =", auc)
    return auc


if __name__ == "__main__":
    evaluate_auc()
