import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve


def _prep_indices(df):
    df = df.copy()
    signal_idx = pd.Categorical(df["signal"]).codes
    num_signals = df["signal"].nunique()
    return df, signal_idx, num_signals


def train_test_split_idx(df, test_size=0.2, random_seed=42):
    return train_test_split(np.arange(len(df)), test_size=test_size, random_state=random_seed)


def predict_proba(df, trace, idx):
    df, signal_idx, _ = _prep_indices(df)
    X = df[["X1", "X2", "X3"]].values
    bypass = df["bypass"].values

    post = trace.posterior
    alpha = post["alpha"].stack(draw=("chain", "draw")).values
    signal_effect = post["signal_effect"].stack(draw=("chain", "draw")).values
    beta_bypass = post["beta_bypass"].stack(draw=("chain", "draw")).values
    delta_signal = post["delta_signal"].stack(draw=("chain", "draw")).values
    beta_X = post["beta_X"].stack(draw=("chain", "draw")).values

    logit = (
        alpha[:, None]
        + signal_effect[:, signal_idx[idx]]
        + (beta_bypass[:, None] + delta_signal[:, signal_idx[idx]]) * bypass[idx]
        + (X[idx] @ beta_X.T).T
    )
    p = 1 / (1 + np.exp(-logit))
    return p.mean(axis=0)


def posterior_predictive_checks(model, trace, num_pp_samples=200):
    with model:
        ppc = pm.sample_posterior_predictive(trace, var_names=["sev_obs"], random_seed=42)
    az.plot_ppc(ppc, num_pp_samples=num_pp_samples)
    plt.show()
    return ppc


def eval_metrics(y_true, p_pred):
    return {
        "log_loss": log_loss(y_true, p_pred),
        "brier": brier_score_loss(y_true, p_pred),
    }


def lift_at_k(y_true, p_pred, k=0.05):
    """
    Lift at top-k fraction.
    k can be fraction (0<k<=1) or integer count.
    Returns (lift, topk_rate, overall_rate).
    """
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)

    if k <= 1:
        n_top = max(1, int(np.ceil(len(y_true) * k)))
    else:
        n_top = int(k)

    idx = np.argsort(p_pred)[::-1][:n_top]
    topk_rate = y_true[idx].mean()
    overall_rate = y_true.mean()
    lift = topk_rate / overall_rate if overall_rate > 0 else np.nan
    return lift, topk_rate, overall_rate


def baseline_global(y_train, y_test):
    global_rate = np.mean(y_train)
    p_base = np.full_like(y_test, fill_value=global_rate, dtype=float)
    return global_rate, p_base


def baseline_signal(df, train_idx, test_idx, fallback_rate):
    signal_rates = df.iloc[train_idx].groupby("signal")["sev"].mean()
    p_sig_base = df.iloc[test_idx]["signal"].map(signal_rates).fillna(fallback_rate).values
    return p_sig_base


def plot_calibration(y_true, p_pred, n_bins=10, title="Calibration"):
    frac_pos, mean_pred = calibration_curve(y_true, p_pred, n_bins=n_bins)
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], "--")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.show()


def signal_sanity(df, idx, p_hat):
    df_sub = df.iloc[idx].copy()
    df_sub["p_hat"] = p_hat
    return (
        df_sub.groupby("signal")
        .agg(n=("sev", "size"), sev_rate=("sev", "mean"), pred_rate=("p_hat", "mean"))
        .sort_values("n", ascending=False)
    )


# Example usage (uncomment to run)
# from model_train import fit_model
#
# df = pd.DataFrame({"signal": signals, "X1": X1, "X2": X2, "X3": X3, "bypass": bypass, "sev": sev})
# train_idx, test_idx = train_test_split_idx(df)
#
# model, trace = fit_model(df.iloc[train_idx])
# p_test = predict_proba(df, trace, test_idx)
# y_test = df.iloc[test_idx]["sev"].values
# y_train = df.iloc[train_idx]["sev"].values
#
# # Baselines
# global_rate, p_base = baseline_global(y_train, y_test)
# p_sig_base = baseline_signal(df, train_idx, test_idx, global_rate)
#
# print("Model:", eval_metrics(y_test, p_test))
# print("Baseline global:", eval_metrics(y_test, p_base))
# print("Baseline signal:", eval_metrics(y_test, p_sig_base))
#
# plot_calibration(y_test, p_test, title="Model calibration")
# print(signal_sanity(df, train_idx, predict_proba(df, trace, train_idx)).head(10))
