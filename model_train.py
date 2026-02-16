import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


def fit_model(df, draws=1000, tune=1000, target_accept=0.9, random_seed=42):
    """
    Fit hierarchical logistic regression for SEV risk.
    Returns (model, trace).
    """
    df = df.copy()
    signal_idx = pd.Categorical(df["signal"]).codes
    num_signals = df["signal"].nunique()

    X = df[["X1", "X2", "X3"]].values
    bypass = df["bypass"].values
    sev = df["sev"].values

    with pm.Model() as model:
        alpha = pm.Normal("alpha", 0, 2)
        signal_effect = pm.Normal("signal_effect", 0, 1, shape=num_signals)

        beta_bypass = pm.Normal("beta_bypass", 0, 1)
        sigma_delta = pm.HalfNormal("sigma_delta", 1)
        delta_signal = pm.Normal("delta_signal", 0, sigma_delta, shape=num_signals)

        beta_X = pm.Normal("beta_X", 0, 1, shape=3)

        logit_p = (
            alpha
            + signal_effect[signal_idx]
            + (beta_bypass + delta_signal[signal_idx]) * bypass
            + (X @ beta_X)
        )
        p = pm.math.sigmoid(logit_p)
        pm.Bernoulli("sev_obs", p=p, observed=sev)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )

    return model, trace


def save_trace(trace, path="trace.nc"):
    """Save ArviZ InferenceData trace to NetCDF."""
    az.to_netcdf(trace, path)


def load_trace(path="trace.nc"):
    """Load ArviZ InferenceData trace from NetCDF."""
    return az.from_netcdf(path)
