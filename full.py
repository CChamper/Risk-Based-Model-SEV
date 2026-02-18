from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Features for propensity model
X_prop = df[["X1", "X2", "X3"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_prop)

prop_model = LogisticRegression()
prop_model.fit(X_scaled, df["bypass"])

# Propensity score
df["propensity"] = prop_model.predict_proba(X_scaled)[:,1]


p_bypass = df["bypass"].mean()

df["ipw"] = np.where(
    df["bypass"] == 1,
    p_bypass / df["propensity"],
    (1 - p_bypass) / (1 - df["propensity"])
)

signal_idx, signal_labels = pd.factorize(df["signal"])

from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(df[["X1","X2","X3"]])

import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:

    # Hyperpriors for signal effects
    mu_signal = pm.Normal("mu_signal", 0, 1)
    sigma_signal = pm.HalfNormal("sigma_signal", 1)

    signal_effect = pm.Normal(
        "signal_effect",
        mu=mu_signal,
        sigma=sigma_signal,
        shape=len(signal_labels)
    )

    # Global coefficients
    beta_bypass = pm.Normal("beta_bypass", 0, 1)
    beta_X = pm.Normal("beta_X", 0, 1, shape=3)

    # Linear predictor
    logit_p = (
        signal_effect[signal_idx]
        + beta_bypass * df["bypass"].values
        + pt.dot(X_scaled, beta_X)
    )

    # Convert IPW into likelihood weights
    w = df["ipw"].values

    pm.Bernoulli(
        "sev",
        logit_p=logit_p,
        observed=df["sev"].values,
        total_size=len(df),
        # weighted likelihood trick
        logp=lambda value: w * pm.logp(pm.Bernoulli.dist(logit_p=logit_p), value)
    )

    trace = pm.sample(1000, tune=1500, target_accept=0.9)


def predict_sev_if_bypass(signal_id, X_new):
    X_new_scaled = scaler_X.transform([X_new])

    signal_index = signal_labels.tolist().index(signal_id)

    logit = (
        trace.posterior["signal_effect"].mean(dim=("chain","draw")).values[signal_index]
        + trace.posterior["beta_bypass"].mean()
        + np.dot(X_new_scaled, trace.posterior["beta_X"].mean(dim=("chain","draw")).values)
    )

    return 1 / (1 + np.exp(-logit))


import numpy as np

def predict_with_uncertainty(signal_id, X_new):

    X_new_scaled = scaler_X.transform([X_new])
    signal_index = signal_labels.tolist().index(signal_id)

    signal_samples = trace.posterior["signal_effect"].stack(sample=("chain","draw")).values[:, signal_index]
    bypass_samples = trace.posterior["beta_bypass"].stack(sample=("chain","draw")).values
    betaX_samples = trace.posterior["beta_X"].stack(sample=("chain","draw")).values

    logit = (
        signal_samples
        + bypass_samples
        + np.dot(betaX_samples.T, X_new_scaled[0])
    )

    prob = 1 / (1 + np.exp(-logit))

    mean = prob.mean()
    lower = np.percentile(prob, 5)
    upper = np.percentile(prob, 95)

    return {
        "mean_risk": mean,
        "90%_CI": (lower, upper),
        "uncertainty": upper - lower
    }
