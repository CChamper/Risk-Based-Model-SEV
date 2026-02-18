from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

features = ["signal", "X1", "X2", "X3"]

preprocess = ColumnTransformer([
    ("signal", OneHotEncoder(drop="first"), ["signal"]),
    ("num", StandardScaler(), ["X1","X2","X3"])
])

prop_model = Pipeline([
    ("prep", preprocess),
    ("logreg", LogisticRegression(max_iter=1000))
])

prop_model.fit(df[features], df["bypass"])

df["propensity"] = prop_model.predict_proba(df[features])[:,1]

# Propensity score
df["propensity"] = prop_model.predict_proba(X_scaled)[:,1]


p_bypass = df["bypass"].mean()

df["ipw"] = np.where(
    df["bypass"] == 1,
    p_bypass / df["propensity"],
    (1 - p_bypass) / (1 - df["propensity"])
)

df["propensity"] = df["propensity"].clip(0.05, 0.95)
df["ipw"] = df["ipw"].clip(0, 10)

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


"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 1500
signals = np.random.choice(['S1','S2','S3','S4','S5'], size=N)
X1 = np.random.normal(0,1,N)
X2 = np.random.normal(0,1,N)
X3 = np.random.normal(0,1,N)

signal_bias = {'S1':-1.5,'S2':-0.5,'S3':0.0,'S4':0.5,'S5':1.0}
bypass_logit = -0.3 + 0.5*X1 - 0.2*X2 + np.array([signal_bias[s] for s in signals])
bypass_prob = 1/(1+np.exp(-bypass_logit))
bypass = np.random.binomial(1,bypass_prob)

sev_signal_effect = {'S1':0.8,'S2':0.2,'S3':0.0,'S4':-0.2,'S5':-0.5}
sev_logit = -3 + 1.2*bypass + 0.3*X1 + np.array([sev_signal_effect[s] for s in signals])
sev_prob = 1/(1+np.exp(-sev_logit))
sev = np.random.binomial(1,sev_prob)

df = pd.DataFrame({
    "signal": signals,
    "X1": X1,
    "X2": X2,
    "X3": X3,
    "bypass": bypass,
    "sev": sev,
})

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

features = ["signal", "X1", "X2", "X3"]

preprocess = ColumnTransformer([
    ("signal", OneHotEncoder(drop="first"), ["signal"]),
    ("num", StandardScaler(), ["X1","X2","X3"])
])

prop_model = Pipeline([
    ("prep", preprocess),
    ("logreg", LogisticRegression(max_iter=1000))
])

prop_model.fit(df[features], df["bypass"])
df["propensity"] = prop_model.predict_proba(df[features])[:,1]

df["propensity"] = df["propensity"].clip(0.05,0.95)

p_bypass = df["bypass"].mean()

df["ipw"] = np.where(
    df["bypass"] == 1,
    p_bypass / df["propensity"],
    (1 - p_bypass) / (1 - df["propensity"])
)

df["ipw"] = df["ipw"].clip(0,10)

import pymc as pm
import pytensor.tensor as pt

signal_idx, signal_labels = pd.factorize(df["signal"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[["X1","X2","X3"]])

with pm.Model() as model:
    mu_signal = pm.Normal("mu_signal", 0, 1)
    sigma_signal = pm.HalfNormal("sigma_signal", 1)

    signal_effect = pm.Normal("signal_effect", mu=mu_signal, sigma=sigma_signal, shape=len(signal_labels))

    beta_bypass = pm.Normal("beta_bypass", 0, 1)
    beta_X = pm.Normal("beta_X", 0, 1, shape=3)

    logit_p = (
        signal_effect[signal_idx]
        + beta_bypass * df["bypass"].values
        + pt.dot(X_scaled, beta_X)
    )

    w = df["ipw"].values

    pm.Potential("weighted_like", w * pm.logp(pm.Bernoulli.dist(logit_p=logit_p), df["sev"].values))

    trace = pm.sample(500, tune=800, target_accept=0.9, progressbar=False)

posterior = trace.posterior

def predict(signal_id, X_new):
    idx = list(signal_labels).index(signal_id)
    X_new_scaled = scaler.transform([X_new])

    sig = posterior["signal_effect"].stack(s=("chain","draw")).values[:,idx]
    bypass = posterior["beta_bypass"].stack(s=("chain","draw")).values
    betaX = posterior["beta_X"].stack(s=("chain","draw")).values

    logit = sig + bypass + np.dot(betaX.T, X_new_scaled[0])
    prob = 1/(1+np.exp(-logit))

    return prob.mean(), np.percentile(prob,5), np.percentile(prob,95)

results = []
for s in signal_labels:
    mean, lo, hi = predict(s,[0,0,0])
    results.append([s,mean,lo,hi])

pd.DataFrame(results, columns=["Signal","Mean SEV Risk if Bypass","5%","95%"])

"""