import numpy as np
import pandas as pd


def predict_sev_probability(trace, signal_id, x1, x2, x3, bypass=0, signal_categories=None):
    """
    Predict P(SEV) for a single input using a PyMC trace.

    Parameters:
      trace: ArviZ InferenceData from model_train.fit_model
      signal_id: signal identifier (string/int)
      x1, x2, x3: feature values
      bypass: 0/1
      signal_categories: list/array of signal categories used in training
                         (if None, will use trace.signal_effect length only)
    """
    post = trace.posterior

    alpha = post["alpha"].stack(draw=("chain", "draw")).values
    signal_effect = post["signal_effect"].stack(draw=("chain", "draw")).values
    beta_bypass = post["beta_bypass"].stack(draw=("chain", "draw")).values
    delta_signal = post["delta_signal"].stack(draw=("chain", "draw")).values
    beta_X = post["beta_X"].stack(draw=("chain", "draw")).values

    # Map signal_id to index
    if signal_categories is None:
        # assume signal_id is already index
        sig_idx = int(signal_id)
    else:
        sig_idx = list(signal_categories).index(signal_id)

    x_vec = np.array([x1, x2, x3])

    logit = (
        alpha
        + signal_effect[:, sig_idx]
        + (beta_bypass + delta_signal[:, sig_idx]) * bypass
        + (beta_X @ x_vec)
    )

    p = 1 / (1 + np.exp(-logit))
    return float(p.mean())


# Example usage:
# from model_train import load_trace
# trace = load_trace("trace.nc")
# prob = predict_sev_probability(trace, signal_id="S3", x1=0.2, x2=1.1, x3=-0.3, bypass=1, signal_categories=["S0","S1","S2","S3",...])
# print(prob)
