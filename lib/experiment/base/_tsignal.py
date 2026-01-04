from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_time_signal(original_signal, reconstructed_signal):
    """
    Evaluates the quality of reconstructed time-domain signals.
    """
    mse = mean_squared_error(original_signal, reconstructed_signal)
    correlation = np.corrcoef(original_signal.ravel(), reconstructed_signal.ravel())[0, 1]
    return {
        "mse": mse,
        "correlation": correlation,
    }
