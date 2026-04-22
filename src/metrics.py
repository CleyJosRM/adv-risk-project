import numpy as np
from sklearn.metrics import mean_squared_error

def get_min_norm_weights(X, y):
    """Returns the minimum-norm least squares solution."""
    return np.linalg.pinv(X) @ y

def get_optimal_solution(m_features: int) -> np.ndarray:
    """Returns the optimal solution for the weak features example."""
    # O vetor de pesos otimos é dado por beta = (1/sqrt(m), ..., 1/sqrt(m))
    beta_value = np.sqrt(1/m_features)
    beta_hat = np.full((m_features, 1), beta_value)

    return beta_hat

def calculate_risks(w, X_test, y_test, epsilon, p):
    """Calculates both standard MSE and Adversarial Risk bounded by epsilon."""
    mse = mean_squared_error(y_test, X_test @ w)
    
    # Dual norm calculation
    if p == np.inf:
        dual_norm_w = np.linalg.norm(w, ord=1)
    elif p == 1:
        dual_norm_w = np.linalg.norm(w, ord=np.inf)
    else:
        q = p/(p - 1)
        dual_norm_w = np.linalg.norm(w, q)
        
    adv_risk = mse + (epsilon * dual_norm_w)**2
    return mse, adv_risk

