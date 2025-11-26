import numpy as np

def min_norm_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calcula os pesos beta usando a Solução de Norma Mínima.

    Se m > n (sobre-parametrizado), existem infinitas soluções.
    A pseudo-inversa seleciona aquela com a menor norma l2.
    
    Args:
        X (np.ndarray): Matriz de treino (n, m).
        y (np.ndarray): Vetor alvo (n, 1).
        
    Returns:
        beta_hat (np.ndarray): Vetor de pesos estimados (m, 1).
    """
    # np.linalg.pinv calcula a Pseudo-Inversa de Moore-Penrose
    # O vetor de parametros estimados é o produto escalar da pseudo-inversa de X com o vetor Y
    beta_hat = np.linalg.pinv(X) @ y
    
    return beta_hat

def get_optimal_solution(m_features: int) -> np.ndarray:
    """
    Calcula a solução otima para os pesos.

    Args:
        m_features (int): Número de features (colunas).

    Returns:
        beta_hat (np.ndarray): Vetor de pesos otimo (m, 1).
    """

    # O vetor de pesos otimos é dado por beta = (1/sqrt(m), ..., 1/sqrt(m))
    beta_hat = np.sqrt(1/m_features)

    return beta_hat
