import numpy as np
from typing import Tuple

def generate_weak_features(n_samples: int, m_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera dados sintéticos baseados no modelo de 'Weak Features'
    
    Neste modelo, o sinal y é distribuído em muitas features x.
    Cada feature individual tem pouco sinal, mas a combinação delas recupera y.
    
    Args:
        n_samples (int): Número de amostras (linhas).
        m_features (int): Número de features (colunas).
        
    Returns:
        X (np.ndarray): Matriz de features (n, m).
        y (np.ndarray): Vetor de alvo (n, 1).
    """
    # Gerar y ~ N(0, 1)
    y = np.random.randn(n_samples, 1)
    
    # Scalling eta = 1 / sqrt(m)
    eta = m_features

    # Média das features = y / sqrt(m)
    # Vetorizar a média para criar a matriz (n, m)
    medias = np.tile(y / eta, (1, m_features))
    
    # Gerar X | y ~ N(media, variancia)
    X = np.random.normal(loc=medias, scale=1.0 / eta)
    
    return X, y