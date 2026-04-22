import numpy as np
from typing import Tuple

def generate_weak_features(n_samples: int, m_features: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera dados sintéticos baseados no modelo de 'Weak Features'
    (Mantido do original)
    """
    # Gerar y ~ N(0, 1)
    y = np.random.randn(n_samples, 1)
    
    # Scalling eta = m
    eta = m_features

    # Média das features = y / m
    medias = np.tile(y / eta, (1, m_features))
    
    # Gerar X | y ~ N(media, variancia)
    X = np.random.normal(loc=medias, scale=1.0 / eta)
    
    return X, y

def generate_isotropic_features(n_samples: int, m_features: int, r_signal: float, sigma_noise: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera dados para o Modelo de Features Isotrópicas (Fig 3 do paper).
    
    X ~ N(0, I)
    y = X @ beta + noise
    ||beta||_2 = r_signal
    """
    # Gerar beta aleatório
    beta_raw = np.random.randn(m_features, 1)
    # Normalizar para ter norma l2 igual a r_signal
    beta = beta_raw / np.linalg.norm(beta_raw) * r_signal
    
    # Gerar X (isotrópico, média 0, variância 1)
    X = np.random.randn(n_samples, m_features)
    
    # Gerar ruído
    noise = np.random.randn(n_samples, 1) * sigma_noise
    
    # Gerar y = X*beta + ruido
    y = X @ beta + noise
    
    return X, y
