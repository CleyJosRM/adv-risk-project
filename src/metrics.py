import numpy as np
from typing import Tuple

def calculate_risks_linf(X_test: np.ndarray, 
                         y_test: np.ndarray, 
                         beta_hat: np.ndarray, 
                         delta: float) -> Tuple[float, float]:
    """
    Calcula o Risco Padrão e o Risco Adversário l_infinito.
    Baseado no Lemma 1 do paper.
    
    Args:
        X_test (np.ndarray): Dados de teste.
        y_test (np.ndarray): Alvo de teste.
        beta_hat (np.ndarray): Pesos do modelo treinado.
        delta (float): Orçamento do ataque adversário.
        
    Returns:
        risk_std (float): Risco Padrão.
        risk_adv (float): Risco Adversário l_infinito estimado.
    """
    # Calcular o erro original (e0) nos dados de teste
    y_pred = X_test @ beta_hat
    e0 = y_test - y_pred
    
    # Risco Padrão: Média do erro ao quadrado
    risk_std = np.mean(e0**2)
    
    # Para ataque l_inf (p=inf), usamos norma l1 (q=1), para satisfazer a condição 1/p + 1/q = 1 do Lemma
    # Calculamos a norma l1 do vetor de pesos
    norm_l1 = np.linalg.norm(beta_hat, 1)
    
    # Risco Adversário: Fórmula Simplificada do Lemma 1
    # R_adv = Mean( (|e0| + delta * ||beta||_1)^2 )
    adv_loss = (np.abs(e0) + delta * norm_l1)**2
    risk_adv = np.mean(adv_loss)
    
    return risk_std, risk_adv