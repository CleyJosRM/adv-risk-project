import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from src.metrics import get_min_norm_weights, get_optimal_solution, calculate_risks
from src.data_generation import generate_weak_features

def run_simulation_in_data(X, y, n_experiments=10, epsilon=0.1, mode='importance', norms=(1, 2, np.inf)):
    """
    Simulates double descent and calculates adversarial risks for a specified list of p-norms.
    """
    n_samples = int(len(X) * 0.8)
    max_features = min(X.shape[1], int(n_samples * 3))
    dimensions = np.unique(np.linspace(2, max_features, 60, dtype=int))

    # Initialize storage: {norm: [experiment_results]}
    all_std_risks = []
    all_adv_risks = {p: [] for p in norms}

    for i in range(n_experiments):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
     
        # Feature Selection Logic
        if mode == 'importance':
            scores, _ = f_regression(X_train, y_train)
            indices = np.argsort(scores)[::-1]
        elif mode == 'random':
            indices = np.random.permutation(X.shape[1])
        else: 
            indices = np.arange(X.shape[1])
            
        X_tr_sorted = X_train[:, indices]
        X_te_sorted = X_test[:, indices]

        exp_std = []
        exp_adv = {p: [] for p in norms}

        for d in dimensions:
            X_tr_d = X_tr_sorted[:, :d]
            X_te_d = X_te_sorted[:, :d]
            
            w = get_min_norm_weights(X_tr_d, y_train)
            
            # Calculate standard risk once, then loop through requested norms
            for j, p in enumerate(norms):
                std, adv = calculate_risks(w, X_te_d, y_test, epsilon, p)
                if j == 0:
                    exp_std.append(std)
                exp_adv[p].append(adv)
            
        all_std_risks.append(exp_std)
        for p in norms:
            all_adv_risks[p].append(exp_adv[p])

    # Helper to aggregate stats
    def get_stats(data):
        return {
            'med': np.median(data, axis=0),
            'q1': np.percentile(data, 25, axis=0),
            'q3': np.percentile(data, 75, axis=0)
        }

    # Construct results dictionary
    results = {
        'dimensions': dimensions,
        'threshold': n_samples,
        'std': get_stats(all_std_risks)
    }
    
    for p in norms:
        # Key naming: adv_linf for np.inf, adv_lp otherwise
        key = f'adv_l{p}' if p != np.inf else 'adv_linf'
        results[key] = get_stats(all_adv_risks[p])
 
    return results

def run_weak_features_simulation(n_train=100, n_test=1000, epsilon=0.05, trials=10, p=np.inf, mode='min_norm'):
    """
    Simulates standard and adversarial double descent using dynamically 
    generated weak features.
    """
    # Define logarithmically spaced dimensions
    start_power = np.log10(n_train / 10)
    end_power = np.log10(n_train * 100)
    m_log_dist = np.logspace(start_power, end_power, num=60)
    
    m_list = np.unique(m_log_dist.astype(int))
    
    # Ensure the interpolation threshold (m = n) is exactly captured
    if n_train not in m_list:
        m_list = np.append(m_list, n_train)
        m_list.sort()

    all_std_risks, all_adv_risks = [], []

    print(f"Simulating for {len(m_list)} dimensions: from {m_list[0]} to {m_list[-1]}")

    for m in m_list:
        exp_std, exp_adv = [], []
        
        for _ in range(trials):
            X_train, y_train = generate_weak_features(n_train, m)
            X_test, y_test = generate_weak_features(n_test, m)

            if mode == 'min_norm':
                beta = get_min_norm_weights(X_train, y_train)
            elif mode == 'optimal':
                beta = get_optimal_solution(m)
                
            std, adv = calculate_risks(beta, X_test, y_test, epsilon, p)
            
            exp_std.append(std)
            exp_adv.append(adv)
            
        all_std_risks.append(exp_std)
        all_adv_risks.append(exp_adv)

    # Convert to arrays to calculate statistics along the trials axis (axis=1)
    return {
        'dimensions': m_list,
        'threshold': n_train,
        'std': {
            'med': np.median(all_std_risks, axis=1),
            'q1': np.percentile(all_std_risks, 25, axis=1),
            'q3': np.percentile(all_std_risks, 75, axis=1)
        },
        'adv': {
            'med': np.median(all_adv_risks, axis=1),
            'q1': np.percentile(all_adv_risks, 25, axis=1),
            'q3': np.percentile(all_adv_risks, 75, axis=1)
        }
    }