import matplotlib.pyplot as plt

def plot_comparison(results, epsilon_label="0.1"):
    """Plots the double descent curves for standard and adversarial risks."""
    plt.figure(figsize=(10, 6))
    dims = results['dimensions']

    plt.plot(dims, results['std']['med'], color='blue', label='Standard Test Risk', lw=2)
    plt.fill_between(dims, results['std']['q1'], results['std']['q3'], color='blue', alpha=0.1)

    if results['adv_l1']:
        plt.plot(dims, results['adv_l1']['med'], color='red', label=rf'Adversarial $\ell_1$ Risk ($\epsilon={epsilon_label}$)', lw=2)
        plt.fill_between(dims, results['adv_l1']['q1'], results['adv_l1']['q3'], color='red', alpha=0.2)
    if results['adv_l2']:
        plt.plot(dims, results['adv_l2']['med'], color='green', label=rf'Adversarial $\ell_2$ Risk ($\epsilon={epsilon_label}$)', lw=2)
        plt.fill_between(dims, results['adv_l2']['q1'], results['adv_l2']['q3'], color='green', alpha=0.2)
    if results['adv_linf']:
        plt.plot(dims, results['adv_linf']['med'], color='yellow', label=rf'Adversarial $\ell_\infty$ Risk ($\epsilon={epsilon_label}$)', lw=2)
        plt.fill_between(dims, results['adv_linf']['q1'], results['adv_linf']['q3'], color='yellow', alpha=0.2)
    
    plt.axvline(x=results['threshold'], color='black', linestyle='--', alpha=0.6, label='Interpolation $n=d$')
  
    plt.yscale('log')
    plt.ylim(1e-1, 1e3)
    plt.xlim(left=1)
    plt.xlabel('Number of Features ($d$)')
    plt.ylabel('Risk (Mean Squared Error)')
    plt.title('Comparison: Standard vs Adversarial Double Descent')
    plt.legend()
    plt.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    plt.show()

def plot_weak_features_comparison(results, epsilon_label="0.05"):
    """
    Plots the double descent curves with error bars for standard 
    and adversarial risk using logarithmically spaced dimensions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dims = results['dimensions']
    n_train = results['threshold']
    
    # Calculate error margins for standard risk
    med_std = results['std']['med']
    err_std_low = med_std - results['std']['q1']
    err_std_high = results['std']['q3'] - med_std
    
    # Calculate error margins for adversarial risk
    med_adv = results['adv']['med']
    err_adv_low = med_adv - results['adv']['q1']
    err_adv_high = results['adv']['q3'] - med_adv

    # Plotting
    ax.errorbar(dims, med_std, yerr=[err_std_low, err_std_high], fmt='o-', 
                linewidth=1.5, markersize=5, capsize=3, label='Standard Risk ($l_2$)')
    
    ax.errorbar(dims, med_adv, yerr=[err_adv_low, err_adv_high], fmt='s-', 
                linewidth=1.5, markersize=5, capsize=3, label=rf'Adversarial Risk ($l_\infty, \epsilon={epsilon_label}$)')

    # Formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axvline(x=n_train, color='k', linestyle='--', alpha=0.7, label=f'Interpolation (m={n_train})')

    ax.set_xlabel(f'Number of parameters (m)', fontsize=12)
    ax.set_ylabel('Risk (MSE)', fontsize=12)
    ax.set_title(rf'Double Descent: Generated Features ($\eta = m, \epsilon = {epsilon_label}$)', fontsize=14, fontweight='bold')
    
    ax.grid(True, which='major', linestyle='-', alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.show()