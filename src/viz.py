import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_error_trace(errors, tokens, title, save_path):
    """
    Plot the prediction error trace over time, highlighting oddballs.
    
    Args:
        errors (np.array): Error magnitude time series (Seq_Len,).
        tokens (np.array): Token indices (Seq_Len,).
        title (str): Plot title.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(12, 4))
    
    # Plot error line
    plt.plot(errors, label='Prediction Error', color='blue', linewidth=2)
    
    # Highlight Oddballs (Token 1)
    oddball_indices = np.where(tokens == 1)[0]
    plt.scatter(oddball_indices, errors[oddball_indices], color='red', zorder=5, label='Oddball (Deviant)')
    
    # Highlight Standards (Token 0) - Optional, maybe just dots
    standard_indices = np.where(tokens == 0)[0]
    plt.scatter(standard_indices, errors[standard_indices], color='green', s=10, alpha=0.5, label='Standard')
    
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Prediction Error (L2 Norm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_oddball_stats(results, save_path):
    """
    Plot average error for Standard vs Oddball.
    
    Args:
        results (dict): {'Standard': [errors], 'Oddball': [errors]}
    """
    means = [np.mean(results['Standard']), np.mean(results['Oddball'])]
    stds = [np.std(results['Standard']), np.std(results['Oddball'])]
    labels = ['Standard', 'Oddball']
    
    plt.figure(figsize=(6, 5))
    plt.bar(labels, means, yerr=stds, capsize=10, color=['green', 'red'], alpha=0.7)
    plt.title('Average Neural Response (Prediction Error)')
    plt.ylabel('Error Magnitude')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_gain_analysis(gains, oddball_responses, save_path):
    """
    Plot Oddball response magnitude vs Gain parameter.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(gains, oddball_responses, marker='o', linestyle='-', color='purple')
    plt.title('Effect of Precision (Gain) on Oddball Response')
    plt.xlabel('Gain (Precision)')
    plt.ylabel('Oddball Error Magnitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
