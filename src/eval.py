import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import defaultdict

from src.data import OddballDataset
from src.models import PredictiveCodingRNN, BaselineRNN
from src.utils import set_seed, get_device, ensure_dir
from src.viz import plot_error_trace, plot_oddball_stats, plot_gain_analysis

def evaluate(args):
    set_seed(args.seed)
    device = get_device()
    ensure_dir(os.path.join(args.output_dir, 'figs'))
    ensure_dir(os.path.join(args.output_dir, 'metrics'))

    # --- Load Model ---
    print(f"Loading {args.model_type} model...")
    if args.model_type == 'pc':
        model = PredictiveCodingRNN(2, args.hidden_size, 2, gain=args.gain)
    else:
        model = BaselineRNN(2, args.hidden_size, 2)
    
    model_path = os.path.join(args.output_dir, f"{args.model_type}_model.pth")
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found! Please train first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Generate Test Data ---
    # Single long sequence for visualization
    test_dataset = OddballDataset(num_samples=1, seq_len=200, p_oddball=args.p_oddball, seed=100)
    x, y = test_dataset[0] # x: 0..T-1, y: 1..T
    x_batch = x.unsqueeze(0).to(device) # (1, Seq)
    
    # --- Run Inference ---
    with torch.no_grad():
        if args.model_type == 'pc':
            outputs, errors, _ = model(x_batch)
            # errors: (1, Seq, 2) -> Norm -> (Seq,)
            error_mags = torch.norm(errors, dim=2).squeeze(0).cpu().numpy()
        else:
            outputs, _, _ = model(x_batch)
            # For baseline, "Surprise" is the Cross Entropy Loss per step
            # outputs: (1, Seq, 2)
            # y: (Seq,)
            # We compute loss per token
            loss_per_step = []
            for t in range(outputs.size(1)):
                l = F.cross_entropy(outputs[:, t, :], y[t].unsqueeze(0).to(device))
                loss_per_step.append(l.item())
            error_mags = np.array(loss_per_step)

    # --- Analysis ---
    # Separate errors by token type (using x, which is the input at time t)
    # Note: error at time t is response to input x[t]
    tokens = x.cpu().numpy()
    
    stats = defaultdict(list)
    for t, token in enumerate(tokens):
        if token == 0:
            stats['Standard'].append(error_mags[t])
        else:
            stats['Oddball'].append(error_mags[t])
            
    print(f"Mean Standard Error: {np.mean(stats['Standard']):.4f}")
    print(f"Mean Oddball Error: {np.mean(stats['Oddball']):.4f}")

    # --- Visualization ---
    # 1. Trace
    plot_error_trace(
        error_mags, tokens, 
        f'{args.model_type.upper()} Model Response (p={args.p_oddball})',
        os.path.join(args.output_dir, 'figs', f'{args.model_type}_trace.png')
    )
    
    # 2. Stats
    plot_oddball_stats(
        stats,
        os.path.join(args.output_dir, 'figs', f'{args.model_type}_stats.png')
    )

    # --- Gain Sweep (Only for PC) ---
    if args.model_type == 'pc' and args.do_sweep:
        print("Running Gain Sweep...")
        gains = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        oddball_means = []
        
        for g in gains:
            # Reload model to reset state (though we just change gain)
            # Actually we can just set the attribute
            model.gain = g
            
            with torch.no_grad():
                _, errors, _ = model(x_batch)
                mags = torch.norm(errors, dim=2).squeeze(0).cpu().numpy()
                
            # Get mean oddball error
            oddball_errs = [mags[t] for t, tok in enumerate(tokens) if tok == 1]
            oddball_means.append(np.mean(oddball_errs))
            
        plot_gain_analysis(
            gains, oddball_means,
            os.path.join(args.output_dir, 'figs', 'gain_sweep.png')
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='pc', choices=['pc', 'baseline'])
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--p_oddball', type=float, default=0.1)
    parser.add_argument('--gain', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--do_sweep', action='store_true', help="Run gain sweep (PC only)")
    
    args = parser.parse_args()
    evaluate(args)
