import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from src.data import OddballDataset
from src.models import PredictiveCodingRNN, BaselineRNN
from src.utils import set_seed, get_device, ensure_dir

def train(args):
    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)

    # --- Data ---
    print("Generating data...")
    dataset = OddballDataset(
        num_samples=args.num_samples, 
        seq_len=args.seq_len, 
        p_oddball=args.p_oddball,
        context_switch=args.context_switch
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # --- Model ---
    print(f"Initializing {args.model_type} model...")
    if args.model_type == 'pc':
        model = PredictiveCodingRNN(
            input_size=2, # Binary tokens (0, 1) -> One-hot dim 2
            hidden_size=args.hidden_size,
            output_size=2,
            gain=args.gain
        )
    else:
        model = BaselineRNN(
            input_size=2,
            hidden_size=args.hidden_size,
            output_size=2
        )
    
    model.to(device)
    
    # --- Optimization ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    print("Starting training...")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            # Note: PC model returns (outputs, errors, h)
            # Baseline returns (outputs, None, h)
            outputs, _, _ = model(x)
            
            # Reshape for Loss: (Batch * Seq, Num_Classes) vs (Batch * Seq)
            loss = criterion(outputs.reshape(-1, 2), y.reshape(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # --- Save ---
    save_path = os.path.join(args.output_dir, f"{args.model_type}_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='pc', choices=['pc', 'baseline'])
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--seq_len', type=int, default=50)
    parser.add_argument('--p_oddball', type=float, default=0.1)
    parser.add_argument('--context_switch', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gain', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='outputs')
    
    args = parser.parse_args()
    train(args)
