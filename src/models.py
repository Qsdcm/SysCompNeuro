import torch
import torch.nn as nn
import torch.nn.functional as F

class PredictiveCodingRNN(nn.Module):
    """
    Hierarchical Predictive Coding Network.
    
    Structure:
    - Top Layer (State Unit): Maintains internal state (hypothesis) h_t.
      Generates top-down prediction for the next sensory input.
    - Bottom Layer (Error Unit): Receives actual sensory input x_t and top-down prediction x_hat_t.
      Computes prediction error e_t.
    
    The 'neural response' is modeled primarily by the error signal e_t.
    """
    def __init__(self, input_size, hidden_size, output_size, gain=1.0):
        super(PredictiveCodingRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gain = gain # Precision / Attention gain
        
        # State Unit: Takes prediction error as input, updates state
        # We use a GRU to model the dynamics of the internal state
        self.rnn = nn.GRUCell(input_size, hidden_size)
        
        # Prediction mapping: h_t -> x_hat_{t+1}
        self.predictor = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h=None):
        """
        Args:
            x: Input sequence (Batch, Seq_Len). LongTensor of token indices.
            h: Initial hidden state.
        Returns:
            outputs: Predictions (Batch, Seq_Len, Output_Size)
            errors: Prediction errors (Batch, Seq_Len, Input_Size) - The "Surprise" signal
            h: Final hidden state
        """
        batch_size, seq_len = x.size()
        
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size).to(x.device)
            
        # Initial prediction (prior) for the first time step
        # Can be initialized to uniform or learned. Here we start with the predictor's output on h_0
        x_hat = self.predictor(h) 
        
        outputs = []
        error_signals = []
        
        for t in range(seq_len):
            # 1. Receive Sensory Input x_t
            x_t_idx = x[:, t]
            # Convert to one-hot for error calculation
            x_t_onehot = F.one_hot(x_t_idx, num_classes=self.output_size).float()
            
            # 2. Compute Prediction Error e_t
            # e_t = Actual - Predicted (Probabilities)
            # We use softmax to get probabilities from logits x_hat
            x_hat_probs = F.softmax(x_hat, dim=1)
            e_t = x_t_onehot - x_hat_probs
            
            # Store error (L2 norm or raw vector) for analysis
            # Here we store the raw vector to analyze magnitude later
            error_signals.append(e_t)
            
            # 3. Apply Precision / Gain
            # "Attention" boosts the error signal
            e_t_weighted = e_t * self.gain
            
            # 4. Update State Unit (Hypothesis)
            # The RNN takes the error as input to update its belief
            h = self.rnn(e_t_weighted, h)
            
            # 5. Generate Next Prediction
            x_hat = self.predictor(h)
            outputs.append(x_hat)
            
        outputs = torch.stack(outputs, dim=1)
        error_signals = torch.stack(error_signals, dim=1)
        
        return outputs, error_signals, h

class BaselineRNN(nn.Module):
    """
    Standard RNN (GRU) Baseline.
    
    Directly maps input x_t to state h_t, then predicts x_{t+1}.
    Does not have an explicit error calculation unit in the forward pass.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(BaselineRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.predictor = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, h=None):
        # x: (Batch, Seq_Len)
        
        # Embed input
        embeds = self.embedding(x) # (Batch, Seq_Len, Hidden)
        
        # RNN pass
        rnn_out, h = self.rnn(embeds, h)
        
        # Predict next token
        logits = self.predictor(rnn_out)
        
        return logits, None, h
