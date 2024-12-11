import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layer (output layer)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch_size, seq_length, input_size)
        
        # Passing the input through LSTM layers
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Applying dropout to the final hidden state
        out = self.dropout(lstm_out[:, -1, :])  # take output from the last time step
        
        # Passing through the fully connected output layer
        out = self.fc(out)
        
        return out