import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head Self-Attention layer in PyTorch.
    """

    def __init__(self, d_model: int, num_heads: int, use_masking: bool = False):
        """
        :param d_model: Dimensionality of the input features.
        :param num_heads: Number of attention heads.
        :param use_masking: Whether to apply causal masking for autoregressive tasks.
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."

        self.d_model = d_model
        self.num_heads = num_heads
        self.use_masking = use_masking
        self.depth = d_model // num_heads

        # Query, Key, and Value projections
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for multi-head self-attention.
        :param x: Input tensor of shape (batch_size, seq_len, d_model).
        :return: Output tensor of shape (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.size()

        # Project inputs to Query, Key, and Value tensors
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, d_model * 3)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads,
                   self.depth).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads,
                   self.depth).permute(0, 2, 3, 1)
        v = v.view(batch_size, seq_len, self.num_heads,
                   self.depth).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        # (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k) / np.sqrt(self.depth)

        if self.use_masking:
            mask = torch.triu(torch.ones(seq_len, seq_len),
                              diagonal=1).bool().to(x.device)
            scores = scores.masked_fill(mask, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        # (batch_size, num_heads, seq_len, depth)
        attention = torch.matmul(attention_weights, v)

        # Reshape and combine heads
        # (batch_size, seq_len, num_heads, depth)
        attention = attention.permute(0, 2, 1, 3).contiguous()
        # (batch_size, seq_len, d_model)
        attention = attention.view(batch_size, seq_len, self.d_model)

        # Apply output projection
        output = self.out_proj(attention)
        return output


class LayerNormalization(nn.Module):
    """
    Implementation of Layer Normalization.
    """

    def __init__(self, d_model: int, eps=1e-5):
        """
        :param d_model: Dimensionality of the input.
        :param eps: A small epsilon to avoid division by zero.
        """
        super(LayerNormalization, self).__init__()
        self.gain = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.gain * normalized + self.bias


class TransformerTransition(nn.Module):
    """
    Transformer transition function with feed-forward layers.
    """

    def __init__(self, d_model: int, size_multiplier: int = 4, activation=F.relu):
        """
        :param d_model: Dimensionality of the input/output.
        :param size_multiplier: Multiplier for the hidden layer size.
        :param activation: Activation function.
        """
        super(TransformerTransition, self).__init__()
        self.activation = activation
        self.hidden_layer = nn.Linear(d_model, size_multiplier * d_model)
        self.output_layer = nn.Linear(size_multiplier * d_model, d_model)

    def forward(self, x):
        x = self.activation(self.hidden_layer(x))
        x = self.output_layer(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer block with self-attention, residual connections,
    normalization, and a feed-forward transition layer.
    """

    def __init__(self, d_model: int, num_heads: int, use_masking: bool = True, size_multiplier: int = 4, dropout_rate: float = 0.1):
        """
        :param d_model: Dimensionality of the model.
        :param num_heads: Number of attention heads.
        :param use_masking: Whether to apply masking in the attention mechanism.
        :param size_multiplier: Multiplier for the hidden size in the feed-forward layer.
        :param dropout_rate: Dropout probability.
        """
        super(TransformerBlock, self).__init__()
        self.self_attention = MultiHeadSelfAttention(
            d_model, num_heads, use_masking)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.transition = TransformerTransition(d_model, size_multiplier)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Self-attention with residual connection and layer normalization
        attention_output = self.self_attention(x)
        attention_output = self.dropout(attention_output)
        x = self.norm1(x + attention_output)

        # Feed-forward transition with residual connection and layer normalization
        transition_output = self.transition(x)
        transition_output = self.dropout(transition_output)
        x = self.norm2(x + transition_output)

        return x


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        # Calculate the padding for causal convolution with dilation
        padding = (self.kernel_size - 1) * self.dilation
        # Pad the input tensor on the left (only past context)
        x = F.pad(x, (padding, 0), mode='constant', value=0)
        # Perform the convolution
        return self.conv(x)


class TransLOB(nn.Module):
    def __init__(self, n_classes=3):
        super(TransLOB, self).__init__()

        self.n_classes = n_classes
        # 1. Convolutional layers
        # 1st convolution:

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # 2nd:
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )

        # 3rd:
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # 2. Inception Layer:
        # 1st inc 1x1 3x1
        self.inc1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # 2nd inc 1x1 5x1
        self.inc2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # 3nd inc max_pool 1x1
        self.inc3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # Dilated CNN:
        self.dilated = nn.Sequential(
            CausalConv1d(192, 14, kernel_size=2, stride=1, dilation=1),
            nn.ReLU(),
            CausalConv1d(14, 14, kernel_size=2, stride=1, dilation=2),
            nn.ReLU(),
            CausalConv1d(14, 14, kernel_size=2, stride=1, dilation=4),
            nn.ReLU(),
            CausalConv1d(14, 14, kernel_size=2, stride=1, dilation=8),
            nn.ReLU(),
            CausalConv1d(14, 14, kernel_size=2, stride=1, dilation=16),
            nn.ReLU(),
        )

        # Layer Normalization block:
        # self.norm1 = nn.LayerNorm()

        # Transformer block
        self.transformer1 = TransformerBlock(d_model=15, num_heads=3)
        self.transformer2 = TransformerBlock(d_model=15, num_heads=3)

        # Feed forward block (MLP)
        self.fc1 = nn.Linear(15*82, 64)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, self.n_classes)
        self.softmax = nn.Softmax(dim=1)

    def positional_encoding(self, x):
        """
        Adds positional encoding to the input tensor in PyTorch.
        :param x: Input tensor of shape (batch_size, steps, d_model)
        :return: Tensor with added positional encoding (batch_size, steps, d_model+1)
        """
        # Extract sequence length and model dimension
        steps, d_model = x.shape[1], x.shape[2]

        # Compute positional encoding
        ps = torch.linspace(-1, 1, steps, dtype=x.dtype,
                            device=x.device).view(-1, 1)  # Shape: (steps, 1)
        # Expand to batch size, Shape: (batch_size, steps, 1)
        ps = ps.unsqueeze(0).expand(x.size(0), -1, -1)

        # Concatenate positional encoding to the input
        # Shape: (batch_size, steps, d_model + 1)
        x = torch.cat([x, ps], dim=-1)

        return x

    def forward(self, x):
        x = x.unsqueeze(1)

        # Normal CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Inception:
        inception1 = self.inc1(x)
        inception2 = self.inc2(x)
        inception3 = self.inc3(x)
        x = torch.cat((inception1, inception2, inception3), dim=1)

        # Reshape for Dilated CNN (from 4D to 3D)
        batch, channels, height, width = x.size()
        x = x.view(batch, channels, -1)

        # Dilated CNN:
        x = self.dilated(x)
        # x = self.norm1(x)

        # Positional encoding:
        x = x.permute(0, 2, 1)
        x = self.positional_encoding(x)
        # print(x.shape)

        # Transformer block:
        x = self.transformer1(x)
        x = self.transformer2(x)

        # MLP and output:
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.relu(x)
        output = self.fc2(x)

        return output
