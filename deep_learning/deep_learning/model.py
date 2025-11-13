import torch
import torch.nn as nn
# from mamba_ssm import Mamba
import math

# Positional Encoding (used for both Transformer and Mamba)
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (max_len, 1, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x): # x shape: (sequence_length, batch_size, embed_dim)
        return x + self.pe[:x.size(0), :]

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, embed_dim)
        attn_output, _ = self.attention(x, x, x)  # Self-attention
        x = self.norm1(x + self.dropout(attn_output))  # Add & Norm
        ffn_output = self.ffn(x)  # Feed-forward network
        x = self.norm2(x + self.dropout(ffn_output))  # Add & Norm
        return x

class MambaBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim=None, dropout=0.1):
        super().__init__()
        self.mamba = Mamba(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        ff_dim = ff_dim or embed_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        residual = x
        x = self.mamba(x)
        x = self.norm1(x + self.dropout(residual))
        residual = x
        x = self.ffn(x)
        return self.norm2(x + self.dropout(residual))

class TemporalConvPool(nn.Module):
    def __init__(self, embed_dim, num_stages=3, kernel_size=32, stride=2, pool_kernel=2, pool_stride=2):
        super().__init__()
        layers = []
        
        for i in range(num_stages):
            e1 = embed_dim * round((num_stages/2 + 1) - abs(i - (num_stages/2)))
            e2 = embed_dim * round((num_stages/2 + 1) - abs(i + 1 - (num_stages/2)))
            layers.append(
                nn.Sequential(
                    nn.Conv1d(e1, e2, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride)
                )
            )
        self.stages = nn.ModuleList(layers)

    def forward(self, x):
        # x shape: (batch_size, embed_dim, sequence_length)
        for stage in self.stages:
            x = stage(x)
        return x

# Unified Model (without joint hierarchy)
class UnifiedModel(nn.Module):
    def __init__(self, input_size, output_size, sequence_length, config, device):
        super(UnifiedModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sequence_length = sequence_length
 
        self.m = config["m"]
        self.embed_dim = config["embed_dim"]
        self.model_type = config["model_type"]

        # Embedding and positional encoding
        self.embedding = nn.Linear(self.input_size, self.embed_dim)
        self.temporalConvPool = TemporalConvPool(self.embed_dim)

        self.positional_encoding = PositionalEncoding(self.embed_dim)

        # Temporal blocks (Transformer or Mamba)
        if self.model_type == "Transformer":
            self.temporal_blocks = nn.ModuleList([
                TransformerBlock(self.embed_dim, config["num_heads"], config["ff_dim"], config["dropout"])
                for _ in range(self.m)
            ])
        elif self.model_type == "Mamba":
            self.temporal_blocks = nn.ModuleList([
                MambaBlock(self.embed_dim, config["dropout"])
                for _ in range(self.m)
            ])
        else:
            raise ValueError("model_type must be 'Transformer' or 'Mamba'")

        self.fc = nn.Linear(self.embed_dim, self.output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, sequence_length, _ = x.shape

        # Embed input
        x = self.embedding(x)  # (batch_size, sequence_length, embed_dim)

        # x = x.permute(0, 2, 1) # (batch_size, embed_dim , sequence_length)
        # x = self.temporalConvPool(x)
        # x = x.permute(2, 0, 1) # (sequence_length, batch_size, embed_dim)
        
        x = x.permute(1, 0, 2) # (sequence_length, batch_size, embed_dim)
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # (batch_size, sequence_length, embed_dim)

        # Apply temporal blocks
        for block in self.temporal_blocks:
            x = block(x)

        # U-Net style encoder over the temporal dimension
        x = x.permute(0, 2, 1)  # (batch_size, embed_dim, sequence_length)

        # # Decode and output
        x = x[:,:,-1]
        out = self.fc(x).unsqueeze(1)  # (batch_size, 1, output_size)
        
        return out