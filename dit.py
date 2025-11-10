import math
import torch
import torch.nn as nn
from typing import Optional

def modulate(x, shift, scale):
    """
    Modulates the input tensor 'x' using a scale and shift.
    This is the core of Adaptive Layer Normalization (adaLN).
    """
    # The unsqueeze is to make the scale and shift broadcastable to the sequence length.
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds a discrete timestep t into a continuous vector.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Creates sinusoidal timestep embeddings.
        Args:
            t (torch.Tensor): A 1-D Tensor of N indices, one per batch element.
            dim (int): The dimension of the output.
            max_period (int): The maximum period for the sinusoidal embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class TransformerBlock(nn.Module):
    """
    A standard Transformer block with Adaptive Layer Normalization (adaLN).
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        # This single MLP layer generates all the conditioning parameters (scale/shift)
        # for the entire block. 2 for norm1, 2 for norm2.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, adaln_input: torch.Tensor) -> torch.Tensor:
        # Generate scale and shift parameters from the timestep embedding
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

        # Attention block with adaLN
        x_norm1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + attn_output

        # MLP block with adaLN
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(x_norm2)
        x = x + mlp_output

        return x

class FinalLayer(nn.Module):
    """
    The final layer of the DiT, which projects the sequence of vectors
    back to the vocabulary space (logits).
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class ChessDiT(nn.Module):
    """
    A Diffusion Transformer (DiT) specifically for chess board generation.
    This model predicts the clean board x_0 from a noisy input x_t.

    Args:
        vocab_size (int): The size of the vocabulary (14 for chess: 12 pieces + empty + absorbing).
        hidden_size (int): The dimensionality of the model (D).
        depth (int): The number of Transformer blocks.
        num_heads (int): The number of attention heads.
        mlp_ratio (float): The ratio for the MLP's hidden dimension in Transformer blocks.
    """
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # 1. Input Embedders
        self.piece_embedder = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedder = nn.Parameter(torch.randn(1, sequence_length, hidden_size)) # Learnable positional embeddings
        self.t_embedder = TimestepEmbedder(hidden_size)

        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # 3. Final Output Layer
        self.final_layer = FinalLayer(hidden_size, vocab_size)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embedding and token embedding
        nn.init.normal_(self.pos_embedder, std=0.02)
        nn.init.normal_(self.piece_embedder.weight, std=0.02)

        # Initialize all Linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out output layers:
        # The last layer of each MLP block
        for block in self.blocks:
            nn.init.constant_(block.mlp[-1].bias, 0)
            nn.init.constant_(block.mlp[-1].weight, 0)
        # The final projection layer
        nn.init.constant_(self.final_layer.linear.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the ChessDiT model.

        Args:
            x_t (torch.Tensor): The noisy input board tensor of shape (B, 64).
            t (torch.Tensor): The timestep tensor of shape (B,).
            cond (Optional[torch.Tensor]): Unused in this unconditional model, kept for API consistency.

        Returns:
            torch.Tensor: The predicted logits for the clean board x_0, of shape (B, 64, vocab_size).
        """
        # (B, 64) -> (B, 64, D)
        x_emb = self.piece_embedder(x_t)
        # (B,) -> (B, D)
        t_emb = self.t_embedder(t)

        # Add positional embeddings
        x = x_emb + self.pos_embedder  # (B, 64, D)

        # Process through Transformer blocks
        for block in self.blocks:
            x = block(x, adaln_input=t_emb)

        # Final projection to get logits
        # The final layer is also conditioned on the timestep
        logits = self.final_layer(x, t_emb) # (B, 64, D) -> (B, 64, vocab_size)

        return logits
