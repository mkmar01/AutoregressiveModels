import abc
import torch
import torch.nn as nn

def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        self.token_embedding = nn.Embedding(n_tokens, d_latent)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=8, 
            dim_feedforward=4 * d_latent, 
            activation='gelu', 
            batch_first=False,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=4)
        self.output_layer = nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, h, w = x.shape
        seq_len = h * w

        # Flatten input and embed tokens
        x_flat = x.view(B, seq_len)  # (B, seq_len)
        x_embedded = self.token_embedding(x_flat)  # (B, seq_len, d_latent)

        # Shift input by one position
        start_token = torch.zeros((B, 1, self.d_latent), device=x.device)  # (B, 1, d_latent)
        x_shifted = torch.cat([start_token, x_embedded[:, :-1, :]], dim=1)  # (B, seq_len, d_latent)

        # Permute for transformer input
        x_permuted = x_shifted.permute(0, 1) # (seq_len, B, d_latent)

        mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device) # (seq_len, seq_len)

        # Pass through transformer encoder
        transformer_output = self.encoder(x_permuted, mask=mask)  # (seq_len, B, d_latent)

        # Permute back
        transformer_output = transformer_output.transpose(0, 1)  # (B, seq_len, d_latent)

        # Output layer to get logits for each token
        logits = self.output_layer(transformer_output)  # (B, seq_len, n_tokens)

        # Reshape logits to (B, h, w, n_tokens)
        logits_reshaped = logits.view(B, h, w, self.n_tokens)  # (B, h, w, n_tokens)

        return logits_reshaped, {}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        raise NotImplementedError()
