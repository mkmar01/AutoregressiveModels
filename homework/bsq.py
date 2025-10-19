import abc

import torch
import torch.nn.functional as F

from .ae import PatchAutoEncoder


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        self._embedding_dim = embedding_dim

        self.down = torch.nn.Linear(embedding_dim, codebook_bits, bias=True)
        self.up   = torch.nn.Linear(codebook_bits, embedding_dim, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        y = self.down(x)                                # (..., m)
        y = F.normalize(y, p=2, dim=-1, eps=1e-8)       # on unit sphere
        c = diff_sign(y)                                # binary codes with STE
        return c

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        return self.up(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        self.codebook_bits = codebook_bits
        self.bsq = BSQ(embedding_dim=latent_dim, codebook_bits=codebook_bits)

    def encode_index(self, x: torch.Tensor) -> torch.Tensor: # tokenizer
        z = self.encode(x)                          # (B,h,w,D)
        B, h, w, D = z.shape
        c = self.bsq.encode(z.view(B, h*w, D)) # (B,L,bits)
        idx = self.bsq._code_to_index(c)             # (B,L)
        return idx.view(B, h, w)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor: # tokenizer
        B, h, w = x.shape
        c = self.bsq._index_to_code(x.view(B, h*w))      # (B,L,bits)
        z = self.bsq.decode(c)                     # (B,L,D)
        z = z.view(B, h, w, -1)                         # (B,h,w,D)
        return self.decode(z)                            # (B,H,W,3)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return super().encode(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == self.codebook_bits:
            # treat as BSQ code
            B, h, w, _ = x.shape
            z = self.bsq.decode(x.view(B, h*w, -1)).view(B, h, w, -1)
            return super().decode(z)
        else:
            # latent to image
            return super().decode(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        Hint: It can be helpful to monitor the codebook usage with

              cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self.codebook_bits)

              and returning

              {
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                ...
              }
        """
        # --- Differentiable BSQ bottleneck path (uses STE) ---
        z = super().encode(x)  # (B, h, w, D)
        B, h, w, D = z.shape

        c = self.bsq.encode(z.view(B, h * w, D)).view(B, h, w, self.codebook_bits)  # (B, h, w, bits)
        x_rec = self.decode(c)  # (B, H, W, 3)

        # --- Monitor codebook usage (token histogram) ---
        with torch.no_grad():
            tokens = self.encode_index(x).view(B, h * w)  # (B, L)
            K = 2 ** self.codebook_bits
            cnt = torch.bincount(tokens.flatten(), minlength=K)  # (K,)

            logs = {
                # Fractions of codes that are never / rarely used
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                # optional extras:
                "token_usage_frac": (cnt > 0).float().mean().detach(),  # how many codes ever used
            }

        return x_rec, logs
