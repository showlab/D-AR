"""
Type definitions for VQ-VAE model outputs.

This module defines dataclasses to replace implicit tuples throughout the codebase,
improving type safety and code clarity.
"""
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class VQCodebookLoss:
    """VQ codebook loss components from vector quantization."""
    vq_loss: torch.Tensor
    commit_loss: torch.Tensor
    entropy_loss: torch.Tensor
    codebook_usage: float
    dead_code_rate: float


@dataclass
class VQQuantizerOutput:
    """Output from vector quantizer forward pass."""
    quantized: torch.Tensor
    loss: VQCodebookLoss
    indices: Optional[torch.Tensor] = None


@dataclass
class DiffusionAux:
    """Auxiliary outputs from diffusion decoder for loss computation."""
    predict_x1: torch.Tensor  # Predicted clean image
    xt: torch.Tensor          # Noisy image at timestep t
    t: torch.Tensor           # Timestep values
    h_repa: torch.Tensor      # REPA features
    fake: torch.Tensor        # Generated/fake samples
    real: torch.Tensor        # Real/target samples


@dataclass
class VQEncoderOutput:
    """Output from VQ encoder."""
    quantized: torch.Tensor
    codebook_loss: VQCodebookLoss
    indices: Optional[torch.Tensor]
    cond: torch.Tensor
    aux_recons: Optional[DiffusionAux] = None


@dataclass
class VQModelOutput:
    """Output from VQ model forward pass during training."""
    reconstructions: DiffusionAux
    codebook_loss: VQCodebookLoss
    diff_loss: torch.Tensor
