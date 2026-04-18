# TinyTrain

A minimal deep learning framework built from scratch. NumPy backend, CuPy for GPU, Triton for fused kernels. No PyTorch at runtime, just tensors, autograd, and raw CUDA.

## Goal

Build every layer of a training stack by hand: autograd -> ops -> modules -> optimizers -> kernels -> training loops.
Ultimately train a Tiny Language Model.

## Requirements

- Python ≥ 3.10, NumPy
- GPU - CuPy, Triton
- Testing - PyTorch (reference oracle), pytest

## Setup

```bash
uv sync
```

## Roadmap

| Modules | Status | Features | Tests |
|---|---|---|---|
| Core autograd | todo | Tensor class (NumPy / CuPy backend) · Topological-sort backward pass · differentiable ops · Broadcasting, batched matmul, slicing | - |
| NN modules | todo | Module base (params, zero_grad, train/eval) · Linear, Embedding · LayerNorm, Dropout · Sequential, ReLU, GELU modules | - |
| Loss & functional | todo | Stable softmax / Log softmax · Cross entropy, MSE loss · Scaled dot-product attention | - |
| Optimizers | todo | SGD · AdamW | - |
| Data loading | todo | DataLoader (batch, shuffle, device) · Yields Tensor objects directly | - |
| Triton kernels | todo | CuPy <-> PyTorch CUDA bridge · Tiled matmul · Flash attention (online softmax, causal) · Fused LayerNorm fwd + bwd · ReLU / GELU fwd + bwd · Auto-dispatch GPU/CPU | - |
| Transformer utils | todo | MHA, FeedForward, TransformerBlock (pre-norm) · Module.cuda() / .cpu(), save/load · Gradient clipping · LR schedulers (Step, Cosine, Warmup) | - |
| End-to-end | todo | Linear regression (CPU + GPU) · Transformer LM · Attention benchmark | - |

## Structure

core/ - tensor, autograd, ops, modules, optimizers, kernels
 
tests/ - unit and integration tests per milestone

script/ - model training scripts