# TinyTrain

A minimal deep learning library built from scratch. NumPy backend, CuPy for GPU, Triton for fused kernels. No PyTorch at runtime, just tensors, autograd, and raw CUDA.

## Goal

Build every layer of a training stack by hand: autograd -> ops -> modules -> optimizers -> kernels -> training loops.
Ultimately train a Tiny Transformer from scratch.

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
| Core autograd | DONE | Tensor class (NumPy / CuPy backend) · Topological-sort backward pass · differentiable ops · Broadcasting, batched matmul, slicing | test_ops.py, test_tensor.py; 36 cases |
| NN modules | DONE | Module base (params, zero_grad, train/eval) · Linear, Embedding · LayerNorm, Dropout · Sequential, ReLU, GELU modules | test_nn.py, 12 cases |
| Loss & functional | DONE | Stable softmax / Log softmax · Cross entropy, MSE loss · Scaled dot-product attention | test_functional.py, 7 cases |
| Optimizers | DONE | SGD · AdamW | test_optim.py; 3 cases |
| Data loading | DONE | DataLoader (batch, shuffle, device) · Yields Tensor objects directly | - |
| Triton kernels | DONE | CuPy <-> PyTorch CUDA bridge · Tiled matmul · Flash attention (online softmax, causal) · Fused LayerNorm fwd + bwd · ReLU / GELU fwd + bwd · Auto-dispatch GPU/CPU | test_kernels.py; 16 cases |
| Utils | todo | MHA, FeedForward, TransformerBlock (pre-norm) · Module.cuda() / .cpu(), save/load · Gradient clipping · LR schedulers (Step, Cosine, Warmup) | - |
| End-to-end | todo | Linear regression (CPU + GPU) · Transformer LM · Attention benchmark | - |

## Ops support

- [DONE] Add, Sub, Mul, Neg, Div, pow
- [DONE] Sum, Mean, Max
- [DONE] Reshape, Transpose, Slice, MatMul, cat
- [DONE] Exp, Log, Tanh, Sigmoid
- [DONE] ReLU, GELU

## Optimizers

- [DONE] SGD
- [DONE] AdamW

## Fused Kernels

- [DONE] Tiled MatMul
- [DONE] FlashAttention (Causal)
- [DONE] LayerNorm fwd + bwd
- [DONE] ReLU fwd + bwd
- [DONE] GELU fwd + bwd

## NN Modules

- [DONE] Module base class
- [DONE] Linear
- [DONE] Embedding
- [DONE] Dropout
- [DONE] ReLU, GELU
- [DONE] Multi-Head Attention
- [DONE] Sequential
- [DONE] Feed Forward
- [DONE] Transformer block

## Functional
- [DONE] Softmax / LogSoftmax
- [DONE] Cross-entropy 
- [DONE] MSE loss
- [DONE] Scaled dot-product attention
- [DONE] MaskedFill

## Utils


## Structure

core/ - tensor, autograd, ops, modules, optimizers, kernels
 
tests/ - unit and integration tests per milestone

script/ - model training scripts

example/ - training examples (real/dummy data)