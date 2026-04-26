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
| Core autograd | DONE | Tensor class (NumPy / CuPy backend) · Topological-sort backward pass · differentiable ops · Broadcasting, batched matmul, slicing | test_ops.py, test_tensor.py; 36 test cases |
| NN modules | todo | Module base (params, zero_grad, train/eval) · Linear, Embedding · LayerNorm, Dropout · Sequential, ReLU, GELU modules | - |
| Loss & functional | todo | Stable softmax / Log softmax · Cross entropy, MSE loss · Scaled dot-product attention | - |
| Optimizers | DONE | SGD · AdamW | test_optim.py; 3 test cases |
| Data loading | todo | DataLoader (batch, shuffle, device) · Yields Tensor objects directly | - |
| Triton kernels | In progress | CuPy <-> PyTorch CUDA bridge · Tiled matmul · Flash attention (online softmax, causal) · Fused LayerNorm fwd + bwd · ReLU / GELU fwd + bwd · Auto-dispatch GPU/CPU | - |
| Transformer utils | todo | MHA, FeedForward, TransformerBlock (pre-norm) · Module.cuda() / .cpu(), save/load · Gradient clipping · LR schedulers (Step, Cosine, Warmup) | - |
| End-to-end | todo | Linear regression (CPU + GPU) · Transformer LM · Attention benchmark | - |

## Ops support

- [DONE] Add
- [DONE] Mul
- [DONE] Neg
- [DONE] Div
- [DONE] Pow
- [DONE] Sum
- [DONE] Mean
- [DONE] Max
- [DONE] Reshape
- [DONE] Transpose
- [DONE] Slice
- [DONE] Exp
- [DONE] Log
- [DONE] Tanh
- [DONE] Sigmoid
- [DONE] cat
- [DONE] MatMul
- [DONE] ReLU
- [DONE] GeLU

## Optimizers

- [DONE] Stochastic Gradient Descent
- [DONE] AdamW

## Fused Kernels

- [TBD] Tiled MatMul
- [TBD] FlashAttention (Causal)
- [TBD] LayerNorm
- [TBD] ReLU
- [TBD] GELU

## NN Modules

- [TBD] Module base class
- [TBD] Linear
- [TBD] Embedding
- [TBD] Dropout
- [TBD] ReLU
- [TBD] GELU
- [TBD] Multi-Head Attention
- [TBD] Sequential
- [TBD] Feed Forward
- [TBD] Transformer block

## Functional [TBD]

## Utils [TBD]

## Structure

core/ - tensor, autograd, ops, modules, optimizers, kernels
 
tests/ - unit and integration tests per milestone

script/ - model training scripts

example/ - training examples (real/dummy data)