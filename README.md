# TinyTrain

A minimal deep learning library built from scratch. NumPy backend, CuPy for GPU, Triton for fused kernels. No PyTorch at runtime, just tensors, autograd, and raw CUDA.


## Requirements

- Python ≥ 3.10, NumPy
- GPU - CuPy, Triton
- Testing - PyTorch (reference oracle), pytest

## Setup

```bash
uv sync
```
For GPU extra
```bash
uv sync --extra gpu --extra triton
```

## Roadmap


| Module | Status | Features | Tests |
|---|---|---|---|
| Core autograd | DONE | Tensor class · Topological-sort backward pass · Scalar/broadcast autograd · Batched matmul · Slicing · No-grad mode · Numerical gradient stress tests · Randomized shape tests · Broadcasting stress tests | `test_tensor.py`, `test_ops.py`, `test_autograd_stress.py`; 199 cases |
| NN modules | DONE | Module base · Parameter tracking · `zero_grad()` · Linear · LayerNorm · Embedding · Dropout · Sequential · Reproducibility | `test_nn.py`; 12 cases |
| Loss & functional | DONE | Softmax · LogSoftmax · Cross entropy · Numerical stability · Backward checks | `test_functional.py`; 7 cases |
| Optimizers | DONE | SGD · Adam/AdamW-style optimizer · Multi-step optimizer behavior | `test_optim.py`; 3 cases |
| Triton kernels | DONE | CuPy <-> PyTorch CUDA bridge · Tiled matmul · Flash attention (online softmax, causal) · Fused LayerNorm fwd + bwd · ReLU / GELU fwd + bwd · Auto-dispatch GPU/CPU | `test_kernels.py`; 16 cases |
| Utils | DONE | Save/load · Gradient norm clipping · Gradient value clipping · StepLR · CosineAnnealingLR · LinearWarmupCosineDecay · Parameter counting through integration tests | `test_integration.py`; 8 utility-related cases |
| Data loading | DONE | DataLoader yields Tensor batches · Regression training integration | `test_integration.py`; 2 cases |
| End-to-end | DONE | MLP XOR training · TransformerBlock loss decrease · Deep gradient flow · MSE loss · MultiHeadAttention forward/backward · TransformerBlock shape/training/param count | `test_integration.py`; 11 model/training cases |

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

- [DONE] Seeding
- [DONE] Device detection
- [DONE] Parameter counting
- [DONE] Save / load checkpoints
- [DONE] Gradient norm clipping
- [DONE] Gradient value clipping
- [DONE] StepLR
- [DONE] CosineAnnealingLR
- [DONE] LinearWarmupCosineDecay

---
