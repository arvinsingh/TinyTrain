"""Tests for Triton GPU kernels. Skipped when no CUDA GPU is available."""
import pytest
import numpy as np

from core.kernels import triton_available
from core.kernels.bridge import cupy_to_torch, torch_to_cupy
from core.kernels.matmul import triton_matmul
from core.kernels.attention import triton_flash_attention
from core.kernels.layernorm import triton_layernorm_forward, triton_layernorm_backward
from core.kernels.activations import triton_relu, triton_gelu, triton_relu_bwd, triton_gelu_bwd


torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

triton = pytest.importorskip("triton")


def to_cuda(arr, dtype=torch.float32):
    return torch.tensor(arr, device='cuda', dtype=dtype)


class TestTritonMatmul:
    def test_square(self):
        
        np.random.seed(42)
        a = np.random.randn(64, 64).astype(np.float32)
        b = np.random.randn(64, 64).astype(np.float32)
        a_t, b_t = to_cuda(a), to_cuda(b)
        out = triton_matmul(a_t, b_t)
        ref = a_t @ b_t
        assert torch.allclose(out, ref, atol=5e-2, rtol=1e-2)

    def test_rectangular(self):
        np.random.seed(0)
        a = np.random.randn(32, 128).astype(np.float32)
        b = np.random.randn(128, 64).astype(np.float32)
        a_t, b_t = to_cuda(a), to_cuda(b)
        out = triton_matmul(a_t, b_t)
        ref = a_t @ b_t
        assert torch.allclose(out, ref, atol=5e-2, rtol=1e-2)

    def test_non_power_of_two(self):
        np.random.seed(7)
        a = np.random.randn(37, 53).astype(np.float32)
        b = np.random.randn(53, 41).astype(np.float32)
        a_t, b_t = to_cuda(a), to_cuda(b)
        out = triton_matmul(a_t, b_t)
        ref = a_t @ b_t
        assert torch.allclose(out, ref, atol=5e-2, rtol=1e-2)

    def test_large(self):
        np.random.seed(99)
        a = np.random.randn(256, 512).astype(np.float32)
        b = np.random.randn(512, 256).astype(np.float32)
        a_t, b_t = to_cuda(a), to_cuda(b)
        out = triton_matmul(a_t, b_t)
        ref = a_t @ b_t
        assert torch.allclose(out, ref, atol=5e-2, rtol=1e-2)


class TestTritonFlashAttention:
    def test_basic(self):
        np.random.seed(42)
        BH, N, D = 2, 16, 32
        q = to_cuda(np.random.randn(BH, N, D).astype(np.float32))
        k = to_cuda(np.random.randn(BH, N, D).astype(np.float32))
        v = to_cuda(np.random.randn(BH, N, D).astype(np.float32))
        out, _ = triton_flash_attention(q, k, v, causal=False)
        scale = 1.0 / (D ** 0.5)
        scores = (q @ k.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        ref = attn @ v
        assert torch.allclose(out, ref, atol=5e-2, rtol=1e-2)

    def test_causal(self):
        np.random.seed(7)
        BH, N, D = 1, 32, 32
        q = to_cuda(np.random.randn(BH, N, D).astype(np.float32))
        k = to_cuda(np.random.randn(BH, N, D).astype(np.float32))
        v = to_cuda(np.random.randn(BH, N, D).astype(np.float32))
        out, _ = triton_flash_attention(q, k, v, causal=True)
        scale = 1.0 / (D ** 0.5)
        scores = (q @ k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(N, N, device='cuda'), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        ref = attn @ v
        assert torch.allclose(out, ref, atol=5e-2, rtol=1e-2)

    def test_single_token(self):
        BH, N, D = 1, 1, 64
        q = to_cuda(np.ones((BH, N, D)).astype(np.float32))
        k = to_cuda(np.ones((BH, N, D)).astype(np.float32))
        v = to_cuda(np.ones((BH, N, D)).astype(np.float32))
        out, _ = triton_flash_attention(q, k, v, causal=False)
        # single token attention weight is 1.0, so output == v
        assert torch.allclose(out, v, atol=1e-2)


class TestTritonLayerNorm:
    def test_forward(self):
        np.random.seed(42)
        M, N = 32, 64
        x = to_cuda(np.random.randn(M, N).astype(np.float32))
        w = to_cuda(np.ones(N).astype(np.float32))
        b = to_cuda(np.zeros(N).astype(np.float32))
        y, mean, rstd = triton_layernorm_forward(x, w, b, eps=1e-5)
        ref = torch.nn.functional.layer_norm(x, (N,), weight=w, bias=b, eps=1e-5)
        assert torch.allclose(y, ref, atol=1e-4, rtol=1e-4)

    def test_with_affine(self):
        np.random.seed(0)
        M, N = 16, 128
        x = to_cuda(np.random.randn(M, N).astype(np.float32))
        w = to_cuda(np.random.randn(N).astype(np.float32))
        b = to_cuda(np.random.randn(N).astype(np.float32))
        y, _, _ = triton_layernorm_forward(x, w, b, eps=1e-5)
        ref = torch.nn.functional.layer_norm(x, (N,), weight=w, bias=b, eps=1e-5)
        assert torch.allclose(y, ref, atol=1e-3, rtol=1e-3)

    def test_backward(self):
        np.random.seed(42)
        M, N = 16, 32
        x = to_cuda(np.random.randn(M, N).astype(np.float32))
        w = to_cuda(np.ones(N).astype(np.float32))
        b = to_cuda(np.zeros(N).astype(np.float32))

        y, mean, rstd = triton_layernorm_forward(x, w, b, eps=1e-5)
        dy = to_cuda(np.random.randn(M, N).astype(np.float32))
        dx, dw, db = triton_layernorm_backward(dy, x, w, mean, rstd)

        x_ref = x.clone().requires_grad_(True)
        w_ref = w.clone().requires_grad_(True)
        b_ref = b.clone().requires_grad_(True)
        y_ref = torch.nn.functional.layer_norm(x_ref, (N,), weight=w_ref, bias=b_ref, eps=1e-5)
        y_ref.backward(dy)

        assert torch.allclose(dx, x_ref.grad, atol=1e-3, rtol=1e-3)
        assert torch.allclose(dw, w_ref.grad, atol=1e-2, rtol=1e-2)
        assert torch.allclose(db, b_ref.grad, atol=1e-2, rtol=1e-2)


class TestTritonActivations:
    def test_relu(self):
        
        np.random.seed(42)
        x = to_cuda(np.random.randn(32, 64).astype(np.float32))
        out = triton_relu(x)
        ref = torch.relu(x)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_relu_backward(self):
        np.random.seed(42)
        x = to_cuda(np.random.randn(32, 64).astype(np.float32))
        dy = to_cuda(np.random.randn(32, 64).astype(np.float32))
        dx = triton_relu_bwd(dy, x)
        ref = dy * (x > 0).float()
        assert torch.allclose(dx, ref, atol=1e-6)

    def test_gelu(self):
        np.random.seed(42)
        x = to_cuda(np.random.randn(32, 64).astype(np.float32))
        out = triton_gelu(x)
        ref = torch.nn.functional.gelu(x, approximate='tanh')
        assert torch.allclose(out, ref, atol=1e-3, rtol=1e-3)

    def test_gelu_backward(self):
        np.random.seed(42)
        x = to_cuda(np.random.randn(32, 64).astype(np.float32))
        dy = to_cuda(np.random.randn(32, 64).astype(np.float32))
        dx = triton_gelu_bwd(dy, x)
        x_ref = x.clone().requires_grad_(True)
        y_ref = torch.nn.functional.gelu(x_ref, approximate='tanh')
        y_ref.backward(dy)
        assert torch.allclose(dx, x_ref.grad, atol=1e-3, rtol=1e-3)


class TestTritonBridge:
    def test_availability_detection(self):
        assert isinstance(triton_available(), bool)

    def test_cupy_torch_roundtrip(self):
        cp = pytest.importorskip("cupy")
        arr = cp.random.randn(4, 4).astype(cp.float32)
        t = cupy_to_torch(arr)
        assert t.device.type == 'cuda'
        back = torch_to_cupy(t)
        assert cp.allclose(arr, back)
