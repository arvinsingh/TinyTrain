"""Train a small Transformer model on random token sequences (CPU & GPU)."""
import time
import numpy as np
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tensor import Tensor
from core import nn, functional, optim, utils

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


class TinyTransformerLM(nn.Module):
    """Decoder-only transformer language model."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = []
        for i in range(n_layers):
            block = nn.TransformerBlock(d_model, n_heads)
            setattr(self, f"block_{i}", block)
            self.blocks.append(block)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        xp = self.token_emb.weight.xp
        pos = xp.arange(T).astype(np.int64)
        tok = self.token_emb(idx)
        pos = self.pos_emb(Tensor(pos))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


def generate_data(n_samples, seq_len, vocab_size, seed=42):
    rng = np.random.RandomState(seed)
    tokens = rng.randint(0, vocab_size, size=(n_samples, seq_len)).astype(np.int64)
    return tokens


def train(device='cpu'):
    vocab_size = 64
    d_model = 64
    n_heads = 4
    n_layers = 2
    seq_len = 32
    batch_size = 16
    n_samples = 256
    n_epochs = 20
    lr = 3e-3

    utils.set_seed(42)

    data = generate_data(n_samples, seq_len + 1, vocab_size)
    inputs = data[:, :-1]
    targets = data[:, 1:]

    model = TinyTransformerLM(vocab_size, d_model, n_heads, n_layers, seq_len)
    if device == 'cuda':
        model.cuda()

    n_params = utils.param_count(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = utils.CosineAnnealingLR(optimizer, T_max=n_epochs)

    print(f"\n{'='*60}")
    print(f"  Transformer LM  |  {device.upper()}  |  {n_params:,} params")
    print(f"  vocab={vocab_size}  d_model={d_model}  heads={n_heads}  layers={n_layers}")
    print(f"  seq_len={seq_len}  batch={batch_size}  epochs={n_epochs}  lr={lr}")
    print(f"{'='*60}")

    losses = []
    t_start = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        idx = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            batch_idx = idx[start:start + batch_size]
            bx = inputs[batch_idx]
            by = targets[batch_idx]

            if device == 'cuda':
                bx = cp.asarray(bx)
                by = cp.asarray(by)

            x_t = Tensor(bx)
            logits = model(x_t)

            B, T, V = logits.shape
            logits_flat = logits.reshape(B * T, V)
            targets_flat = by.ravel()

            loss = functional.cross_entropy(logits_flat, targets_flat)

            model.zero_grad()
            loss.backward()
            utils.clip_grad_norm(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.numpy())
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.perf_counter() - t_start
            print(f"  Epoch {epoch:3d} | Loss: {avg_loss:.4f} | LR: {scheduler.get_lr():.6f} | Time: {elapsed:.1f}s")

    elapsed = time.perf_counter() - t_start
    print(f"\n  Final loss: {losses[-1]:.4f}  (started at {losses[0]:.4f})")
    print(f"  Total time: {elapsed:.1f}s  ({elapsed/n_epochs:.2f}s/epoch)")

    improved = losses[-1] < losses[0]
    ratio = losses[0] / losses[-1]
    print(f"  Loss ratio (first/last): {ratio:.2f}x")
    print(f"  [{'PASS' if improved else 'FAIL'}] Loss decreased during training")

    path = f"/tmp/tinytrain_transformer_{device}.npz"
    utils.save(model, path)
    model2 = TinyTransformerLM(vocab_size, d_model, n_heads, n_layers, seq_len)
    if device == 'cuda':
        model2.cuda()
    utils.load(model2, path)
    p1 = [p.numpy() for p in model.parameters()]
    p2 = [p.numpy() for p in model2.parameters()]
    max_err = max(np.max(np.abs(a - b)) for a, b in zip(p1, p2))
    print(f"  Save/Load roundtrip max error: {max_err:.8e}")
    print(f"  [{'PASS' if max_err == 0 else 'FAIL'}] Save/Load roundtrip\n")

    return improved


if __name__ == "__main__":
    cpu_ok = train('cpu')

    if HAS_CUPY:
        gpu_ok = train('cuda')
    else:
        gpu_ok = True
        print("\n  [SKIP] GPU training CuPy not installed\n")

    print("=" * 60)
    if cpu_ok and gpu_ok:
        print("  All transformer training tests passed!")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)
