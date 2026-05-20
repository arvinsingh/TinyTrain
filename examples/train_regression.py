"""End-to-end linear regression training on CPU and GPU (if CuPy available)."""
import numpy as np
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.tensor import Tensor
from core import nn, functional, optim, data, utils

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def generate_data(n=500, d=5, seed=42):
    np.random.seed(seed)
    w_true = np.random.randn(d, 1).astype(np.float32)
    b_true = np.float32(2.5)
    X = np.random.randn(n, d).astype(np.float32)
    y = X @ w_true + b_true + np.random.randn(n, 1).astype(np.float32) * 0.1
    return X, y, w_true, b_true


def train_regression(device='cpu'):
    utils.set_seed(42)
    X, y, w_true, b_true = generate_data()

    split = 400
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    loader = data.DataLoader(X_train, y_train, batch_size=32, shuffle=True, device=device)

    model = nn.Linear(5, 1)
    if device == 'cuda':
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = utils.CosineAnnealingLR(optimizer, T_max=50)

    print(f"\n{'='*50}")
    print(f"Training on {device.upper()} | Params: {utils.param_count(model)}")
    print(f"{'='*50}")

    for epoch in range(50):
        epoch_loss = 0.0
        n_batches = 0
        for bx, by in loader:
            pred = model(bx)
            loss = functional.mse_loss(pred, by)
            model.zero_grad()
            loss.backward()
            utils.clip_grad_norm(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.numpy())
            n_batches += 1
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg = epoch_loss / n_batches
            print(f"  Epoch {epoch+1:3d} | Loss: {avg:.6f} | LR: {optimizer.lr:.6f}")

    if device == 'cuda':
        X_t = Tensor(cp.asarray(X_test))
        y_t = Tensor(cp.asarray(y_test))
    else:
        X_t = Tensor(X_test)
        y_t = Tensor(y_test)
    model.eval()
    pred_test = model(X_t)
    test_loss = functional.mse_loss(pred_test, y_t)
    test_loss_val = float(test_loss.numpy())
    print(f"\n  Test MSE: {test_loss_val:.6f}")

    w_learned = model.weight.numpy().flatten()
    b_learned = model.bias.numpy().flatten()[0]
    w_true_flat = w_true.flatten()
    w_err = np.abs(w_learned - w_true_flat).max()
    b_err = abs(b_learned - b_true)
    print(f"  Weight max error: {w_err:.4f}")
    print(f"  Bias error: {b_err:.4f}")

    save_path = f"/tmp/tinytrain_test_{device}.npz"
    utils.save(model, save_path)
    model2 = nn.Linear(5, 1)
    if device == 'cuda':
        model2.cuda()
    utils.load(model2, save_path)
    pred2 = model2(X_t)
    reload_err = float((pred_test - pred2).data.max())
    print(f"  Save/Load roundtrip max error: {reload_err:.8f}")
    os.remove(save_path)

    assert test_loss_val < 0.05, f"Test loss too high: {test_loss_val}"
    assert w_err < 0.15, f"Weight error too high: {w_err}"
    assert b_err < 0.15, f"Bias error too high: {b_err}"
    assert abs(reload_err) < 1e-6, f"Save/load mismatch: {reload_err}"

    print(f"\n  [PASS] {device.upper()} regression training verified!")
    return True


if __name__ == '__main__':
    cpu_ok = train_regression('cpu')

    if HAS_CUPY:
        gpu_ok = train_regression('cuda')
    else:
        print("\n  [SKIP] GPU test CuPy not installed")

    print(f"\n{'='*50}")
    print("All training tests passed!")
    print(f"{'='*50}")
