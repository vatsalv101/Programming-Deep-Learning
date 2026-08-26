import torch
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────
# 1. ReLU
# ────────────────────────────────────────────────────────────────

def test_relu(relu_fn):
    """Tests relu for correctness, shape preservation, and edge cases."""

    all_ok = True

    # Basic cases
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    y = relu_fn(x)
    if not torch.allclose(y, expected):
        print(f"❌ FAIL: relu({x.tolist()}) = {y.tolist()}, expected {expected.tolist()}")
        all_ok = False

    # Shape preservation with matrix
    x_mat = torch.randn(4, 5)
    y_mat = relu_fn(x_mat)
    if y_mat.shape != x_mat.shape:
        print(f"❌ FAIL: relu output shape {y_mat.shape} != input shape {x_mat.shape}")
        all_ok = False

    # No negative outputs
    if (y_mat < 0).any():
        print("❌ FAIL: relu produced negative values")
        all_ok = False

    # Positive values unchanged
    ref = torch.clamp(x_mat, min=0)
    if not torch.allclose(y_mat, ref):
        print("❌ FAIL: relu disagrees with torch.clamp(x, min=0)")
        all_ok = False

    if all_ok:
        print("✅ PASS: relu is correct.")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 2. ReLU backward
# ────────────────────────────────────────────────────────────────

def test_relu_backward(relu_fn, relu_backward_fn):
    """Tests relu_backward via numerical gradient check."""

    all_ok = True
    torch.manual_seed(42)
    x = torch.randn(5, 4, dtype=torch.float64) * 10
    ## Avoid values near zero where ReLU is non-differentiable
    x += (x.abs() < 0.05).to(x.dtype) * torch.sign(x) * 1.0

    dout = torch.randn(5, 4, dtype=torch.float64) * 10

    dx_analytical = relu_backward_fn(dout, x)

    # Numerical gradient
    eps = 1e-5
    dx_numerical = (relu_fn(x + eps) - relu_fn(x - eps)) / (2 * eps)
    dx_numerical = dout * dx_numerical

    if not torch.allclose(dx_analytical, dx_numerical, atol=1e-4):
        max_err = (dx_analytical - dx_numerical).abs().max().item()
        #print max error and excpected value for the max value
        print(f"Analytical dx: {dx_analytical}")
        print(f"Numerical dx: {dx_numerical}")
        print(f"❌ FAIL: relu_backward max error = {max_err:.2e}")
        print("   💡 Remember: relu_backward uses the activation output (or any equivalent mask input).")
        all_ok = False

    # Specific: zero gradient where x < 0
    x_neg = torch.tensor([-1.0, -5.0, -0.1])
    dout_neg = torch.ones(3)
    if not torch.allclose(relu_backward_fn(dout_neg, x_neg), torch.zeros(3)):
        print("❌ FAIL: relu_backward should be zero for negative inputs")
        all_ok = False

    if all_ok:
        print("✅ PASS: relu_backward is correct (matches numerical gradient).")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 3. Tanh forward
# ────────────────────────────────────────────────────────────────

def test_tanh_forward(tanh_fn):
    """Tests tanh_forward against torch.tanh."""

    all_ok = True

    x = torch.linspace(-5, 5, 50)
    y = tanh_fn(x)
    ref = torch.tanh(x)
    if not torch.allclose(y, ref, atol=1e-6):
        max_err = (y - ref).abs().max().item()
        print(f"❌ FAIL: tanh_forward max error = {max_err:.2e} vs torch.tanh")
        all_ok = False

    # Edge: large values should saturate near ±1
    large = tanh_fn(torch.tensor([30.0, -30.0]))
    if not (abs(large[0].item() - 1.0) < 1e-3 and abs(large[1].item() + 1.0) < 1e-3):
        print("❌ FAIL: tanh_forward should saturate to ±1 for large inputs")
        all_ok = False

    # Shape preservation
    x_mat = torch.randn(3, 4)
    if tanh_fn(x_mat).shape != x_mat.shape:
        print("❌ FAIL: tanh_forward changed the shape")
        all_ok = False

    if all_ok:
        print("✅ PASS: tanh_forward is correct.")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 4. Tanh backward
# ────────────────────────────────────────────────────────────────

def test_tanh_backward(tanh_fn, tanh_backward_fn):
    """Tests tanh_backward via numerical gradient check."""

    all_ok = True
    x = torch.linspace(-3, 3, 100, dtype=torch.float64)
    dout = torch.ones_like(x)

    tanh_out = tanh_fn(x)
    dx_analytical = tanh_backward_fn(dout, tanh_out)

    # Numerical gradient
    eps = 1e-5
    dx_numerical = (tanh_fn(x + eps) - tanh_fn(x - eps)) / (2 * eps)

    if not torch.allclose(dx_analytical, dx_numerical, atol=1e-4):
        max_err = (dx_analytical - dx_numerical).abs().max().item()
        print(f"❌ FAIL: tanh_backward max error = {max_err:.2e}")
        print("   💡 Remember: tanh_backward takes the tanh *output*, not the raw input x.")
        print("   The formula is: dout * (1 - tanh_out^2)")
        all_ok = False

    if all_ok:
        print("✅ PASS: tanh_backward is correct (matches numerical gradient).")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 5. Linear forward
# ────────────────────────────────────────────────────────────────

def test_linear_forward(linear_forward_fn):
    """Tests linear_forward against x @ W + b."""

    all_ok = True
    torch.manual_seed(0)

    # Single-sample
    x1 = torch.randn(1, 5)
    W1 = torch.randn(5, 3)
    b1 = torch.randn(3)
    out1, cache1 = linear_forward_fn(x1, W1, b1)
    ref1 = x1 @ W1 + b1
    if not torch.allclose(out1, ref1, atol=1e-6):
        print(f"❌ FAIL: linear_forward single-sample output differs from x @ W + b")
        all_ok = False
    if out1.shape != (1, 3):
        print(f"❌ FAIL: linear_forward output shape {out1.shape}, expected (1, 3)")
        all_ok = False

    # Batch
    x2 = torch.randn(8, 10)
    W2 = torch.randn(10, 7)
    b2 = torch.randn(7)
    out2, cache2 = linear_forward_fn(x2, W2, b2)
    ref2 = x2 @ W2 + b2
    if not torch.allclose(out2, ref2, atol=1e-6):
        print(f"❌ FAIL: linear_forward batch output differs from x @ W + b")
        all_ok = False
    if out2.shape != (8, 7):
        print(f"❌ FAIL: linear_forward output shape {out2.shape}, expected (8, 7)")
        all_ok = False

    # Cache check
    if cache2[0] is not x2 or cache2[1] is not W2 or cache2[2] is not b2:
        print("❌ FAIL: linear_forward cache should store (x, W, b)")
        all_ok = False

    if all_ok:
        print("✅ PASS: linear_forward is correct.")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 6. Linear backward
# ────────────────────────────────────────────────────────────────

def test_linear_backward(linear_forward_fn, linear_backward_fn):
    """Tests linear_backward against PyTorch autograd."""

    all_ok = True
    torch.manual_seed(1)

    x = torch.randn(4, 6, requires_grad=True) * 10
    W = torch.randn(6, 3, requires_grad=True) * 10
    b = torch.randn(3, requires_grad=True) * 10
    dout = torch.randn(4, 3) * 10

    x.retain_grad()
    W.retain_grad()
    b.retain_grad()

    # Reference via autograd
    out_ref = x @ W + b
    out_ref.backward(dout)
    dx_ref, dW_ref, db_ref = x.grad.clone(), W.grad.clone(), b.grad.clone()

    # Student
    x_plain = x.detach().clone()
    W_plain = W.detach().clone()
    b_plain = b.detach().clone()
    out_student, cache = linear_forward_fn(x_plain, W_plain, b_plain)
    dx, dW, db = linear_backward_fn(dout, cache)

    for name, student, ref in [('dx', dx, dx_ref), ('dW', dW, dW_ref), ('db', db, db_ref)]:
        if student.shape != ref.shape:
            print(f"❌ FAIL: linear_backward {name} shape {student.shape}, expected {ref.shape}")
            all_ok = False
        elif not torch.allclose(student, ref, atol=1e-5):
            max_err = (student - ref).abs().max().item()
            print(f"❌ FAIL: linear_backward {name} max error = {max_err:.2e}")
            all_ok = False

    if all_ok:
        print("✅ PASS: linear_backward is correct (matches PyTorch autograd).")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 7. Softmax
# ────────────────────────────────────────────────────────────────

def test_softmax(softmax_fn):
    """Tests softmax against F.softmax."""

    all_ok = True
    torch.manual_seed(2)

    logits = torch.randn(8, 10)
    probs = softmax_fn(logits)
    ref = F.softmax(logits, dim=1)

    if probs.shape != ref.shape:
        print(f"❌ FAIL: softmax output shape {probs.shape}, expected {ref.shape}")
        all_ok = False

    if not torch.allclose(probs, ref, atol=1e-6):
        max_err = (probs - ref).abs().max().item()
        print(f"❌ FAIL: softmax max error = {max_err:.2e} vs F.softmax")
        all_ok = False

    # Rows sum to 1
    row_sums = probs.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones(8), atol=1e-5):
        print(f"❌ FAIL: softmax rows don't sum to 1: {row_sums.tolist()}")
        all_ok = False

    # All positive
    if (probs < 0).any():
        print("❌ FAIL: softmax produced negative values")
        all_ok = False

    if all_ok:
        print("✅ PASS: softmax is correct.")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 8. Cross-entropy loss
# ────────────────────────────────────────────────────────────────

def test_cross_entropy_loss(softmax_fn, loss_fn):
    """Tests cross_entropy_loss against F.cross_entropy and autograd gradients."""

    all_ok = True
    torch.manual_seed(3)

    logits = torch.randn(16, 10, requires_grad=True)
    targets = torch.randint(0, 10, (16,))

    # Reference loss
    ref_loss = F.cross_entropy(logits, targets)

    # Reference gradient
    ref_loss.backward()
    dlogits_ref = logits.grad.clone()

    # Student
    logits_plain = logits.detach().clone()
    probs = softmax_fn(logits_plain)
    loss, dlogits = loss_fn(probs, targets)

    if abs(loss - ref_loss.item()) > 1e-4:
        print(f"❌ FAIL: cross_entropy_loss value {loss:.6f}, expected {ref_loss.item():.6f}")
        all_ok = False

    if dlogits.shape != dlogits_ref.shape:
        print(f"❌ FAIL: cross_entropy_loss gradient shape {dlogits.shape}, expected {dlogits_ref.shape}")
        all_ok = False
    elif not torch.allclose(dlogits, dlogits_ref, atol=1e-5):
        max_err = (dlogits - dlogits_ref).abs().max().item()
        print(f"❌ FAIL: cross_entropy_loss gradient max error = {max_err:.2e}")
        print("   💡 The gradient w.r.t. logits should be (probs - one_hot) / batch_size")
        all_ok = False

    if all_ok:
        print("✅ PASS: cross_entropy_loss is correct (matches F.cross_entropy).")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 9. Init model
# ────────────────────────────────────────────────────────────────

def test_init_model(init_fn):
    """Tests init_model for correct shapes, zero biases, and weight scale."""

    all_ok = True

    params = init_fn()

    expected_shapes = {
        'W1': (784, 256), 'b1': (256,),
        'W2': (256, 128), 'b2': (128,),
        'W3': (128, 64),  'b3': (64,),
        'W4': (64, 10),   'b4': (10,),
    }

    for key, shape in expected_shapes.items():
        if key not in params:
            print(f"❌ FAIL: params missing key '{key}'")
            all_ok = False
            continue
        if params[key].shape != torch.Size(shape):
            print(f"❌ FAIL: params['{key}'].shape = {tuple(params[key].shape)}, expected {shape}")
            all_ok = False

    # Biases should be zero
    for bkey in ['b1', 'b2', 'b3', 'b4']:
        if bkey in params and params[bkey].abs().sum().item() > 1e-8:
            print(f"❌ FAIL: params['{bkey}'] should be all zeros")
            all_ok = False

    # Weights should be small (~0.1 scale)
    for wkey in ['W1', 'W2', 'W3', 'W4']:
        if wkey in params:
            std = params[wkey].std().item()
            if std < 0.01 or std > 0.5:
                print(f"❌ FAIL: params['{wkey}'].std() = {std:.4f}, expected ~0.1")
                all_ok = False

    if all_ok:
        print("✅ PASS: init_model is correct.")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 10. Forward pass
# ────────────────────────────────────────────────────────────────

def test_forward_pass(forward_fn, init_fn):
    """Tests forward_pass output shape and values against manual computation."""

    all_ok = True
    torch.manual_seed(5)
    params = init_fn()
    X = torch.randn(4, 784)

    # Test with ReLU
    from mlp_layers import relu_forward
    logits, cache = forward_fn(X, params, relu_forward)

    if logits.shape != (4, 10):
        print(f"❌ FAIL: forward_pass output shape {logits.shape}, expected (4, 10)")
        all_ok = False

    # Manual reference computation
    z1 = X @ params['W1'] + params['b1']
    a1 = torch.clamp(z1, min=0)
    z2 = a1 @ params['W2'] + params['b2']
    a2 = torch.clamp(z2, min=0)
    z3 = a2 @ params['W3'] + params['b3']
    a3 = torch.clamp(z3, min=0)
    z4 = a3 @ params['W4'] + params['b4']

    if not torch.allclose(logits, z4, atol=1e-5):
        max_err = (logits - z4).abs().max().item()
        print(f"❌ FAIL: forward_pass (relu) max error = {max_err:.2e} vs manual computation")
        all_ok = False

    # Test with tanh
    from mlp_layers import tanh_forward
    logits_t, _ = forward_fn(X, params, tanh_forward)
    z1t = X @ params['W1'] + params['b1']
    a1t = torch.tanh(z1t)
    z2t = a1t @ params['W2'] + params['b2']
    a2t = torch.tanh(z2t)
    z3t = a2t @ params['W3'] + params['b3']
    a3t = torch.tanh(z3t)
    z4t = a3t @ params['W4'] + params['b4']

    if not torch.allclose(logits_t, z4t, atol=1e-5):
        max_err = (logits_t - z4t).abs().max().item()
        print(f"❌ FAIL: forward_pass (tanh) max error = {max_err:.2e} vs manual computation")
        all_ok = False

    if cache is None or not isinstance(cache, dict):
        print("❌ FAIL: forward_pass should return a cache dict")
        all_ok = False

    if all_ok:
        print("✅ PASS: forward_pass is correct.")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 11. Backward pass
# ────────────────────────────────────────────────────────────────

def test_backward_pass(forward_fn, backward_fn, softmax_fn, loss_fn, init_fn):
    """Tests backward_pass gradients against PyTorch autograd."""

    all_ok = True
    torch.manual_seed(7)

    params = init_fn()
    X = torch.randn(8, 784)
    targets = torch.randint(0, 10, (8,))

    # --- Build autograd reference ---
    W1 = params['W1'].clone().requires_grad_(True)
    b1 = params['b1'].clone().requires_grad_(True)
    W2 = params['W2'].clone().requires_grad_(True)
    b2 = params['b2'].clone().requires_grad_(True)
    W3 = params['W3'].clone().requires_grad_(True)
    b3 = params['b3'].clone().requires_grad_(True)
    W4 = params['W4'].clone().requires_grad_(True)
    b4 = params['b4'].clone().requires_grad_(True)

    z1 = X @ W1 + b1
    a1 = torch.clamp(z1, min=0)
    z2 = a1 @ W2 + b2
    a2 = torch.clamp(z2, min=0)
    z3 = a2 @ W3 + b3
    a3 = torch.clamp(z3, min=0)
    z4 = a3 @ W4 + b4
    ref_loss = F.cross_entropy(z4, targets)
    ref_loss.backward()

    ref_grads = {
        'dW1': W1.grad.clone(), 'db1': b1.grad.clone(),
        'dW2': W2.grad.clone(), 'db2': b2.grad.clone(),
        'dW3': W3.grad.clone(), 'db3': b3.grad.clone(),
        'dW4': W4.grad.clone(), 'db4': b4.grad.clone(),
    }

    # --- Student computation ---
    from mlp_layers import relu_forward, relu_backward
    logits, cache = forward_fn(X, params, relu_forward)
    probs = softmax_fn(logits)
    loss, dlogits = loss_fn(probs, targets)
    grads = backward_fn(dlogits, cache, relu_backward)

    for key in ref_grads:
        if key not in grads:
            print(f"❌ FAIL: grads missing key '{key}'")
            all_ok = False
            continue
        if grads[key].shape != ref_grads[key].shape:
            print(f"❌ FAIL: grads['{key}'] shape {grads[key].shape}, expected {ref_grads[key].shape}")
            all_ok = False
        elif not torch.allclose(grads[key], ref_grads[key], atol=1e-4):
            max_err = (grads[key] - ref_grads[key]).abs().max().item()
            print(f"❌ FAIL: grads['{key}'] max error = {max_err:.2e}")
            all_ok = False

    if all_ok:
        print("✅ PASS: backward_pass is correct (matches PyTorch autograd).")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 12. Compute accuracy
# ────────────────────────────────────────────────────────────────

def test_compute_accuracy(accuracy_fn, forward_fn, init_fn):
    """Tests compute_accuracy returns a sensible value."""

    all_ok = True
    torch.manual_seed(8)

    from mlp_layers import relu_forward
    params = init_fn()

    # Random data — accuracy should be near 10% (random guessing, 10 classes)
    X = torch.randn(200, 784)
    y = torch.randint(0, 10, (200,))
    acc = accuracy_fn(X, y, params, forward_fn, relu_forward)

    if not isinstance(acc, float):
        print(f"❌ FAIL: compute_accuracy should return a float, got {type(acc)}")
        all_ok = False
    elif acc < 0 or acc > 1:
        print(f"❌ FAIL: accuracy {acc:.4f} not in [0, 1]")
        all_ok = False
    elif acc > 0.3:
        print(f"❌ FAIL: random model accuracy = {acc:.2%}, expected ~10% (random guessing)")
        all_ok = False

    # Perfect model: logits = one-hot of labels
    X_dummy = torch.randn(50, 784)
    y_dummy = torch.randint(0, 10, (50,))

    class _PerfectForward:
        """Fake forward that returns perfect logits."""
        def __call__(self, X, params, act_fn):
            logits = torch.zeros(X.shape[0], 10)
            logits[torch.arange(X.shape[0]), y_dummy] = 10.0
            return logits, None

    acc_perfect = accuracy_fn(X_dummy, y_dummy, params, _PerfectForward(), relu_forward)
    if abs(acc_perfect - 1.0) > 1e-5:
        print(f"❌ FAIL: perfect model accuracy = {acc_perfect:.4f}, expected 1.0")
        all_ok = False

    if all_ok:
        print("✅ PASS: compute_accuracy is correct.")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 13. Create batches
# ────────────────────────────────────────────────────────────────

def test_create_batches(create_batches_fn):
    """Tests create_batches for shuffling, sizes, and consistency."""

    all_ok = True
    torch.manual_seed(9)

    N = 100
    X = torch.arange(N).float().unsqueeze(1).expand(N, 5)  # each row is [i, i, i, i, i]
    y = torch.arange(N)
    batch_size = 32

    batches = create_batches_fn(X, y, batch_size)

    # Total samples
    total = sum(xb.shape[0] for xb, yb in batches)
    if total != N:
        print(f"❌ FAIL: total samples in batches = {total}, expected {N}")
        all_ok = False

    # Batch sizes
    for i, (xb, yb) in enumerate(batches):
        if i < len(batches) - 1 and xb.shape[0] != batch_size:
            print(f"❌ FAIL: batch {i} has {xb.shape[0]} samples, expected {batch_size}")
            all_ok = False
        if xb.shape[0] != yb.shape[0]:
            print(f"❌ FAIL: batch {i} X and y have different sizes")
            all_ok = False

    # X and y should be consistently shuffled (same permutation)
    all_x_first_col = torch.cat([xb[:, 0] for xb, yb in batches])
    all_y = torch.cat([yb for xb, yb in batches])
    if not torch.allclose(all_x_first_col, all_y.float()):
        print("❌ FAIL: X and y are not consistently shuffled together")
        print("   💡 Make sure you apply the same permutation to both X and y")
        all_ok = False

    # Should be shuffled (not in original order)
    if torch.equal(all_y, torch.arange(N)):
        print("❌ FAIL: data is not shuffled (still in original order)")
        all_ok = False

    if all_ok:
        print("✅ PASS: create_batches is correct.")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 14. SGD step
# ────────────────────────────────────────────────────────────────

def test_sgd_step(sgd_step_fn):
    """Tests sgd_step for correct parameter updates."""

    all_ok = True

    params = {
        'W1': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        'b1': torch.tensor([0.5, 0.5]),
    }
    grads = {
        'dW1': torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        'db1': torch.tensor([0.05, 0.05]),
    }
    lr = 0.1

    W1_expected = torch.tensor([[1.0 - 0.1*0.1, 2.0 - 0.1*0.2],
                                 [3.0 - 0.1*0.3, 4.0 - 0.1*0.4]])
    b1_expected = torch.tensor([0.5 - 0.1*0.05, 0.5 - 0.1*0.05])

    sgd_step_fn(params, grads, lr)

    if not torch.allclose(params['W1'], W1_expected, atol=1e-6):
        print(f"❌ FAIL: sgd_step W1 = {params['W1'].tolist()}, expected {W1_expected.tolist()}")
        all_ok = False

    if not torch.allclose(params['b1'], b1_expected, atol=1e-6):
        print(f"❌ FAIL: sgd_step b1 = {params['b1'].tolist()}, expected {b1_expected.tolist()}")
        all_ok = False

    if all_ok:
        print("✅ PASS: sgd_step is correct.")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 15. Train
# ────────────────────────────────────────────────────────────────

def test_train(train_fn, forward_fn, backward_fn, softmax_fn, loss_fn, init_fn, load_mnist_fn):
    """Tests train by running 2 epochs on a small subset and checking learning signal."""

    all_ok = True

    X_train, y_train, X_val, y_val, _, _ = load_mnist_fn()

    # Use a small subset for speed
    X_train_small = X_train[:1000]
    y_train_small = y_train[:1000]
    X_val_small = X_val[:500]
    y_val_small = y_val[:500]

    from mlp_layers import relu_forward, relu_backward
    params = init_fn()

    history = train_fn(
        X_train_small, y_train_small, X_val_small, y_val_small, params,
        forward_fn, backward_fn, softmax_fn, loss_fn,
        relu_forward, relu_backward,
        epochs=3, batch_size=64, learning_rate=0.01
    )

    # Check structure
    expected_keys = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
    for key in expected_keys:
        if key not in history:
            print(f"❌ FAIL: history missing key '{key}'")
            all_ok = False
        elif len(history[key]) != 3:
            print(f"❌ FAIL: history['{key}'] has {len(history[key])} entries, expected 3")
            all_ok = False

    # Loss should decrease
    if all_ok and history['train_loss'][2] >= history['train_loss'][0]:
        print(f"⚠️  WARNING: train_loss did not decrease ({history['train_loss'][0]:.4f} -> {history['train_loss'][2]:.4f})")
        print("   This may indicate a bug in the training loop.")
        all_ok = False
        
    # Accuracy should be above random guessing (10%)
    if all_ok and history['val_acc'][2] < 0.1:
        print(f"⚠️  WARNING: val_acc is below random guessing (10%), got {history['val_acc'][2]:.4f}")
        print("   This may indicate a bug.")
        all_ok = False

    # Accuracy should be above 25% after 3 epochs on this small subset
    if all_ok and history['val_acc'][2] < 0.25:
        print(f"⚠️  WARNING: val_acc is low after 3 epochs, got {history['val_acc'][2]:.4f}")
        print("   We get ~32% with a correct implementation, so below 25% suggests a significant issue.")
        print("   This may indicate a bug.")
        all_ok = False

    if all_ok:
        print("✅ PASS: train returns correct history structure and shows learning signal.")
    return all_ok
