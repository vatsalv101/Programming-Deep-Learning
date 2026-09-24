import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ────────────────────────────────────────────────────────────────
# 1. One-hot encoding
# ────────────────────────────────────────────────────────────────

def test_one_hot_encode(one_hot_fn):
    """Tests one_hot_encode for shape, dtype, correct placement, and sum."""

    all_ok = True
    vocab_size = 10

    for ix in [0, 5, 9]:
        v = one_hot_fn(ix, vocab_size)

        if v.shape != torch.Size([vocab_size, 1]):
            print(f"❌ FAIL: one_hot_encode({ix}, {vocab_size}).shape = {tuple(v.shape)}, expected ({vocab_size}, 1)")
            all_ok = False
            continue

        if v[ix, 0].item() != 1.0:
            print(f"❌ FAIL: one_hot_encode({ix}, {vocab_size})[{ix}, 0] = {v[ix, 0].item()}, expected 1.0")
            all_ok = False

        if v.sum().item() != 1.0:
            print(f"❌ FAIL: one_hot_encode({ix}, {vocab_size}).sum() = {v.sum().item()}, expected 1.0")
            all_ok = False

    # Edge: large vocab
    v = one_hot_fn(42, 100)
    if v.shape != torch.Size([100, 1]) or v[42, 0].item() != 1.0 or v.sum().item() != 1.0:
        print("❌ FAIL: one_hot_encode(42, 100) incorrect")
        all_ok = False

    if all_ok:
        print("✅ PASS: one_hot_encode is correct.")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 2. Sigmoid
# ────────────────────────────────────────────────────────────────

def test_sigmoid(sigmoid_fn):
    """Tests sigmoid against known values and edge cases."""

    cases = [
        (torch.tensor([0.0]),  torch.tensor([0.5])),
        (torch.tensor([10.0]), torch.tensor([1.0 / (1 + math.exp(-10))])),
        (torch.tensor([-10.0]), torch.tensor([1.0 / (1 + math.exp(10))])),
    ]
    all_ok = True
    for x, expected in cases:
        y = sigmoid_fn(x)
        if not torch.allclose(y, expected, atol=1e-6):
            print(f"❌ FAIL: sigmoid({x.item():.1f}) = {y.item():.6f}, expected {expected.item():.6f}")
            all_ok = False

    # Vector input
    x_vec = torch.linspace(-5, 5, 50)
    y_vec = sigmoid_fn(x_vec)
    ref = torch.sigmoid(x_vec)
    if not torch.allclose(y_vec, ref, atol=1e-6):
        print("❌ FAIL: sigmoid disagrees with torch.sigmoid on vector input.")
        all_ok = False

    if all_ok:
        print("✅ PASS: sigmoid is correct.")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 3. Sigmoid derivative
# ────────────────────────────────────────────────────────────────

def test_sigmoid_deriv(sigmoid_fn, sigmoid_deriv_fn):
    """Tests sigmoid_deriv(s) where s = sigmoid(x), via numerical gradient."""

    # Use float64 for accurate numerical gradient
    x = torch.linspace(-5, 5, 100, dtype=torch.float64)
    s = sigmoid_fn(x)
    ds_analytical = sigmoid_deriv_fn(s)

    # Numerical derivative
    eps = 1e-5
    ds_numerical = (sigmoid_fn(x + eps) - sigmoid_fn(x - eps)) / (2 * eps)

    if torch.allclose(ds_analytical, ds_numerical, atol=1e-4):
        print("✅ PASS: sigmoid_deriv is correct (matches numerical gradient).")
        return True
    else:
        max_err = (ds_analytical - ds_numerical).abs().max().item()
        print(f"❌ FAIL: sigmoid_deriv max error = {max_err:.2e}.")
        print("   💡 Remember: sigmoid_deriv takes the sigmoid *output* s, not the raw input x.")
        print("   The formula is: s * (1 - s)")
        return False


# ────────────────────────────────────────────────────────────────
# 4. LSTM cell forward
# ────────────────────────────────────────────────────────────────

def _reorder_gates_to_pytorch(W, H):
    """Reorders gate rows from our (f,i,g,o) to PyTorch's (i,f,g,o)."""
    W_pt = W.clone()
    W_pt[0:H]   = W[H:2*H]    # PyTorch slot 0 = input gate  = our slot 1
    W_pt[H:2*H] = W[0:H]      # PyTorch slot 1 = forget gate = our slot 0
    return W_pt


def test_lstm_cell_forward(cell_forward_fn, H=8):
    """Tests a single LSTM cell forward step against PyTorch's LSTMCell."""

    vocab_size = 10
    torch.manual_seed(42)

    # Random parameters
    Wx = torch.randn(4*H, vocab_size) * 2.00
    Wh = torch.randn(4*H, H) * 2.00
    b  = torch.randn(4*H, 1) * 2.00

    # Random input and states
    x_t    = torch.zeros(vocab_size, 1); x_t[3, 0] = 1.0
    h_prev = torch.randn(H, 1) * 0.1
    c_prev = torch.randn(H, 1) * 0.1

    # Student
    h_stu, c_stu, cache = cell_forward_fn(x_t, h_prev, c_prev, Wx, Wh, b)

    # Reference: PyTorch LSTMCell (gate order: i,f,g,o — swap f and i)
    cell = nn.LSTMCell(input_size=vocab_size, hidden_size=H)
    with torch.no_grad():
        cell.weight_ih.copy_(_reorder_gates_to_pytorch(Wx, H))
        cell.weight_hh.copy_(_reorder_gates_to_pytorch(Wh, H))
        cell.bias_ih.copy_(_reorder_gates_to_pytorch(b, H).view(-1))
        cell.bias_hh.zero_()
    cell.eval()
    h_ref, c_ref = cell(x_t.t(), (h_prev.t(), c_prev.t()))

    all_ok = True

    # Check shapes
    for name, tensor, shape in [('h', h_stu, (H, 1)), ('c', c_stu, (H, 1))]:
        if tensor.shape != torch.Size(shape):
            print(f"❌ FAIL: {name}.shape = {tuple(tensor.shape)}, expected {shape}")
            all_ok = False
        else:
            print(f"✅ PASS: {name} has correct shape {shape}")

    # Check values
    tol = 1e-5
    if torch.allclose(h_stu, h_ref.t(), atol=tol):
        print(f"✅ PASS: h matches reference (max diff {(h_stu - h_ref.t()).abs().max():.2e})")
    else:
        print(f"❌ FAIL: h differs from reference (max diff {(h_stu - h_ref.t()).abs().max():.2e})")
        all_ok = False

    if torch.allclose(c_stu, c_ref.t(), atol=tol):
        print(f"✅ PASS: c matches reference (max diff {(c_stu - c_ref.t()).abs().max():.2e})")
    else:
        print(f"❌ FAIL: c differs from reference (max diff {(c_stu - c_ref.t()).abs().max():.2e})")
        all_ok = False

    # Check cache
    required_keys = {'x_t', 'h_prev', 'c_prev', 'f', 'i', 'g', 'o', 'c', 'preactivations', 'Wh'}
    if not isinstance(cache, dict):
        print(f"❌ FAIL: cache should be a dict, got {type(cache)}")
        all_ok = False
    else:
        missing = required_keys - set(cache.keys())
        if missing:
            print(f"❌ FAIL: cache is missing keys: {missing}")
            all_ok = False
        else:
            print(f"✅ PASS: cache contains all required keys")

    if all_ok:
        print("\n🎉 lstm_cell_forward is correct!")
    else:
        print("\n🚨 Some checks failed — review the messages above.")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 5. LSTM cell backward
# ────────────────────────────────────────────────────────────────

def test_lstm_cell_backward(cell_forward_fn, cell_backward_fn, H=8):
    """Tests a single LSTM cell backward step against PyTorch autograd."""

    vocab_size = 10
    torch.manual_seed(42)

    Wx = torch.randn(4*H, vocab_size) * 0.7
    Wh = torch.randn(4*H, H) * 0.7
    b  = torch.randn(4*H, 1) * 0.7
    x_t = torch.zeros(vocab_size, 1); x_t[3, 0] = 1.0
    h_prev = torch.randn(H, 1) * 0.1
    c_prev = torch.randn(H, 1) * 0.1

    # Student forward + backward
    h_stu, c_stu, cache = cell_forward_fn(x_t, h_prev, c_prev, Wx, Wh, b)
    dh_next = torch.randn(H, 1) * 0.3
    dc_next = torch.randn(H, 1) * 0.3
    dWx_stu, dWh_stu, db_stu, dh_prev_stu, dc_prev_stu = cell_backward_fn(dh_next, dc_next, cache)

    # Reference via autograd — swap f↔i gates for PyTorch ordering
    Wx_pt = _reorder_gates_to_pytorch(Wx, H)
    Wh_pt = _reorder_gates_to_pytorch(Wh, H)
    b_pt  = _reorder_gates_to_pytorch(b, H)

    Wx_ref = Wx_pt.clone().requires_grad_(True)
    Wh_ref = Wh_pt.clone().requires_grad_(True)
    b_ref  = b_pt.clone().requires_grad_(True)
    h_prev_ref = h_prev.clone().requires_grad_(True)
    c_prev_ref = c_prev.clone().requires_grad_(True)

    cell_ref = nn.LSTMCell(input_size=vocab_size, hidden_size=H)
    with torch.no_grad():
        cell_ref.weight_ih.copy_(Wx_ref)
        cell_ref.weight_hh.copy_(Wh_ref)
        cell_ref.bias_ih.copy_(b_ref.view(-1))
        cell_ref.bias_hh.zero_()
    cell_ref.bias_hh.requires_grad_(False)

    cell_ref.weight_ih = nn.Parameter(Wx_ref)
    cell_ref.weight_hh = nn.Parameter(Wh_ref)
    cell_ref.bias_ih = nn.Parameter(b_ref.view(-1))

    h_r, c_r = cell_ref(x_t.t(), (h_prev_ref.t(), c_prev_ref.t()))

    # Fake loss: dh_next.T @ h + dc_next.T @ c (matches our upstream gradients)
    fake_loss = (dh_next.t() @ h_r.t()).sum() + (dc_next.t() @ c_r.t()).sum()
    fake_loss.backward()

    # Reorder reference gradients back from PyTorch (i,f,g,o) to our (f,i,g,o)
    dWx_ref_reordered = _reorder_gates_to_pytorch(cell_ref.weight_ih.grad, H)
    dWh_ref_reordered = _reorder_gates_to_pytorch(cell_ref.weight_hh.grad, H)
    db_ref_reordered  = _reorder_gates_to_pytorch(cell_ref.bias_ih.grad.view(4*H, 1), H)

    all_ok = True
    tol = 1e-4

    checks = [
        ('dWx', dWx_stu, dWx_ref_reordered),
        ('dWh', dWh_stu, dWh_ref_reordered),
        ('db',  db_stu,  db_ref_reordered),
        ('dh_prev', dh_prev_stu, h_prev_ref.grad),
        ('dc_prev', dc_prev_stu, c_prev_ref.grad),
    ]

    for name, stu, ref in checks:
        if stu.shape != ref.shape:
            print(f"❌ FAIL: {name}.shape = {tuple(stu.shape)}, expected {tuple(ref.shape)}")
            all_ok = False
            continue
        max_err = (stu - ref).abs().max().item()
        if max_err <= tol:
            print(f"✅ PASS: {name} matches reference (max error {max_err:.2e})")
        else:
            print(f"❌ FAIL: {name} max error = {max_err:.2e} > tol={tol}")
            all_ok = False

    if all_ok:
        print("\n🎉 lstm_cell_backward is correct!")
    else:
        print("\n🚨 Some checks failed — review your gate gradient formulas.")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 6. Model initialization
# ────────────────────────────────────────────────────────────────

def test_init_model(init_fn, vocab_size, H, tol=1e-8):
    """Tests init_model for correct shapes, zero biases, and reasonable magnitudes."""

    torch.manual_seed(0)
    out = init_fn(vocab_size, H)

    if not (isinstance(out, tuple) and len(out) == 5):
        print(f"❌ FAIL: expected a tuple of 5 tensors, got {type(out)} of length "
              f"{len(out) if isinstance(out, tuple) else 'N/A'}.")
        print("   💡 Return exactly (Wx, Wh, b, Wy, by).")
        return False
    Wx, Wh, b, Wy, by = out

    all_ok = True

    # Shape checks
    shapes = [
        ('Wx', Wx, (4*H, vocab_size)),
        ('Wh', Wh, (4*H, H)),
        ('b',  b,  (4*H, 1)),
        ('Wy', Wy, (vocab_size, H)),
        ('by', by, (vocab_size, 1)),
    ]
    for name, tensor, expected in shapes:
        if not isinstance(tensor, torch.Tensor):
            print(f"❌ FAIL: {name} is not a torch.Tensor (got {type(tensor)}).")
            all_ok = False
        elif tensor.shape != torch.Size(expected):
            print(f"❌ FAIL: {name}.shape = {tuple(tensor.shape)}, expected {expected}")
            all_ok = False
        else:
            print(f"✅ PASS: {name} shape {expected}")

    # Bias zero checks
    for name, tensor in [('b', b), ('by', by)]:
        if torch.allclose(tensor, torch.zeros_like(tensor), atol=tol):
            print(f"✅ PASS: {name} is all zeros")
        else:
            print(f"❌ FAIL: {name} is not zero")
            all_ok = False

    # Magnitude checks
    for name, tensor in [('Wx', Wx), ('Wh', Wh), ('Wy', Wy)]:
        maxval = tensor.abs().max().item()
        if maxval < 0.1:
            print(f"✅ PASS: {name} max|val| = {maxval:.2e} (small)")
        else:
            print(f"❌ FAIL: {name} max|val| = {maxval:.2e} — too large")
            all_ok = False

    # Std check
    for name, tensor in [('Wx', Wx), ('Wh', Wh), ('Wy', Wy)]:
        std = tensor.std().item()
        if abs(std - 0.01) <= 0.005:
            print(f"✅ PASS: {name}.std() = {std:.2e} ≈ 0.01")
        else:
            print(f"❌ FAIL: {name}.std() = {std:.2e}, expected ~0.01 (init with torch.randn * 0.01)")
            all_ok = False

    if all_ok:
        print("\n🎉 init_model looks good!")
    else:
        print("\n🚨 Some checks failed — review the messages above.")
    return all_ok


# ────────────────────────────────────────────────────────────────
# 7. Sequence-level forward pass
# ────────────────────────────────────────────────────────────────

def test_forward_pass(forward_fn, init_fn, vocab_size, H, seq=None, tol=1e-5):
    """Tests forward_pass loss against a PyTorch LSTMCell reference."""

    print("\n🔍 Comparing your forward_pass loss against PyTorch reference ...")

    # Use large-scale weights and non-zero states so a broken implementation
    # (e.g. always h=0) produces a measurably different loss from a correct one.
    torch.manual_seed(0)
    Wx = torch.randn(4*H, vocab_size) * 0.5
    Wh = torch.randn(4*H, H) * 0.5
    b  = torch.randn(4*H, 1) * 0.1
    Wy = torch.randn(vocab_size, H) * 0.5
    by = torch.randn(vocab_size, 1) * 0.1
    hprev = torch.randn(H, 1) * 0.5
    cprev = torch.randn(H, 1) * 0.5

    if seq is None:
        seq = list(range(vocab_size))
    T = len(seq)

    # Student
    try:
        loss_stu, cache_stu = forward_fn(seq, Wx, Wh, b, Wy, by, hprev, cprev, vocab_size)
    except Exception as e:
        print(f"❌ FAIL: forward_pass raised:\n   {e}")
        return False

    if isinstance(loss_stu, torch.Tensor):
        loss_stu = loss_stu.item()
    loss_stu = float(loss_stu)

    # Reference
    cell = nn.LSTMCell(input_size=vocab_size, hidden_size=H)
    with torch.no_grad():
        cell.weight_ih.copy_(_reorder_gates_to_pytorch(Wx, H))
        cell.weight_hh.copy_(_reorder_gates_to_pytorch(Wh, H))
        cell.bias_ih.copy_(_reorder_gates_to_pytorch(b, H).view(-1))
        cell.bias_hh.zero_()
    cell.eval()

    h_ref = hprev.t()
    c_ref = cprev.t()
    loss_ref = 0.0

    for t in range(T):
        x = torch.zeros(1, vocab_size)
        x[0, seq[t]] = 1.0
        h_ref, c_ref = cell(x, (h_ref, c_ref))
        y = Wy @ h_ref.t() + by
        p = torch.softmax(y, dim=0)
        tgt = seq[(t + 1) % T]
        loss_ref += -math.log(p[tgt, 0].item())

    if abs(loss_stu - loss_ref) < tol:
        print(f"✅ PASS: loss = {loss_stu:.6f}, reference = {loss_ref:.6f} (within tol={tol})")
    else:
        print(f"❌ FAIL: loss = {loss_stu:.6f}, reference = {loss_ref:.6f}")
        print(f"   💡 Difference of {abs(loss_stu - loss_ref):.2e} > tol")
        return False

    # Cache check
    if not isinstance(cache_stu, dict):
        print(f"⚠️  WARNING: cache should be a dict, got {type(cache_stu)}")
    else:
        cells = cache_stu.get('cell_caches', {})
        if len(cells) >= T:
            print(f"✅ PASS: cache contains {len(cells)} cell caches for {T} timesteps")
        else:
            print(f"⚠️  WARNING: expected {T} cell caches, found {len(cells)}")

    print("\n🎉 forward_pass looks correct!")
    return True


# ────────────────────────────────────────────────────────────────
# 8. Sequence-level backward pass (BPTT)
# ────────────────────────────────────────────────────────────────

def test_backward_pass(backward_fn, forward_fn, init_fn, vocab_size, H, seq=None, tol=1e-4):
    """Tests BPTT gradients against PyTorch autograd reference."""

    # Use larger weights and non-zero initial states so that ALL gradient
    # components (especially dWh, dby) have meaningful magnitudes and cannot
    # be masked by the tolerance.
    torch.manual_seed(0)
    Wx = torch.randn(4*H, vocab_size) * 0.5
    Wh = torch.randn(4*H, H) * 0.5
    b  = torch.randn(4*H, 1) * 0.1
    Wy = torch.randn(vocab_size, H) * 0.5
    by = torch.randn(vocab_size, 1) * 0.1
    hprev = torch.randn(H, 1) * 0.5
    cprev = torch.randn(H, 1) * 0.5

    if seq is None:
        seq = list(range(vocab_size))
    T = len(seq)
    targets = [seq[(t + 1) % T] for t in range(T)]

    # Student forward + backward
    try:
        loss_stu, cache = forward_fn(seq, Wx, Wh, b, Wy, by, hprev, cprev, vocab_size)
    except Exception as e:
        print(f"❌ FAIL: forward_fn raised:\n   {e}")
        return False

    try:
        out = backward_fn(targets, cache, Wy, by)
    except Exception as e:
        print(f"❌ FAIL: backward_fn raised:\n   {e}")
        return False

    if not (isinstance(out, (tuple, list)) and len(out) == 7):
        print(f"❌ FAIL: expected 7 return values (dWx, dWh, dWy, db, dby, h_last, c_last), "
              f"got {len(out) if isinstance(out, (tuple, list)) else type(out)}")
        return False

    dWx_stu, dWh_stu, dWy_stu, db_stu, dby_stu, h_last, c_last = out

    # Shape checks
    all_ok = True
    shape_checks = [
        ('dWx',    dWx_stu,  (4*H, vocab_size)),
        ('dWh',    dWh_stu,  (4*H, H)),
        ('dWy',    dWy_stu,  (vocab_size, H)),
        ('db',     db_stu,   (4*H, 1)),
        ('dby',    dby_stu,  (vocab_size, 1)),
        ('h_last', h_last,   (H, 1)),
        ('c_last', c_last,   (H, 1)),
    ]
    for name, tensor, expected in shape_checks:
        if not isinstance(tensor, torch.Tensor):
            print(f"❌ FAIL: {name} is not a Tensor (got {type(tensor)})")
            return False
        if tensor.shape != torch.Size(expected):
            print(f"❌ FAIL: {name}.shape = {tuple(tensor.shape)}, expected {expected}")
            return False
        print(f"✅ PASS: {name} shape {expected}")

    # PyTorch reference
    cell = nn.LSTMCell(input_size=vocab_size, hidden_size=H, bias=True)
    linear = nn.Linear(H, vocab_size, bias=True)
    with torch.no_grad():
        cell.weight_ih.copy_(_reorder_gates_to_pytorch(Wx, H))
        cell.weight_hh.copy_(_reorder_gates_to_pytorch(Wh, H))
        cell.bias_ih.copy_(_reorder_gates_to_pytorch(b, H).view(-1))
        cell.bias_hh.zero_()
        linear.weight.copy_(Wy)
        linear.bias.copy_(by.view(-1))

    cell.bias_hh.requires_grad_(False)
    cell.zero_grad()
    linear.zero_grad()
    for p in [cell.weight_ih, cell.weight_hh, cell.bias_ih, linear.weight, linear.bias]:
        p.requires_grad_(True)

    h = hprev.t()
    c = cprev.t()
    loss_ref = 0.0
    for t in range(T):
        x = torch.zeros(1, vocab_size)
        x[0, seq[t]] = 1.0
        h, c = cell(x, (h, c))
        y = linear(h)
        logp = F.log_softmax(y, dim=1)
        loss_ref = loss_ref - logp[0, targets[t]]
    loss_ref.backward()

    h_last_ref = h.detach().t()   # (H, 1) — final hidden state
    c_last_ref = c.detach().t()   # (H, 1) — final cell state

    dWx_ref = _reorder_gates_to_pytorch(cell.weight_ih.grad.clone(), H)
    dWh_ref = _reorder_gates_to_pytorch(cell.weight_hh.grad.clone(), H)
    db_ref  = _reorder_gates_to_pytorch(cell.bias_ih.grad.view(4*H, 1).clone(), H)
    dWy_ref = linear.weight.grad.clone()
    dby_ref = linear.bias.grad.view(vocab_size, 1).clone()

    grad_checks = [
        ('dWx', dWx_stu, dWx_ref),
        ('dWh', dWh_stu, dWh_ref),
        ('dWy', dWy_stu, dWy_ref),
        ('db',  db_stu,  db_ref),
        ('dby', dby_stu, dby_ref),
    ]
    for name, stu, ref in grad_checks:
        max_err = (stu - ref).abs().max().item()
        if max_err <= tol:
            print(f"✅ PASS: {name} gradients match (max error {max_err:.2e})")
        else:
            print(f"❌ FAIL: {name} max error = {max_err:.2e} > tol={tol}")
            all_ok = False

    # Check returned final states (not just shapes)
    for name, stu, ref in [('h_last', h_last, h_last_ref), ('c_last', c_last, c_last_ref)]:
        max_err = (stu - ref).abs().max().item()
        if max_err <= tol:
            print(f"✅ PASS: {name} matches reference (max error {max_err:.2e})")
            #print(stu)
            #print(ref)
        else:
            print(f"❌ FAIL: {name} max error = {max_err:.2e} > tol={tol}")
            all_ok = False

    if all_ok:
        print("\n🎉 All gradient checks passed! Your backward_pass matches PyTorch autograd.")
    else:
        print("\n🚨 Some gradient checks failed — review your BPTT logic.")
    return all_ok
