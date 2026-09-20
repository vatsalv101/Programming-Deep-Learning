import torch
import optimizers


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _make_params():
    params = [
        torch.tensor([[1.0, -2.0], [3.0, -4.0]], requires_grad=True),
        torch.tensor([0.5, -0.5], requires_grad=True)
    ]

    params[0].grad = torch.tensor([[0.10, -0.20], [0.30, -0.40]])
    params[1].grad = torch.tensor([0.05, -0.05])

    return params


def _make_second_grads(params):
    params[0].grad = torch.tensor([[-0.15, 0.05], [0.20, -0.10]])
    params[1].grad = torch.tensor([-0.03, 0.07])


def _clone_params(params):
    cloned = []

    for param in params:
        new_param = param.detach().clone().requires_grad_(True)
        if param.grad is not None:
            new_param.grad = param.grad.detach().clone()
        cloned.append(new_param)

    return cloned


def _first_bad_entry(actual, expected):
    error = (actual - expected).abs()
    flat_index = torch.argmax(error).item()
    index = torch.unravel_index(torch.tensor(flat_index), actual.shape)

    return (
        tuple(i.item() for i in index),
        actual[index].item(),
        expected[index].item(),
        error[index].item()
    )


def _check_tensor_list(name, actual_list, expected_list, atol=1e-6):
    ok = True

    if len(actual_list) != len(expected_list):
        print(
            f"❌ FAIL: {name} has length {len(actual_list)}, "
            f"expected {len(expected_list)}."
        )
        return False

    for i, (actual, expected) in enumerate(zip(actual_list, expected_list)):

        if actual.shape != expected.shape:
            print(
                f"❌ FAIL: {name}[{i}] has shape {tuple(actual.shape)}, "
                f"expected {tuple(expected.shape)}."
            )
            ok = False
            continue

        if not torch.allclose(actual, expected, atol=atol, rtol=1e-5):
            idx, got, exp, err = _first_bad_entry(actual, expected)
            print(
                f"❌ FAIL: {name}[{i}] is incorrect. "
                f"Max error {err:.3e} at index {idx}: "
                f"got {got:.6f}, expected {exp:.6f}."
            )
            ok = False

    return ok


def _check_scalar(name, actual, expected):
    if actual == expected:
        return True

    print(f"❌ FAIL: {name} is incorrect. Got {actual}, expected {expected}.")
    return False


# ────────────────────────────────────────────────────────────────
# SGD and zero_grad
# ────────────────────────────────────────────────────────────────

def test_sgd():
    """Tests zero_grad and SGD against torch.optim.SGD."""
    all_ok = True

    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([-1.0], requires_grad=True)

    p1.grad = torch.tensor([0.25, -0.25])
    p2.grad = None

    optimizers.zero_grad([p1, p2])

    zero_grad_ok = (
        p1.grad is not None
        and torch.allclose(p1.grad, torch.zeros_like(p1.grad))
        and p2.grad is None
    )

    if zero_grad_ok:
        print("✅ PASS: zero_grad is correct.")
    else:
        print("❌ FAIL: zero_grad is incorrect.")
        if p1.grad is None:
            print("   p1.grad became None, but it should be zero.")
        elif not torch.allclose(p1.grad, torch.zeros_like(p1.grad)):
            print(f"   p1.grad is {p1.grad}, expected zeros.")
        if p2.grad is not None:
            print("   p2.grad was None, but zero_grad created a gradient.")
        all_ok = False

    params = _make_params()
    reference_params = _clone_params(params)

    learning_rate = 0.1

    reference_optimizer = torch.optim.SGD(
        reference_params,
        lr=learning_rate
    )

    reference_optimizer.step()
    optimizers.sgd_step(
        params,
        None,
        learning_rate=learning_rate
    )

    expected_params = [p.detach() for p in reference_params]
    actual_params = [p.detach() for p in params]

    params_ok = _check_tensor_list(
        "parameters after sgd_step",
        actual_params,
        expected_params
    )

    if params_ok:
        print("✅ PASS: sgd_step is correct.")
    else:
        all_ok = False

    print()
    if all_ok:
        print("🎉 SGD tests passed.")
    else:
        print("🚨 Some SGD tests failed.")

    return all_ok


# ────────────────────────────────────────────────────────────────
# SGD with Momentum
# ────────────────────────────────────────────────────────────────

def test_sgd_momentum():
    """Tests momentum initialization and SGD momentum against torch.optim.SGD."""
    all_ok = True

    params = _make_params()
    velocity = optimizers.init_momentum(params)

    init_ok = (
        len(velocity) == len(params)
        and all(v.shape == p.shape for v, p in zip(velocity, params))
        and all(torch.allclose(v, torch.zeros_like(p)) for v, p in zip(velocity, params))
    )

    if init_ok:
        print("✅ PASS: init_momentum is correct.")
    else:
        print("❌ FAIL: init_momentum is incorrect.")
        if len(velocity) != len(params):
            print(f"   velocity has length {len(velocity)}, expected {len(params)}.")
        for i, (v, p) in enumerate(zip(velocity, params)):
            if v.shape != p.shape:
                print(
                    f"   velocity[{i}] has shape {tuple(v.shape)}, "
                    f"expected {tuple(p.shape)}."
                )
            elif not torch.allclose(v, torch.zeros_like(p)):
                print(f"   velocity[{i}] is not initialized to zeros.")
        all_ok = False

    params = _make_params()
    reference_params = _clone_params(params)

    learning_rate = 0.1
    momentum = 0.9

    velocity = optimizers.init_momentum(params)

    reference_optimizer = torch.optim.SGD(
        reference_params,
        lr=learning_rate,
        momentum=momentum
    )

    reference_optimizer.step()
    velocity = optimizers.sgd_momentum_step(
        params,
        velocity,
        learning_rate=learning_rate,
        momentum=momentum
    )

    expected_params = [p.detach() for p in reference_params]
    actual_params = [p.detach() for p in params]

    expected_velocity = [
        -learning_rate * reference_optimizer.state[p]["momentum_buffer"].detach()
        for p in reference_params
    ]

    params_ok = _check_tensor_list(
        "parameters after sgd_momentum_step",
        actual_params,
        expected_params
    )

    velocity_ok = _check_tensor_list(
        "velocity",
        velocity,
        expected_velocity
    )

    if params_ok and velocity_ok:
        print("✅ PASS: sgd_momentum_step is correct.")
    else:
        all_ok = False

    print()
    if all_ok:
        print("🎉 SGD with momentum tests passed.")
    else:
        print("🚨 Some SGD with momentum tests failed.")

    return all_ok


# ────────────────────────────────────────────────────────────────
# AdaGrad
# ────────────────────────────────────────────────────────────────

def test_adagrad():
    """Tests AdaGrad initialization and updates against torch.optim.Adagrad."""

    all_ok = True

    params = _make_params()
    G = optimizers.init_adagrad(params)

    init_ok = (
        len(G) == len(params)
        and all(g.shape == p.shape for g, p in zip(G, params))
        and all(torch.allclose(g, torch.zeros_like(p)) for g, p in zip(G, params))
    )

    if init_ok:
        print("✅ PASS: init_adagrad is correct.")
    else:
        print("❌ FAIL: init_adagrad is incorrect.")
        if len(G) != len(params):
            print(f"   G has length {len(G)}, expected {len(params)}.")
        for i, (g, p) in enumerate(zip(G, params)):
            if g.shape != p.shape:
                print(
                    f"   G[{i}] has shape {tuple(g.shape)}, "
                    f"expected {tuple(p.shape)}."
                )
            elif not torch.allclose(g, torch.zeros_like(p)):
                print(f"   G[{i}] is not initialized to zeros.")
        all_ok = False

    params = _make_params()
    reference_params = _clone_params(params)

    learning_rate = 0.1
    eps = 1e-8

    G = optimizers.init_adagrad(params)

    reference_optimizer = torch.optim.Adagrad(
        reference_params,
        lr=learning_rate,
        eps=eps
    )

    reference_optimizer.step()
    G = optimizers.adagrad_step(
        params,
        G,
        learning_rate=learning_rate,
        eps=eps
    )

    _make_second_grads(params)
    _make_second_grads(reference_params)

    reference_optimizer.step()
    G = optimizers.adagrad_step(
        params,
        G,
        learning_rate=learning_rate,
        eps=eps
    )

    expected_params = [p.detach() for p in reference_params]
    actual_params = [p.detach() for p in params]

    expected_G = [
        reference_optimizer.state[p]["sum"].detach()
        for p in reference_params
    ]

    params_ok = _check_tensor_list(
        "parameters after adagrad_step",
        actual_params,
        expected_params
    )

    G_ok = _check_tensor_list("G", G, expected_G)

    if params_ok and G_ok:
        print("✅ PASS: adagrad_step is correct.")
    else:
        all_ok = False

    print()
    if all_ok:
        print("🎉 AdaGrad tests passed.")
    else:
        print("🚨 Some AdaGrad tests failed.")

    return all_ok


# ────────────────────────────────────────────────────────────────
# RMSProp
# ────────────────────────────────────────────────────────────────

def test_rmsprop():
    """Tests RMSProp initialization and updates against torch.optim.RMSprop."""

    all_ok = True

    params = _make_params()
    G = optimizers.init_rmsprop(params)

    init_ok = (
        len(G) == len(params)
        and all(g.shape == p.shape for g, p in zip(G, params))
        and all(torch.allclose(g, torch.zeros_like(p)) for g, p in zip(G, params))
    )

    if init_ok:
        print("✅ PASS: init_rmsprop is correct.")
    else:
        print("❌ FAIL: init_rmsprop is incorrect.")
        if len(G) != len(params):
            print(f"   G has length {len(G)}, expected {len(params)}.")
        for i, (g, p) in enumerate(zip(G, params)):
            if g.shape != p.shape:
                print(
                    f"   G[{i}] has shape {tuple(g.shape)}, "
                    f"expected {tuple(p.shape)}."
                )
            elif not torch.allclose(g, torch.zeros_like(p)):
                print(f"   G[{i}] is not initialized to zeros.")
        all_ok = False

    params = _make_params()
    reference_params = _clone_params(params)

    learning_rate = 0.1
    gamma = 0.9
    eps = 1e-8

    G = optimizers.init_rmsprop(params)

    reference_optimizer = torch.optim.RMSprop(
        reference_params,
        lr=learning_rate,
        alpha=gamma,
        eps=eps,
        momentum=0.0,
        centered=False,
        weight_decay=0.0
    )

    reference_optimizer.step()
    G = optimizers.rmsprop_step(
        params,
        G,
        learning_rate=learning_rate,
        gamma=gamma,
        eps=eps
    )

    _make_second_grads(params)
    _make_second_grads(reference_params)

    reference_optimizer.step()
    G = optimizers.rmsprop_step(
        params,
        G,
        learning_rate=learning_rate,
        gamma=gamma,
        eps=eps
    )

    expected_params = [p.detach() for p in reference_params]
    actual_params = [p.detach() for p in params]

    expected_G = [
        reference_optimizer.state[p]["square_avg"].detach()
        for p in reference_params
    ]

    params_ok = _check_tensor_list(
        "parameters after rmsprop_step",
        actual_params,
        expected_params
    )

    G_ok = _check_tensor_list("G", G, expected_G)

    if params_ok and G_ok:
        print("✅ PASS: rmsprop_step is correct.")
    else:
        all_ok = False

    print()
    if all_ok:
        print("🎉 RMSProp tests passed.")
    else:
        print("🚨 Some RMSProp tests failed.")

    return all_ok


# ────────────────────────────────────────────────────────────────
# Adam
# ────────────────────────────────────────────────────────────────

def test_adam():
    """Tests Adam initialization and Adam updates against torch.optim.Adam."""

    all_ok = True

    params = _make_params()
    m, v, t = optimizers.init_adam(params)

    init_ok = (
        t == 0
        and len(m) == len(params)
        and len(v) == len(params)
        and all(torch.allclose(m_i, torch.zeros_like(p)) for m_i, p in zip(m, params))
        and all(torch.allclose(v_i, torch.zeros_like(p)) for v_i, p in zip(v, params))
    )

    if init_ok:
        print("✅ PASS: init_adam is correct.")
    else:
        print("❌ FAIL: init_adam is incorrect.")
        if t != 0:
            print(f"   t is {t}, expected 0.")
        if len(m) != len(params):
            print(f"   m has length {len(m)}, expected {len(params)}.")
        if len(v) != len(params):
            print(f"   v has length {len(v)}, expected {len(params)}.")
        _check_tensor_list("m", m, [torch.zeros_like(p) for p in params])
        _check_tensor_list("v", v, [torch.zeros_like(p) for p in params])
        all_ok = False

    params = _make_params()
    reference_params = _clone_params(params)

    learning_rate = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    state = optimizers.init_adam(params)

    reference_optimizer = torch.optim.Adam(
        reference_params,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps
    )

    reference_optimizer.step()
    state = optimizers.adam_step(
        params,
        state,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps
    )

    _make_second_grads(params)
    _make_second_grads(reference_params)

    reference_optimizer.step()
    state = optimizers.adam_step(
        params,
        state,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps
    )

    m, v, t = state

    expected_params = [p.detach() for p in reference_params]
    actual_params = [p.detach() for p in params]

    expected_m = [
        reference_optimizer.state[p]["exp_avg"].detach()
        for p in reference_params
    ]

    expected_v = [
        reference_optimizer.state[p]["exp_avg_sq"].detach()
        for p in reference_params
    ]

    expected_t = 2

    params_ok = _check_tensor_list(
        "parameters after adam_step",
        actual_params,
        expected_params
    )

    m_ok = _check_tensor_list("m", m, expected_m)
    v_ok = _check_tensor_list("v", v, expected_v)
    t_ok = _check_scalar("t", t, expected_t)

    if params_ok and m_ok and v_ok and t_ok:
        print("✅ PASS: adam_step is correct.")
    else:
        all_ok = False

    print()
    if all_ok:
        print("🎉 Adam tests passed.")
    else:
        print("🚨 Some Adam tests failed.")

    return all_ok


# ────────────────────────────────────────────────────────────────
# AdamW
# ────────────────────────────────────────────────────────────────

def test_adamw():
    """Tests Adam initialization and AdamW updates against torch.optim.AdamW."""

    all_ok = True

    params = _make_params()
    m, v, t = optimizers.init_adam(params)

    init_ok = (
        t == 0
        and len(m) == len(params)
        and len(v) == len(params)
        and all(torch.allclose(m_i, torch.zeros_like(p)) for m_i, p in zip(m, params))
        and all(torch.allclose(v_i, torch.zeros_like(p)) for v_i, p in zip(v, params))
    )

    if init_ok:
        print("✅ PASS: init_adam is correct.")
    else:
        print("❌ FAIL: init_adam is incorrect.")
        if t != 0:
            print(f"   t is {t}, expected 0.")
        if len(m) != len(params):
            print(f"   m has length {len(m)}, expected {len(params)}.")
        if len(v) != len(params):
            print(f"   v has length {len(v)}, expected {len(params)}.")
        _check_tensor_list("m", m, [torch.zeros_like(p) for p in params])
        _check_tensor_list("v", v, [torch.zeros_like(p) for p in params])
        all_ok = False

    params = _make_params()
    reference_params = _clone_params(params)

    learning_rate = 0.1
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.01

    state = optimizers.init_adam(params)

    reference_optimizer = torch.optim.AdamW(
        reference_params,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay
    )

    reference_optimizer.step()
    state = optimizers.adamw_step(
        params,
        state,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay
    )

    _make_second_grads(params)
    _make_second_grads(reference_params)

    reference_optimizer.step()
    state = optimizers.adamw_step(
        params,
        state,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay
    )

    m, v, t = state

    expected_params = [p.detach() for p in reference_params]
    actual_params = [p.detach() for p in params]

    expected_m = [
        reference_optimizer.state[p]["exp_avg"].detach()
        for p in reference_params
    ]

    expected_v = [
        reference_optimizer.state[p]["exp_avg_sq"].detach()
        for p in reference_params
    ]

    expected_t = 2

    params_ok = _check_tensor_list(
        "parameters after adamw_step",
        actual_params,
        expected_params
    )

    m_ok = _check_tensor_list("m", m, expected_m)
    v_ok = _check_tensor_list("v", v, expected_v)
    t_ok = _check_scalar("t", t, expected_t)

    if params_ok and m_ok and v_ok and t_ok:
        print("✅ PASS: adamw_step is correct.")
    else:
        all_ok = False

    print()
    if all_ok:
        print("🎉 AdamW tests passed.")
    else:
        print("🚨 Some AdamW tests failed.")

    return all_ok
