import torch

import weight_init


def _allclose_with_error(student, reference, atol=1e-8, rtol=1e-6):
    if student.dtype != reference.dtype:
        return False, None, f"wrong dtype: got {student.dtype}, expected {reference.dtype}"

    if student.shape != reference.shape:
        return False, None, f"wrong shape: got {student.shape}, expected {reference.shape}"

    max_abs_error = (student - reference).abs().max().item()

    if not torch.allclose(student, reference, atol=atol, rtol=rtol):
        return False, max_abs_error, f"max abs error: {max_abs_error:.6e}"

    return True, max_abs_error, None


def test_linear_layer_size():
    torch.manual_seed(0)

    test_cases = [
        dict(out_features=10, in_features=5, expected=(5, 10)),
        dict(out_features=20, in_features=15, expected=(15, 20)),
        dict(out_features=50, in_features=100, expected=(100, 50)),
    ]

    for cfg in test_cases:
        weight = torch.randn(cfg["out_features"], cfg["in_features"])
        expected_n_in, expected_n_out = cfg["expected"]
        n_in, n_out = weight_init.linear_layer_size(weight)

        if n_in != expected_n_in:
            print(f"❌ FAIL: n_in of linear_layer_size is incorrect. Expected {expected_n_in}, got {n_in}.")
            return

        if n_out != expected_n_out:
            print(f"❌ FAIL: n_out of linear_layer_size is incorrect. Expected {expected_n_out}, got {n_out}.")
            return

    print("✅ PASS: linear_layer_size is correct.")


def test_conv2d_layer_size():
    torch.manual_seed(0)

    test_cases = [
        dict(C_out=16, C_in=3, K_h=3, K_w=3, expected=(27, 144)),
        dict(C_out=32, C_in=16, K_h=5, K_w=5, expected=(400, 800)),
        dict(C_out=64, C_in=32, K_h=7, K_w=7, expected=(1568, 3136)),
    ]

    for cfg in test_cases:
        weight = torch.randn(cfg["C_out"], cfg["C_in"], cfg["K_h"], cfg["K_w"])
        expected_n_in, expected_n_out = cfg["expected"]
        n_in, n_out = weight_init.conv2d_layer_size(weight)

        if n_in != expected_n_in:
            print(f"❌ FAIL: n_in of conv2d_layer_size is incorrect. Expected {expected_n_in}, got {n_in}.")
            return

        if n_out != expected_n_out:
            print(f"❌ FAIL: n_out of conv2d_layer_size is incorrect. Expected {expected_n_out}, got {n_out}.")
            return

    print("✅ PASS: conv2d_layer_size is correct.")


def _test_weight_init(init_fn_, fn_name, test_cases):
    torch.manual_seed(0)

    weight = torch.randn(10, 10, requires_grad=True)
    try:
        init_fn_(weight, n_in=10, n_out=10)
    except RuntimeError as e:
        if "leaf variable that requires grad" in str(e).lower():
            print(f"❌ FAIL: gradient tracking is not disabled.")
            return
        else:
            print(f"❌ FAIL: {fn_name} raised an unexpected error: {e}")
            return

    for cfg in test_cases:
        weight = torch.randn(cfg["weight_shape"], dtype=torch.float64)
        old_weight = weight.clone()
        return_value = init_fn_(weight, cfg["n_in"], cfg["n_out"])

        if return_value is not None:
            print(f"❌ FAIL: {fn_name} should not return anything, but got {return_value}.")
            return

        if torch.equal(weight, old_weight):
            print(f"❌ FAIL: {fn_name} did not modify the weights in-place.")
            return

        if not torch.allclose(weight.mean(), torch.tensor(0.0, dtype=torch.float64), atol=cfg["atol"], rtol=0):
            print(f"❌ FAIL: mean of weights initialized by {fn_name} is not close to 0 (got {weight.mean().item():.6e}).")
            return

        expected_var = torch.tensor(cfg["expected_var"], dtype=torch.float64)

        ok, max_abs_error, msg = _allclose_with_error(weight.std(), expected_var.sqrt(), atol=cfg["atol"], rtol=0)
        if not ok:
            print(f"❌ FAIL: std of weights is incorrect ({msg}).")

            if torch.allclose(weight.std(), expected_var, atol=cfg["atol"], rtol=0):
                print("💡 SUGGESTION: It looks like you used the variance instead of the standard deviation to sample from a normal distribution.")

            return

    print(f"✅ PASS: {fn_name} is correct.")


def test_lecun_init():
    _test_weight_init(weight_init.lecun_init_, "lecun_init_", test_cases=[
        dict(weight_shape=(60, 40), n_in=40, n_out=60, expected_var=0.025, atol=0.01),
        dict(weight_shape=(64, 3, 3, 3), n_in=27, n_out=1728, expected_var=0.037037, atol=0.01),
        dict(weight_shape=(128, 64, 5, 5), n_in=1600, n_out=51200, expected_var=0.000625, atol=0.0001),
    ])


def test_xavier_init():
    _test_weight_init(weight_init.xavier_init_, "xavier_init_", test_cases=[
        dict(weight_shape=(60, 40), n_in=40, n_out=60, expected_var=0.02, atol=0.01),
        dict(weight_shape=(64, 3, 3, 3), n_in=27, n_out=1728, expected_var=0.001139, atol=0.001),
        dict(weight_shape=(128, 64, 5, 5), n_in=1600, n_out=51200, expected_var=0.000038, atol=0.0001),
    ])


def test_kaiming_he_init():
    _test_weight_init(weight_init.kaiming_he_init_, "kaiming_he_init_", test_cases=[
        dict(weight_shape=(60, 40), n_in=40, n_out=60, expected_var=0.05, atol=0.01),
        dict(weight_shape=(64, 3, 3, 3), n_in=27, n_out=1728, expected_var=0.074074, atol=0.01),
        dict(weight_shape=(128, 64, 5, 5), n_in=1600, n_out=51200, expected_var=0.00125, atol=0.0001),
    ])
