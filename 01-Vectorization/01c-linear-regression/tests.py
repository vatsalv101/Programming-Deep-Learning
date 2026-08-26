import torch
from typing import Callable, Any, Optional, List
import sys
import model

class DiffOperation:
    def __init__(self, func: Callable[[dict[str, torch.Tensor]], Any], params: dict[str, torch.Tensor]):
        self.func = func
        self.params = params

    def __call__(self):
        return self.func(self.params)

def central_diff(do: DiffOperation, h=1e-6):
    grad = {}
    for n, b in do.params.items():
        b_flat = b.reshape(-1)
        grad_b = torch.zeros_like(b_flat)
        for d in range(len(grad_b)):
            original = b[d].item()
            b[d] = original - h
            f_low = do()

            b[d] = original + h
            f_high = do()

            b[d] = original

            grad_b[d] = (f_high - f_low) / (2 * h)
        grad[n] = grad_b.reshape(b.shape)

    return grad

def forward_diff(do: DiffOperation, h=1e-6):
    grad = {}

    f = do()

    for n, b in do.params.items():
        grad_b = torch.zeros_like(b)
        for d in range(len(grad_b)):
            original = b[d].item()
            b[d] = original + h
            f_high = do()

            b[d] = original

            grad_b[d] = (f_high - f) / h
        grad[n] = grad_b

    return grad

def test_mse(loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    y0 = torch.randn(5)
    y1 = torch.randn(5)
    predicted_mse = loss_func(y0, y1)
    true_mse = torch.nn.functional.mse_loss(y0, y1)
    if torch.isclose(true_mse, predicted_mse):
        print("MSE is correct")
    else:
        print(f"MSE is incorrect. Got {predicted_mse}, expected {true_mse}.")

def test_mse_grad(loss_grad_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    y0 = torch.randn(5, dtype=torch.float64)
    y1 = torch.randn(5, dtype=torch.float64)

    grad = loss_grad_func(y0, y1)
    do = DiffOperation(lambda d: model.mse(d['y'], y1), {'y': y0})
    numeric_grad = central_diff(do, h=1e-8)

    close = torch.isclose(grad, numeric_grad['y'])

    if torch.all(close):
        print("The gradient is correct")
    else:
        idx = torch.where(~close)[0]
        for i in idx:
            print(f"Error: Gradient at index {i} is wrong, got {grad[i]:.3f} instead of {numeric_grad[i]:.3f}.",
                  file=sys.stderr)

def test_linear_predict(predict_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]):
    X = torch.randn(15, 5)
    X_single = torch.randn(1, 5)
    w = torch.randn(5)
    b = torch.randn(1)

    prediction_single = predict_func(X_single, w, b)
    true_single = X_single @ w + b

    close = torch.allclose(prediction_single, true_single)

    if close:
        print("Single data: The prediction is correct")
    else:
        print(f"Single data: Error: Prediction is wrong, got {prediction_single:.3f} instead of {true_single:.3f}.",
                      file=sys.stderr)


    prediction = predict_func(X, w, b)
    true = X @ w + b

    close = torch.isclose(prediction, true)

    if torch.all(close):
        print("Batch data: The prediction is correct")
    else:
        if prediction.shape != true.shape:
            print(f"Batch data: Error: Wrong prediction shape. Expected {true.shape}, got {prediction.shape}.", file=sys.stderr)
        else:
            idx = torch.where(~close)[0]
            for i in idx:
                print(f"Batch data: Error: Prediction at index {i} is wrong, got {prediction[i]:.3f} instead of {true[i]:.3f}.",
                      file=sys.stderr)

def test_linear_gradients(gradient_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    N_test, D_test = 10, 4

    w = torch.randn(D_test, dtype=torch.float64)
    b = torch.randn(1, dtype=torch.float64)

    X = torch.randn(N_test, D_test, dtype=torch.float64)
    y = torch.randn(N_test, dtype=torch.float64)

    X_single_point = torch.randn(1, D_test, dtype=torch.float64)
    y_single_point = torch.randn(1, dtype=torch.float64)

    _, g_w_s, g_b_s = gradient_func(X_single_point, y_single_point, w, b)

    do = DiffOperation(lambda d: model.mse(model.linear_predict(X_single_point, d['w'], d['b']), y_single_point), {'w': w, 'b': b})
    cd_grads_s = central_diff(do, h=1e-8)

    close = torch.isclose(g_w_s, cd_grads_s['w'])

    if torch.all(close):
        print("Single data: The gradient for the weight is correct")
    else:
        idx = torch.where(~close)[0]
        for i in idx:
            print(f"Single data: Error: Weight gradient at index {i} is wrong, got {g_w_s[i]:.3f} instead of {cd_grads_s['w'][i]:.3f}.",
                  file=sys.stderr)

    if torch.isclose(g_b_s, cd_grads_s['b']):
        print("Single data: The gradient for the bias is correct")
    else:
        print(f"Single data: Bias gradient is incorrect. Got {g_b_s:3.f}, expected {cd_grads_s['b']:.3f}.", )

    _, g_w, g_b = gradient_func(X, y, w, b)


    do = DiffOperation(lambda d: model.mse(model.linear_predict(X, d['w'], d['b']), y), {'w': w, 'b': b})
    cd_grads = central_diff(do, h=1e-8)

    close = torch.isclose(g_w, cd_grads['w'])

    if torch.all(close):
        print("Batch data: The gradient for the weight is correct")
    else:
        idx = torch.where(~close)[0]
        for i in idx:
            print(f"Batch data: Error: Weight gradient at index {i} is wrong, got {g_w[i]:.3f} instead of {cd_grads['w'][i]:.3f}.",
                  file=sys.stderr)

    if torch.isclose(g_b, cd_grads['b']):
        print("Batch data: The gradient for the bias is correct")
    else:
        print(f"Batch data: Bias gradient is incorrect. Got {g_b:3.f}, expected {cd_grads['b']:.3f}.",)

def test_fit(fit_function: Callable[[torch.utils.data.DataLoader, float, int, Optional[torch.Tensor], Optional[torch.Tensor]], tuple[torch.Tensor, torch.Tensor, List[float]]]):
    N_test, D_test = 10, 4
    w_init = torch.randn(D_test, dtype=torch.float64)
    b_init = torch.randn(1, dtype=torch.float64)

    X = torch.randn(N_test, D_test, dtype=torch.float64)
    y = torch.randn(N_test, dtype=torch.float64)

    lr = 0.1

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=N_test)

    do = DiffOperation(lambda d: model.mse(model.linear_predict(X, d['w'], d['b']), y), {'w': w_init, 'b': b_init})
    cd_grads = central_diff(do, h=1e-6)

    w, b, _ = fit_function(dataloader, lr, 1, w_init.clone(), b_init.clone())

    numeric_w = w_init - lr * cd_grads['w']
    numeric_b = b_init - lr * cd_grads['b']
    close_w = torch.isclose(numeric_w, w)

    if torch.all(close_w):
        print("The new weight is correct.")
    else:
        idx = torch.where(~close_w)[0]
        for i in idx:
            print(f"Error: Weight at index {i} is wrong, got {w[i]:.3f} instead of {numeric_w[i]:.3f}.",
                  file=sys.stderr)
    if torch.isclose(b, numeric_b):
        print("The new bias is correct")
    else:
        print(f"New bias is incorrect. Got {b.item():.3f}, expected {numeric_b.item():.3f}.", file=sys.stderr)