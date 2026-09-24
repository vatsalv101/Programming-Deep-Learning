import torch
import io
import base64
from lstm_cells import sigmoid

try:
    from IPython.display import display, HTML, clear_output
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    _IN_NOTEBOOK = True
except ImportError:
    _IN_NOTEBOOK = False

def sample(h_prev, c_prev, seed_ix, length, Wx, Wh, b, Wy, by, vocab_size):
    """
    Samples a sequence of character indices from the model.

    Args:
        h_prev   (Tensor): Initial hidden state (H, 1).
        c_prev   (Tensor): Initial cell state (H, 1).
        seed_ix  (int): Starting character index.
        length   (int): Number of characters to generate.
        Wx, Wh, b, Wy, by: Model parameters.
        vocab_size (int): Vocabulary size.

    Returns:
        List[int]: Generated character indices.
    """
    H = h_prev.shape[0]
    x = torch.zeros(vocab_size, 1)
    x[seed_ix, 0] = 1.0
    h = h_prev.clone()
    c = c_prev.clone()
    indices = []

    for _ in range(length):
        tmp = Wx @ x + Wh @ h + b
        f = sigmoid(tmp[0:H])
        i = sigmoid(tmp[H:2*H])
        g = torch.tanh(tmp[2*H:3*H])
        o = sigmoid(tmp[3*H:4*H])

        c = f * c + i * g
        h = o * torch.tanh(c)

        y = Wy @ h + by
        p = torch.exp(y) / torch.exp(y).sum()

        ix = torch.multinomial(p.view(-1), num_samples=1).item()
        x = torch.zeros(vocab_size, 1)
        x[ix, 0] = 1.0
        indices.append(ix)

    return indices


def _escape_html(text):
    """Escape HTML special characters in generated text."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_loss_chart(loss_history):
    """Render loss_history as a base64-encoded PNG for embedding in HTML."""
    if not loss_history or len(loss_history) < 2:
        return ""
    fig, ax = plt.subplots(figsize=(8, 2.5), dpi=100)
    ax.plot(loss_history, linewidth=1.2, color='#1976D2')
    ax.set_xlabel('Iteration (×100)', fontsize=9)
    ax.set_ylabel('Smoothed Loss', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def _render_progress(n, max_iters, smooth_loss, initial_loss, sample_text, loss_history=None):
    """Build an HTML widget showing training progress."""
    # Progress bar
    if max_iters is not None:
        pct = min(n / max_iters * 100, 100)
        bar_html = (
            f'<div style="background:#e0e0e0;border-radius:4px;height:20px;width:100%;margin:6px 0;">'
            f'<div style="background:linear-gradient(90deg,#4CAF50,#66BB6A);height:100%;'
            f'border-radius:4px;width:{pct:.1f}%;transition:width 0.3s;"></div></div>'
            f'<div style="font-size:12px;color:#666;margin-bottom:8px;">'
            f'{n:,} / {max_iters:,} iterations ({pct:.1f}%)</div>'
        )
    else:
        bar_html = (
            f'<div style="font-size:12px;color:#666;margin-bottom:8px;">'
            f'Iteration {n:,}</div>'
        )

    # Loss indicator with color
    loss_ratio = smooth_loss / initial_loss if initial_loss > 0 else 1.0
    if loss_ratio < 0.5:
        loss_color = "#4CAF50"  # green
    elif loss_ratio < 0.8:
        loss_color = "#FF9800"  # orange
    else:
        loss_color = "#f44336"  # red

    loss_html = (
        f'<div style="font-size:18px;font-weight:bold;color:{loss_color};margin:4px 0;">'
        f'Loss: {smooth_loss:.2f}</div>'
    )

    # Sample text box
    escaped = _escape_html(sample_text) if sample_text else ""
    sample_html = ""
    if escaped:
        sample_html = (
            f'<details open style="margin-top:10px;">'
            f'<summary style="cursor:pointer;font-weight:bold;font-size:13px;color:#101;">'
            f'📝 Generated LSTM Sample (iter {n:,})</summary>'
            f'<pre style="background:#f5f5f5;border:1px solid #ddd;border-radius:6px;color:#101;'
            f'padding:10px;font-size:12px;line-height:1.5;max-height:400px;overflow-y:auto;'
            f'white-space:pre-wrap;word-wrap:break-word;margin-top:6px;">{escaped}</pre>'
            f'</details>'
        )

    # Loss chart
    chart_html = ""
    if loss_history is not None:
        b64 = _render_loss_chart(loss_history)
        if b64:
            chart_html = (
                f'<div style="margin-top:10px;">'
                f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:4px;" />'
                f'</div>'
            )

    return (
        f'<div style="font-family:system-ui,sans-serif;padding:12px;'
        f'border:1px solid #ddd;border-radius:8px;background:#fafafa;margin:4px 0;">'
        f'<div style="font-size:14px;font-weight:bold;margin-bottom:6px;">🔄 Training LSTM</div>'
        f'{bar_html}{loss_html}{chart_html}{sample_html}</div>'
    )


def train(data, vocab_size, char_to_ix, ix_to_char,
          model_params, seq_length, H, learning_rate,
          forward_pass, backward_pass, max_iters=None):
    """
    Trains the LSTM using Adagrad optimization.

    Args:
        data       (str): Full training text.
        vocab_size (int): Number of unique characters.
        char_to_ix (dict): Character → index mapping.
        ix_to_char (dict): Index → character mapping.
        model_params (tuple): (Wx, Wh, b, Wy, by) from init_model.
        seq_length (int): Sequence length for each training step.
        H          (int): Hidden size.
        learning_rate (float): Learning rate for Adagrad.
        forward_pass  (callable): Student's forward_pass function.
        backward_pass (callable): Student's backward_pass function.
        max_iters  (int or None): Stop after this many iterations. None = infinite.

    Returns:
        List[float]: Smoothed loss values recorded every 100 iterations.
    """
    Wx, Wh, b, Wy, by = model_params

    # Adagrad memory
    mWx = torch.zeros_like(Wx)
    mWh = torch.zeros_like(Wh)
    mWy = torch.zeros_like(Wy)
    mb  = torch.zeros_like(b)
    mby = torch.zeros_like(by)

    initial_loss = -torch.log(torch.tensor(1.0 / vocab_size)).item() * seq_length
    smooth_loss = initial_loss
    loss_history = []
    n, p = 0, 0
    hprev = torch.zeros(H, 1)
    cprev = torch.zeros(H, 1)
    last_sample_text = ""

    # For notebook: create a display handle to update in-place
    handle = None
    if _IN_NOTEBOOK:
        handle = display(HTML(_render_progress(0, max_iters, smooth_loss, initial_loss, "", loss_history)),
                         display_id=True)

    while True:
        # Reset if end of data or first iteration
        if p + seq_length + 1 >= len(data) or n == 0:
            hprev = torch.zeros(H, 1)
            cprev = torch.zeros(H, 1)
            p = 0

        inputs  = [char_to_ix[ch] for ch in data[p:p + seq_length]]
        targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

        # Sample from model every 1000 iterations
        if n % 1000 == 0:
            sample_ix = sample(hprev, cprev, inputs[0], 500,
                               Wx, Wh, b, Wy, by, vocab_size)
            last_sample_text = ''.join(ix_to_char[ix] for ix in sample_ix)

        # Forward & backward
        loss, cache = forward_pass(inputs, Wx, Wh, b, Wy, by, hprev, cprev, vocab_size)
        dWx, dWh, dWy, db, dby, hprev, cprev = backward_pass(targets, cache, Wy, by)

        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 100 == 0:
            loss_history.append(float(smooth_loss))

        # Update display every 500 iterations
        if n % 500 == 0:
            if _IN_NOTEBOOK and handle is not None:
                handle.update(HTML(_render_progress(
                    n, max_iters, smooth_loss, initial_loss, last_sample_text, loss_history)))
            else:
                if n % 1000 == 0:
                    print(f'iter {n:>6,}  |  loss: {smooth_loss:.2f}')
                    if last_sample_text:
                        print(f'  Sample: {last_sample_text[:80]}...')

        # Adagrad update
        for param, dparam, mem in zip(
            [Wx,  Wh,  Wy,  b,  by],
            [dWx, dWh, dWy, db, dby],
            [mWx, mWh, mWy, mb, mby]
        ):
            mem += dparam * dparam
            param += -learning_rate * dparam / torch.sqrt(mem + 1e-8)

        p += seq_length
        n += 1

        if max_iters is not None and n >= max_iters:
            break

    # Final update
    if _IN_NOTEBOOK and handle is not None:
        handle.update(HTML(_render_progress(
            n, max_iters, smooth_loss, initial_loss, last_sample_text, loss_history)))

    return loss_history
