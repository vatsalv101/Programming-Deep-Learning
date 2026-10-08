import html

import matplotlib


def is_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return "IPKernelApp" in shell.config
    except Exception:
        return False


if not is_notebook():
    matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt


TOKEN_COLORS = (
    "#93c5ea",
    "#afebb4",
    "#e6cda5",
    "#ed7ea3",
    "#c0a0ef",
    "#b9f0ed",
)

MASK_COLORS = {
    "ignored": ("#f4b8b8", "#9f6a6a", "prompt ignored"),
    "supervised": ("#9fe3a1", "#2e7d32", "loss used"),
    "padding": ("#e5e7eb", "#9ca3af", "padding ignored"),
}


def token_to_display_text(tokenizer, token_id):
    token_id = int(token_id)

    if token_id == tokenizer.pad_id:
        return "<pad>"
    if token_id == tokenizer.bos_id:
        return "<bos>"
    if token_id == tokenizer.eos_id:
        return "<eos>"

    return tokenizer.decode([token_id], skip_special_tokens=False)


def token_table(tokenizer, text):
    """
    Creates a table with token ids and readable token pieces.

    Args:
        tokenizer (BytePairTokenizer): Tokenizer used for encoding.
        text (str): Text to tokenize.

    Returns:
        rows (list): Token table rows.
    """

    token_ids = tokenizer.encode(text)
    rows = []

    for index in range(len(token_ids)):
        token_id = token_ids[index]
        byte_values = tokenizer.token_to_bytes(token_id)

        rows.append(
            {
                "index": index,
                "token_id": token_id,
                "text": token_to_display_text(tokenizer, token_id),
                "bytes": byte_values,
                "byte_count": len(byte_values),
            }
        )

    return rows


def show_tokenization(tokenizer, text):
    """
    Displays colored BPE tokens for a short piece of text.

    Args:
        tokenizer (BytePairTokenizer): Tokenizer used for encoding.
        text (str): Text to tokenize.

    Returns:
        html_text (HTML): HTML visualization in notebooks.
    """

    rows = token_table(tokenizer, text)
    parts = ["<div style='font-family: ui-monospace, Consolas, monospace;'>"]

    for row in rows:
        color = TOKEN_COLORS[row["index"] % len(TOKEN_COLORS)]
        title = f"id {row['token_id']}, bytes {row['byte_count']}"
        token_text = html.escape(row["text"])
        token_text = token_text.replace("\n", "\\n")
        token_text = token_text.replace(" ", "&nbsp;")

        parts.append(
            "<span title='"
            + html.escape(title)
            + "' style='display:inline-block; padding:4px 6px; margin:2px; "
            + "color:#000000; border:1px solid #cccccc; border-radius:4px; background:"
            + color
            + ";'>"
            + token_text
            + "</span>"
        )

    parts.append("</div>")
    html_text = "".join(parts)

    try:
        from IPython.display import HTML

        return HTML(html_text)
    except ImportError:
        return html_text


def html_result(html_text):
    try:
        from IPython.display import HTML

        return HTML(html_text)
    except ImportError:
        return html_text


def clean_token_text(piece):
    piece = html.escape(piece)
    piece = piece.replace("\n", "\\n")
    piece = piece.replace(" ", "&nbsp;")
    return piece


def tokenized_display_length(tokenizer, model_input, labels, ignore_index):
    display_length = len(labels)
    while display_length > 0:
        is_padding_input = int(model_input[display_length - 1]) == tokenizer.pad_id
        is_ignored_label = int(labels[display_length - 1]) == ignore_index
        if not (is_padding_input and is_ignored_label):
            break
        display_length -= 1
    return display_length


def token_label_state(tokenizer, model_token, label, ignore_index):
    if int(model_token) == tokenizer.pad_id and int(label) == ignore_index:
        return "padding"
    if int(label) == ignore_index:
        return "ignored"
    return "supervised"


def count_tokenized_states(tokenizer, model_input, labels, ignore_index):
    counts = {"ignored": 0, "supervised": 0, "padding": 0}
    for index in range(len(labels)):
        state = token_label_state(tokenizer, model_input[index], labels[index], ignore_index)
        counts[state] += 1
    return counts


def show_tokenized_example(
    tokenizer,
    tokenized_example,
    ignore_index=-100,
    max_tokens=None,
    show_padding=True,
):
    """
    Displays the shifted next-token labels for one tokenized dataset example.

    Args:
        tokenizer (BytePairTokenizer): Tokenizer used for decoding.
        tokenized_example (dict): Example from BPE_Token_Dataset.
        ignore_index (int): Label value ignored by cross entropy loss.
        max_tokens (int): Optional number of token positions to display.
        show_padding (bool): If True, show trailing padding positions.

    Returns:
        html_text (HTML): HTML visualization in notebooks.
    """

    model_input = tokenized_example["model_input"]
    labels = tokenized_example["labels"]
    if show_padding:
        display_length = len(labels)
    else:
        display_length = tokenized_display_length(tokenizer, model_input, labels, ignore_index)
    if max_tokens is not None:
        display_length = min(display_length, max_tokens)

    counts = count_tokenized_states(tokenizer, model_input, labels, ignore_index)
    parts = [
        "<div style='font-family: ui-monospace, Consolas, monospace; "
        + "line-height:1.6;'>",
        "<div style='margin:4px 0 8px 0; color:#334155;'>",
        "positions "
        + str(len(labels))
        + " | prompt ignored "
        + str(counts["ignored"])
        + " | loss used "
        + str(counts["supervised"])
        + " | padding "
        + str(counts["padding"])
        + "</div>",
        "<div style='margin:4px 0 10px 0;'>",
    ]

    for state in ("ignored", "supervised", "padding"):
        background, border, title = MASK_COLORS[state]
        parts.append(
            "<span style='display:inline-block; padding:2px 7px; margin-right:6px; "
            + "border:1px solid "
            + border
            + "; background:"
            + background
            + "; color:#111827; border-radius:4px;'>"
            + title
            + "</span>"
        )

    parts.append("</div>")

    for index in range(display_length):
        label = int(labels[index])
        state = token_label_state(tokenizer, model_input[index], label, ignore_index)
        background, border, title = MASK_COLORS[state]

        if state == "supervised":
            token_id = label
            piece = token_to_display_text(tokenizer, token_id)
        elif index + 1 < len(model_input):
            token_id = int(model_input[index + 1])
            piece = token_to_display_text(tokenizer, token_id)
        else:
            token_id = None
            piece = "<ignored>"

        if token_id is None:
            token_title = "none"
        else:
            token_title = str(token_id)

        parts.append(
            "<span title='"
            + html.escape(
                "position "
                + str(index)
                + " | target id "
                + token_title
                + " | "
                + title
            )
            + "' style='display:inline-block; padding:4px 6px; margin:2px; "
            + "color:#000000; border:1px solid "
            + border
            + "; border-radius:4px; background:"
            + background
            + ";'>"
            + clean_token_text(piece)
            + "</span>"
        )

    parts.append("</div>")
    html_text = "".join(parts)
    return html_result(html_text)


def show_loss_mask(tokenizer, tokenized_example, ignore_index=-100):
    """
    Displays which next-token labels contribute to the loss.

    Args:
        tokenizer (BytePairTokenizer): Tokenizer used for decoding.
        tokenized_example (dict): Example from BPE_Token_Dataset.

    Returns:
        html_text (HTML): HTML visualization in notebooks.
    """

    return show_tokenized_example(
        tokenizer,
        tokenized_example,
        ignore_index,
        show_padding=False,
    )


def show_tokenized_batch(tokenizer, batch, ignore_index=-100, max_rows=12, max_cols=128):
    """
    Displays a compact heatmap of ignored, supervised, and padding positions.

    Args:
        tokenizer (BytePairTokenizer): Tokenizer used for pad id.
        batch (dict): Batch from a DataLoader.
        ignore_index (int): Label value ignored by cross entropy loss.
        max_rows (int): Maximum number of batch rows to display.
        max_cols (int): Maximum number of token positions to display.

    Returns:
        html_text (HTML): HTML visualization in notebooks.
    """

    model_input = batch["model_input"]
    labels = batch["labels"]
    row_count = min(len(labels), max_rows)
    col_count = min(len(labels[0]), max_cols)

    parts = [
        "<div style='font-family: ui-monospace, Consolas, monospace;'>",
        "<div style='margin:4px 0 8px 0; color:#334155;'>",
        "showing "
        + str(row_count)
        + " examples x "
        + str(col_count)
        + " positions",
        "</div>",
        "<div style='margin:4px 0 10px 0;'>",
    ]

    for state in ("ignored", "supervised", "padding"):
        background, border, title = MASK_COLORS[state]
        parts.append(
            "<span style='display:inline-block; padding:2px 7px; margin-right:6px; "
            + "border:1px solid "
            + border
            + "; background:"
            + background
            + "; color:#111827; border-radius:4px;'>"
            + title
            + "</span>"
        )

    parts.append("</div>")

    for row in range(row_count):
        parts.append("<div style='display:flex; align-items:center; margin:2px 0;'>")
        parts.append(
            "<span style='display:inline-block; width:34px; color:#475569;'>"
            + str(row)
            + "</span>"
        )
        for col in range(col_count):
            state = token_label_state(tokenizer, model_input[row][col], labels[row][col], ignore_index)
            background, border, title = MASK_COLORS[state]
            parts.append(
                "<span title='row "
                + str(row)
                + ", position "
                + str(col)
                + ": "
                + title
                + "' style='display:inline-block; width:7px; height:16px; "
                + "margin-right:1px; border:1px solid "
                + border
                + "; background:"
                + background
                + ";'></span>"
            )
        parts.append("</div>")

    parts.append("</div>")
    html_text = "".join(parts)
    return html_result(html_text)


def plot_token_count_histogram(tokenizer, texts, max_examples=200, ax=None):
    """
    Plots how many BPE tokens each text needs.

    Args:
        tokenizer (BytePairTokenizer): Tokenizer used for encoding.
        texts (iterable): Texts to measure.
        max_examples (int): Maximum number of examples.
        ax (Axes): Optional matplotlib axes.

    Returns:
        ax (Axes): Matplotlib axes.
    """

    lengths = []
    count = 0

    for text in texts:
        lengths.append(len(tokenizer.encode(text)))
        count += 1
        if count >= max_examples:
            break

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))

    ax.hist(lengths, bins=30, color="#4c78a8", edgecolor="white")
    ax.set_title("BPE token counts")
    ax.set_xlabel("tokens per example")
    ax.set_ylabel("examples")

    return ax


def plot_merge_byte_lengths(tokenizer, max_merges=100, ax=None):
    """
    Plots how many bytes each learned merge token represents.

    Args:
        tokenizer (BytePairTokenizer): Tokenizer with learned merges.
        max_merges (int): Maximum number of merge tokens to plot.
        ax (Axes): Optional matplotlib axes.

    Returns:
        ax (Axes): Matplotlib axes.
    """

    ranks = []
    byte_lengths = []

    merge_count = len(tokenizer.merges) + 256 + 4 
    if merge_count > max_merges:
        merge_count = max_merges

    for index in range(merge_count):
        token_id = index
        ranks.append(index + 1)
        byte_lengths.append(len(tokenizer.token_to_bytes(token_id)))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))

    ax.plot(ranks, byte_lengths, color="#f58518")
    ax.set_title("UTF-8 Bytes and Learned BPE merge lengths")
    ax.set_xlabel("token id")
    ax.set_ylabel("bytes represented")

    return ax
