import html
import time
from pathlib import Path

import torch
from IPython.display import HTML, display

import dataset
import llm
import sampling


def latest_checkpoint(checkpoint_dir):
    """
    Finds the final checkpoint, or the latest numbered checkpoint if needed.

    Args:
        checkpoint_dir (str or Path): Directory containing model checkpoints.

    Returns:
        checkpoint_path (Path): Checkpoint path.
    """

    checkpoint_dir = Path(checkpoint_dir)
    final_path = checkpoint_dir / "tiny_llm_final.pt"
    if final_path.exists():
        return final_path

    checkpoints = sorted(checkpoint_dir.glob("tiny_llm_step_*.pt"))
    if len(checkpoints) == 0:
        raise FileNotFoundError("No trained tiny LLM checkpoint found in the cache folder.")
    return checkpoints[-1]


def load_saved_model(cache_dir=None, device=None):
    """
    Loads the tokenizer and trained tiny LLM from the shared cache folder.

    Args:
        cache_dir (str or Path, optional): Folder containing saved tokenizer and checkpoint files.
        device (str, optional): Device for the model.

    Returns:
        model (TinyLLM): Loaded model.
        tokenizer (BytePairTokenizer): Loaded tokenizer.
        info (dict): Paths, checkpoint step, and device.
    """

    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent / "13c-llm" / "cache"
    cache_dir = Path(cache_dir)

    checkpoint_dir = cache_dir / "checkpoints"
    tokenizer_path = cache_dir / "bpe_tokenizer.json"

    checkpoint_path = latest_checkpoint(checkpoint_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, checkpoint = llm.load_model_checkpoint(checkpoint_path, device=device)

    saved_tokenizer_path = checkpoint.get("metadata", {}).get("tokenizer_path")
    if saved_tokenizer_path is not None and Path(saved_tokenizer_path).exists():
        tokenizer_path = Path(saved_tokenizer_path)

    tokenizer = dataset.load_bpe_tokenizer(tokenizer_path)
    info = {
        "tokenizer_path": str(tokenizer_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint["step"],
        "device": device,
    }
    return model, tokenizer, info


def render_token_spans(token_pieces, status):
    if len(token_pieces) == 0:
        if status == "generating":
            return "<span style='background:#fde68a; color:#000000;'> </span>"
        return ""

    spans = []
    for index, piece in enumerate(token_pieces):
        style = "color:#000000;"
        if status == "generating" and index == len(token_pieces) - 1:
            style += " background:#fde68a;"
        spans.append("<span style='" + style + "'>" + html.escape(piece) + "</span>")
    return "".join(spans)


def render_answer_html(instruction, input_text, token_pieces, status, tokens_per_second=None):
    speed_text = "calculating"
    if tokens_per_second is not None:
        speed_text = f"{tokens_per_second:.2f} tokens/s"

    parts = [
        "<div style='font-family: ui-monospace, Consolas, monospace; line-height:1.45; max-width:900px; color:#000000;'>",
        "<div style='margin-bottom:8px;'>",
        "<strong>Instruction</strong><br>",
        "<pre style='white-space:pre-wrap; margin:4px 0 0 0; padding:10px; background:#f8fafc; border:1px solid #cbd5e1; color:#000000;'>",
        html.escape(instruction),
        "</pre></div>",
    ]

    if input_text:
        parts += [
            "<div style='margin-bottom:8px;'>",
            "<strong>Input</strong><br>",
            "<pre style='white-space:pre-wrap; margin:4px 0 0 0; padding:10px; background:#f8fafc; border:1px solid #cbd5e1; color:#000000;'>",
            html.escape(input_text),
            "</pre></div>",
        ]

    parts += [
        "<div style='margin-bottom:8px;'>",
        "<strong>Generated answer</strong>",
        "<span style='margin-left:10px; color:#334155;'>",
        html.escape(status),
        "</span><br>",
        "<pre style='white-space:pre-wrap; min-height:120px; margin:4px 0 0 0; padding:10px; background:#ecfdf5; border:1px solid #86efac; color:#000000;'>",
        render_token_spans(token_pieces, status),
        "</pre>",
        "<div style='font-size:12px; color:#334155;'>generated tokens: "
        + str(len(token_pieces))
        + " | speed: "
        + speed_text
        + "</div>",
        "</div></div>",
    ]
    return "".join(parts)


@torch.no_grad()
def show_answer(model, tokenizer, instruction, input_text="", max_new_tokens=200, temperature=0.8, device=None):
    """
    Generates an answer and updates the notebook output once per produced token.

    Args:
        model (TinyLLM): Loaded language model.
        tokenizer (BytePairTokenizer): Matching tokenizer.
        instruction (str): User instruction.
        input_text (str): Optional instruction input.
        max_new_tokens (int): Generation limit.
        temperature (float): Sampling temperature. Use 0 for greedy decoding.
        device (str, optional): Device. If None, uses the model device.
    """

    if device is None:
        device = next(model.parameters()).device

    example = {"instruction": instruction, "input": input_text, "output": ""}
    prompt = dataset.format_instruction_prompt(example)
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    token_ids = list(prompt_ids)
    token_pieces = []
    start_time = None

    model.eval()
    handle = display(
        HTML(render_answer_html(instruction, input_text, token_pieces, "generating")),
        display_id=True,
    )

    for _ in range(max_new_tokens):
        context = token_ids[-model.context_window :]
        model_input = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(model_input)
        next_logits = logits[0, -1]
        next_id = sampling.sample_next_token(next_logits, temperature)

        token_ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break

        if start_time is None:
            start_time = time.perf_counter()

        token_pieces.append(tokenizer.decode([next_id]))
        elapsed = max(time.perf_counter() - start_time, 1e-9)
        tokens_per_second = len(token_pieces) / elapsed
        handle.update(
            HTML(render_answer_html(instruction, input_text, token_pieces, "generating", tokens_per_second))
        )

    final_speed = None
    if start_time is not None:
        final_speed = len(token_pieces) / max(time.perf_counter() - start_time, 1e-9)

    handle.update(
        HTML(render_answer_html(instruction, input_text, token_pieces, "done", final_speed))
    )
