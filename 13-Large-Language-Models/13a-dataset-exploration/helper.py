from pathlib import Path
from html import escape
import random

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, display

from instruction_dataset import DEFAULT_OUTPUT_PATH, InstructionDataset


def resolve_dataset_path(path=None):
    """Return the folder-local raw dataset path."""
    if path is not None:
        return Path(path)

    return DEFAULT_OUTPUT_PATH


def configure_plotting():
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "axes.facecolor": "#f8fafc",
            "figure.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 1.0,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )


def load_dataset(path=None):
    return InstructionDataset.load(resolve_dataset_path(path))


def display_dataset_summary(dataset, path=None):
    dataset_path = resolve_dataset_path(path)
    summary = {
        "samples": len(dataset),
        "columns": ", ".join(dataset.column_names),
        "file size": f"{dataset_path.stat().st_size / 1024**2:.1f} MB",
    }

    cards = "".join(
        f"""
        <div style='border:1px solid #e2e8f0; border-radius:10px; padding:16px; background:white;'>
          <div style='font-size:13px; color:#64748b; text-transform:uppercase; letter-spacing:.06em;'>{escape(label)}</div>
          <div style='font-size:24px; font-weight:700; color:#0f172a; margin-top:6px;'>{escape(str(value))}</div>
        </div>
        """
        for label, value in summary.items()
    )
    display(
        HTML(
            "<div style='display:grid; "
            "grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); "
            f"gap:12px;'>{cards}</div>"
        )
    )


def text_block(title, text, color="#2563eb", max_chars=900):
    text = "" if text is None else str(text)
    shown = text if len(text) <= max_chars else text[:max_chars].rstrip() + " ..."
    shown = escape(shown).replace("\n", "<br>")
    if not shown:
        shown = "<span style='color:#94a3b8; font-style:italic;'>empty</span>"

    return f"""
    <div style='border:1px solid #e2e8f0; border-top:4px solid {color}; border-radius:10px; background:white; overflow:hidden;'>
      <div style='padding:10px 12px; background:#f8fafc; font-weight:700; color:#0f172a;'>{escape(title)}</div>
      <pre style='white-space:pre-wrap; margin:0; padding:12px; font-family:ui-monospace, SFMono-Regular, Consolas, monospace; font-size:13px; line-height:1.45; color:#1e293b;'>{shown}</pre>
    </div>
    """


def display_example(example, index=None):
    heading = "Dataset example" if index is None else f"Dataset example #{index}"
    html = f"""
    <div style='margin: 8px 0 14px 0;'>
      <h3 style='margin:0 0 4px 0; color:#0f172a;'>{escape(heading)}</h3>
      <div style='color:#64748b;'>Read left to right: prompt fields first, target output last.</div>
    </div>
    <div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:12px;'>
      {text_block('Instruction', example['instruction'], '#2563eb')}
      {text_block('Input', example['input'], '#14b8a6')}
      {text_block('Output', example['output'], '#f59e0b')}
    </div>
    """
    display(HTML(html))


def build_sample_table(dataset, sample_size=5000, seed=13):
    sample_size = min(sample_size, len(dataset))
    rng = random.Random(seed)
    sample_indices = rng.sample(range(len(dataset)), sample_size)

    records = []
    for index in sample_indices:
        row = dataset[index]
        instruction = row["instruction"] or ""
        input_text = row["input"] or ""
        output = row["output"] or ""
        records.append(
            {
                "index": index,
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "has_input": bool(input_text.strip()),
                "instruction_chars": len(instruction),
                "input_chars": len(input_text),
                "output_chars": len(output),
                "total_chars": len(instruction) + len(input_text) + len(output),
                "output_lines": output.count("\n") + 1 if output else 0,
            }
        )

    return pd.DataFrame(records)


def display_sample_stats(sample_df):
    stats = {
        "sampled examples": f"{len(sample_df):,}",
        "with input": f"{sample_df['has_input'].mean():.1%}",
        "median instruction": f"{sample_df['instruction_chars'].median():.0f} chars",
        "median output": f"{sample_df['output_chars'].median():.0f} chars",
        "longest output": f"{sample_df['output_chars'].max():,} chars",
    }

    palette = ["#2563eb", "#14b8a6", "#f59e0b", "#7c3aed", "#ef4444"]
    stat_cards = "".join(
        f"""
        <div style='border:1px solid #e2e8f0; border-bottom:4px solid {palette[i % len(palette)]}; border-radius:10px; padding:14px; background:white;'>
          <div style='font-size:12px; color:#64748b; text-transform:uppercase; letter-spacing:.06em;'>{escape(label)}</div>
          <div style='font-size:22px; font-weight:700; color:#0f172a; margin-top:6px;'>{escape(value)}</div>
        </div>
        """
        for i, (label, value) in enumerate(stats.items())
    )

    display(
        HTML(
            "<div style='display:grid; "
            "grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); "
            f"gap:12px;'>{stat_cards}</div>"
        )
    )


def plot_length_distributions(sample_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    columns = ["instruction_chars", "input_chars", "output_chars"]
    titles = ["Instruction length", "Input length", "Output length"]
    colors = ["#2563eb", "#14b8a6", "#f59e0b"]

    for ax, column, title, color in zip(axes, columns, titles, colors):
        clipped = sample_df[column].clip(upper=sample_df[column].quantile(0.98))
        ax.hist(clipped, bins=35, color=color, alpha=0.82)
        ax.set_title(title)
        ax.set_xlabel("characters, clipped at 98th percentile")
        ax.set_ylabel("examples")

    fig.tight_layout()
    plt.show()


def plot_input_presence(sample_df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    input_counts = sample_df["has_input"].map({True: "has input", False: "empty input"}).value_counts()
    ax.bar(input_counts.index, input_counts.values, color=["#14b8a6", "#94a3b8"])
    ax.set_title("How often does the optional input field appear?")
    ax.set_ylabel("sampled examples")
    for i, value in enumerate(input_counts.values):
        ax.text(i, value, f" {value:,}", va="bottom", ha="center", fontweight="bold", color="#0f172a")

    plt.show()


def show_examples(dataset, frame, title, n=3):
    display(HTML(f"<h3 style='margin-bottom:6px; color:#0f172a;'>{escape(title)}</h3>"))
    for _, row in frame.head(n).iterrows():
        display_example(dataset[int(row["index"])], index=int(row["index"]))


def interesting_example_frames(sample_df):
    return {
        "long_outputs": sample_df.sort_values("output_chars", ascending=False),
        "with_inputs": sample_df[sample_df["has_input"]].sort_values("input_chars", ascending=False),
        "without_inputs": sample_df[~sample_df["has_input"]].sample(
            min(3, (~sample_df["has_input"]).sum()), random_state=1
        ),
    }


def format_instruction_prompt(example):
    instruction = example["instruction"].strip()
    input_text = example["input"].strip()

    if input_text:
        return (
            "### Instruction:\n"
            + instruction
            + "\n\n"
            + "### Input:\n"
            + input_text
            + "\n\n"
            + "### Output:\n"
        )

    return "### Instruction:\n" + instruction + "\n\n" + "### Output:\n"


def format_instruction_text(example):
    return format_instruction_prompt(example) + example["output"].strip() + "\n"


def display_prompt_preview(example, index=None, max_chars=1600):
    text = format_instruction_text(example)
    shown = text[:max_chars].rstrip()
    if len(text) > max_chars:
        shown += "\n..."

    title = "Formatted training text" if index is None else f"Formatted training text #{index}"
    display(
        HTML(
            f"""
            <div style='border:1px solid #e2e8f0; border-radius:10px; background:white; overflow:hidden;'>
              <div style='padding:12px 14px; background:#0f172a; color:white;'>
                <div style='font-weight:700;'>{escape(title)}</div>
                <div style='font-size:13px; opacity:.78;'>This is the exact prompt-plus-answer layout used for training.</div>
              </div>
              <pre style='white-space:pre-wrap; margin:0; padding:14px; max-height:520px; overflow:auto; font-family:ui-monospace, SFMono-Regular, Consolas, monospace; font-size:13px; line-height:1.48; color:#1e293b; background:#f8fafc;'>{escape(shown)}</pre>
            </div>
            """
        )
    )


def find_keyword_matches(sample_df, keyword):
    return sample_df[
        sample_df["instruction"].str.contains(keyword, case=False, na=False)
        | sample_df["input"].str.contains(keyword, case=False, na=False)
        | sample_df["output"].str.contains(keyword, case=False, na=False)
    ]
