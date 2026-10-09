import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from dataset import BPE_Token_Dataset, bpe_tokenizer_fingerprint


def make_causal_mask(seq_len, device):

    """
    Creates a causal attention mask.

    A True entry means that attention to that position is blocked.
    Position i may attend to positions <= i, but not to future positions.

    Args:
        seq_len (int): Sequence length.
        device (torch.device): Device where the mask should live.

    Returns:
        mask (Tensor): Boolean mask with shape (seq_len, seq_len).
    """

    ################################################################
    # TODO
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask == 1
    ################################################################

    return mask


class TinyLLM(nn.Module):

    """A small causal Transformer language model."""

    def __init__(
        self,
        vocab_size,
        context_window,
        embedding_dim,
        num_heads,
        num_layers,
        dropout,
    ):
        """
        Creates a small causal Transformer language model.

        Args:
            vocab_size (int): Number of tokenizer vocabulary entries.
            context_window (int): Maximum number of input tokens.
            embedding_dim (int): Size of token and position embeddings.
            num_heads (int): Number of attention heads per Transformer layer.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout probability inside Transformer layers.
        """

        super().__init__()

    ################################################################
    # TODO

        self.vocab_size = vocab_size
        self.context_window = context_window
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_window, embedding_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.output_head = nn.Linear(embedding_dim, vocab_size)

        self.register_buffer(
            "position_ids",
            torch.arange(context_window, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            make_causal_mask(context_window, "cpu"),
            persistent=False,
        )

    def forward(self, model_input):

        """
        Predicts next-token logits for a batch of token ids.

        Args:
            model_input (Tensor): Token ids with shape (batch_size, sequence_length).

        Returns:
            logits (Tensor): Vocabulary logits with shape
                (batch_size, sequence_length, vocab_size).
        """

        _, seq_len = model_input.shape

        if seq_len > self.context_window:
            raise ValueError("Sequence is longer than model context_window.")

        positions = self.position_ids[:, :seq_len]
        x = self.token_embedding(model_input) + self.position_embedding(positions)
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        x = self.blocks(x, mask=causal_mask)
        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits

    ################################################################



def training_step(model, batch, optimizer, device):

    """
    Runs one optimization step for next-token prediction.

    Args:
        model (TinyLLM): Language model.
        batch (dict): Batch with model_input and labels.
        optimizer (Optimizer): Optimizer used to update the model.
        device (str or torch.device): Computation device.

    Returns:
        loss_value (float or None): Training loss, or None if no supervised
            target token is present in the batch.
    """

    ################################################################
    # TODO
    model.train()

    model_input = batch["model_input"].to(device)
    labels = batch["labels"].to(device)

    logits = model(model_input)
    supervised = labels != -100
    if supervised.sum() == 0:
        return None

    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ################################################################

    return float(loss.item())
































































######### REST IS GIVEN #########


def make_warmup_cosine_scheduler(
    optimizer,
    total_steps,
    warmup_steps=0,
    min_learning_rate_ratio=0.1,
):
    """
    Creates a learning-rate scheduler with short warmup and cosine decay.

    Args:
        optimizer (Optimizer): Optimizer whose learning rate is scheduled.
        total_steps (int): Total number of training updates.
        warmup_steps (int): Number of initial warmup updates.
        min_learning_rate_ratio (float): Final learning-rate ratio relative to
            the base learning rate.

    Returns:
        scheduler (LambdaLR): PyTorch learning-rate scheduler.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")

    warmup_steps = max(0, min(int(warmup_steps), int(total_steps)))
    min_learning_rate_ratio = float(min_learning_rate_ratio)

    def multiplier(step_index):
        if warmup_steps > 0 and step_index < warmup_steps:
            return float(step_index + 1) / float(warmup_steps)

        decay_steps = max(1, total_steps - warmup_steps)
        progress = (step_index - warmup_steps) / decay_steps
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate_ratio + (1.0 - min_learning_rate_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def save_model_checkpoint(
    model,
    optimizer,
    step,
    losses,
    path,
    scheduler=None,
    metadata=None,
):
    """
    Saves a training checkpoint to disk.

    Args:
        model (TinyLLM): Model being trained.
        optimizer (Optimizer): Optimizer state to save.
        step (int): Current training step.
        losses (list): Loss history.
        path (str or Path): Output checkpoint path.
        scheduler (LRScheduler, optional): Scheduler state to save.
        metadata (dict, optional): Extra run information.

    Returns:
        path (Path): Path of the written checkpoint.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "losses": list(losses),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "vocab_size": model.vocab_size,
            "context_window": model.context_window,
            "embedding_dim": model.embedding_dim,
            "num_heads": model.num_heads,
            "num_layers": model.num_layers,
            "dropout": model.dropout,
        },
        "metadata": metadata or {},
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)
    return path


def load_model_checkpoint(path, device=None):
    """
    Loads a TinyLLM checkpoint saved by save_model_checkpoint.

    Args:
        path (str or Path): Checkpoint path.
        device (str, optional): Device for the loaded model.

    Returns:
        model (TinyLLM): Reconstructed model with loaded weights.
        checkpoint (dict): Full checkpoint dictionary.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(Path(path), map_location=device)
    model = TinyLLM(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def _format_seconds(seconds):
    if seconds is None:
        return "unknown"

    seconds = int(max(0, seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def train_tiny_llm(
    model,
    dataloader,
    steps=100,
    learning_rate=3e-4,
    device=None,
    warmup_steps=0,
    min_learning_rate_ratio=1.0,
    checkpoint_dir=None,
    checkpoint_every=None,
    save_final_checkpoint=False,
    checkpoint_prefix="tiny_llm",
    checkpoint_metadata=None,
    progress_callback=None,
    progress_every=10,
):
    """
    Trains the tiny language model for a fixed number of update steps.

    Args:
        model (TinyLLM): Language model.
        dataloader (DataLoader): Tokenized instruction mini-batches.
        steps (int): Number of optimizer updates.
        learning_rate (float): AdamW learning rate.
        device (str, optional): Device. If None, chooses CUDA when available.
        warmup_steps (int): Number of short learning-rate warmup steps.
        min_learning_rate_ratio (float): Final LR ratio for cosine decay.
        checkpoint_dir (str or Path, optional): Directory for model checkpoints.
        checkpoint_every (int, optional): Save every this many training steps.
        save_final_checkpoint (bool): Whether to save a final checkpoint too.
        checkpoint_prefix (str): Filename prefix for checkpoints.
        checkpoint_metadata (dict, optional): Extra metadata stored in checkpoints.
        progress_callback (callable, optional): Receives a metrics dictionary.
        progress_every (int): Emit progress every this many updates.

    Returns:
        losses (list): Training loss values.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = None
    if warmup_steps > 0 or min_learning_rate_ratio < 1.0:
        scheduler = make_warmup_cosine_scheduler(
            optimizer,
            total_steps=steps,
            warmup_steps=warmup_steps,
            min_learning_rate_ratio=min_learning_rate_ratio,
        )

    losses = []
    iterator = iter(dataloader)
    start_time = time.perf_counter()
    tokens_seen = 0
    supervised_tokens_seen = 0
    loss_ema = None

    try:
        batches_per_epoch = len(dataloader)
    except TypeError:
        batches_per_epoch = None

    updates = 0
    attempts = 0
    max_attempts = steps * 10
    last_checkpoint_path = None

    while updates < steps and attempts < max_attempts:
        attempts += 1

        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        current_learning_rate = optimizer.param_groups[0]["lr"]
        loss = training_step(model, batch, optimizer, device)
        if loss is None:
            continue

        losses.append(loss)
        updates += 1
        tokens_seen += int(batch["model_input"].numel())
        supervised_tokens_seen += int((batch["labels"] != -100).sum().item())

        if scheduler is not None:
            scheduler.step()

        loss_ema = loss if loss_ema is None else 0.95 * loss_ema + 0.05 * loss

        checkpoint_path = None
        if checkpoint_dir is not None and checkpoint_every is not None:
            if checkpoint_every > 0 and updates % checkpoint_every == 0:
                checkpoint_path = save_model_checkpoint(
                    model,
                    optimizer,
                    updates,
                    losses,
                    Path(checkpoint_dir) / f"{checkpoint_prefix}_step_{updates:06d}.pt",
                    scheduler=scheduler,
                    metadata={
                        "device": str(device),
                        "learning_rate": learning_rate,
                        "warmup_steps": warmup_steps,
                        "min_learning_rate_ratio": min_learning_rate_ratio,
                        **(checkpoint_metadata or {}),
                    },
                )
                last_checkpoint_path = checkpoint_path

        elapsed_seconds = time.perf_counter() - start_time
        steps_per_second = updates / elapsed_seconds if elapsed_seconds > 0 else 0.0
        eta_seconds = None
        estimated_total_seconds = None
        if steps_per_second > 0:
            eta_seconds = (steps - updates) / steps_per_second
            estimated_total_seconds = steps / steps_per_second

        recent_window = losses[-min(50, len(losses)) :]
        recent_loss = sum(recent_window) / len(recent_window)
        epochs_done = None
        if batches_per_epoch:
            epochs_done = updates / batches_per_epoch

        metrics = {
            "step": updates,
            "total_steps": steps,
            "loss": loss,
            "loss_ema": loss_ema,
            "recent_loss": recent_loss,
            "learning_rate": current_learning_rate,
            "next_learning_rate": optimizer.param_groups[0]["lr"],
            "steps_per_second": steps_per_second,
            "tokens_per_second": tokens_seen / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            "tokens_seen": tokens_seen,
            "supervised_tokens_seen": supervised_tokens_seen,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
            "estimated_total_seconds": estimated_total_seconds,
            "elapsed": _format_seconds(elapsed_seconds),
            "eta": _format_seconds(eta_seconds),
            "estimated_total_runtime": _format_seconds(estimated_total_seconds),
            "epochs_done": epochs_done,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "last_checkpoint_path": str(last_checkpoint_path) if last_checkpoint_path is not None else None,
        }

        should_report = updates == 1 or updates == steps
        if progress_every is not None and progress_every > 0:
            should_report = should_report or updates % progress_every == 0

        if should_report:
            if progress_callback is not None:
                progress_callback(metrics)

    if len(losses) == 0:
        raise ValueError("No supervised output tokens found. Increase max_length.")

    if checkpoint_dir is not None and save_final_checkpoint:
        save_model_checkpoint(
            model,
            optimizer,
            updates,
            losses,
            Path(checkpoint_dir) / f"{checkpoint_prefix}_final.pt",
            scheduler=scheduler,
            metadata={
                "device": str(device),
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps,
                "min_learning_rate_ratio": min_learning_rate_ratio,
                **(checkpoint_metadata or {}),
            },
        )

    return losses


def build_tiny_llm(tokenizer, context_window):

    """
    Creates the default tiny language model used in the notebook.

    Args:
        tokenizer (BytePairTokenizer): Tokenizer providing vocab_size.
        context_window (int): Maximum sequence length.

    Returns:
        model (TinyLLM): Small causal Transformer.
    """

    return TinyLLM(
        vocab_size=tokenizer.vocab_size,
        context_window=context_window,
        embedding_dim=256,
        num_heads=4,
        num_layers=8,
        dropout=0.05,
    )


_TOKENIZED_DATASET_CACHE_VERSION = 1


def _json_hash(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def _dataset_fingerprint(dataset):
    dataset_type = type(dataset).__module__ + "." + type(dataset).__name__

    if isinstance(dataset, Subset):
        indices = list(dataset.indices)
        return {
            "type": dataset_type,
            "length": len(dataset),
            "indices_hash": _json_hash(indices),
            "parent": _dataset_fingerprint(dataset.dataset),
        }

    dataset_path = getattr(dataset, "path", None)
    if dataset_path is not None:
        dataset_path = Path(dataset_path)
        try:
            stat = dataset_path.stat()
        except OSError:
            stat = None

        if stat is not None:
            return {
                "type": dataset_type,
                "length": len(dataset),
                "path": str(dataset_path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }

    hasher = hashlib.sha256()
    for index in range(len(dataset)):
        encoded = json.dumps(
            dataset[index],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "little"))
        hasher.update(encoded)

    return {
        "type": dataset_type,
        "length": len(dataset),
        "content_hash": hasher.hexdigest(),
    }


def _tokenized_dataset_cache_metadata(dataset, tokenizer, max_length, ignore_index):
    return {
        "version": _TOKENIZED_DATASET_CACHE_VERSION,
        "dataset": _dataset_fingerprint(dataset),
        "tokenizer_fingerprint": bpe_tokenizer_fingerprint(tokenizer),
        "max_length": int(max_length),
        "ignore_index": int(ignore_index),
    }


def _torch_load_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _has_valid_token_ids(tensor, vocab_size):
    if tensor.numel() == 0:
        return True

    valid = (0 <= tensor) & (tensor < vocab_size)
    return bool(valid.all().item())


def _is_cache_example_compatible(example, tokenizer, max_length, ignore_index):
    if not isinstance(example, dict):
        return False
    if set(example.keys()) != {"model_input", "labels"}:
        return False

    model_input = example["model_input"]
    labels = example["labels"]
    if not torch.is_tensor(model_input) or not torch.is_tensor(labels):
        return False
    if model_input.dtype != torch.long or labels.dtype != torch.long:
        return False
    if tuple(model_input.shape) != (max_length,) or tuple(labels.shape) != (max_length,):
        return False
    if not _has_valid_token_ids(model_input, tokenizer.vocab_size):
        return False

    target_labels = labels[labels != ignore_index]
    if not _has_valid_token_ids(target_labels, tokenizer.vocab_size):
        return False

    return True


def _load_tokenized_dataset_cache(cache_path, expected_metadata, tokenizer, expected_length):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None, "not found"

    try:
        payload = _torch_load_cache(cache_path)
    except Exception:
        return None, "could not be read"

    if not isinstance(payload, dict):
        return None, "has an invalid format"
    if payload.get("metadata") != expected_metadata:
        return None, "metadata does not match"

    examples = payload.get("examples")
    if not isinstance(examples, list) or len(examples) != expected_length:
        return None, "has invalid contents"

    max_length = expected_metadata["max_length"]
    ignore_index = expected_metadata["ignore_index"]
    for example in examples:
        if not _is_cache_example_compatible(example, tokenizer, max_length, ignore_index):
            return None, "is incompatible with the current tokenizer"

    return examples, "loaded"


def _save_tokenized_dataset_cache(cache_path, metadata, examples):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": metadata,
            "examples": examples,
        },
        cache_path,
    )


def make_tiny_dataloader(
    dataset,
    tokenizer,
    max_length=256,
    batch_size=8,
    max_examples=512,
    shuffle=True,
    num_workers=0,
    preprocess_workers=None,
    cache_path=None,
):
    """
    Builds a DataLoader for fixed-length tokenized instruction examples.

    Args:
        dataset (Dataset): Raw instruction dataset.
        tokenizer (BytePairTokenizer): Tokenizer used by BPE_Token_Dataset.
        max_length (int): Fixed context length.
        batch_size (int): Number of examples per mini-batch.
        max_examples (int): Optional limit for faster experiments.
        shuffle (bool): Whether to shuffle examples.
        num_workers (int): Number of DataLoader workers.
        preprocess_workers (int): Number of worker processes used to pre-tokenize examples.
        cache_path (str or Path): Optional path for saving/reusing tokenized examples.

    Returns:
        dataloader (DataLoader): Mini-batches with model_input and labels.
    """

    if max_examples is not None:
        size = min(max_examples, len(dataset))
        dataset = Subset(dataset, list(range(size)))

    if preprocess_workers is None:
        preprocess_workers = num_workers

    ignore_index = -100
    cache_metadata = None
    tokenized_cache = None
    if cache_path is not None:
        cache_metadata = _tokenized_dataset_cache_metadata(
            dataset,
            tokenizer,
            max_length,
            ignore_index,
        )
        tokenized_cache, cache_status = _load_tokenized_dataset_cache(
            cache_path,
            cache_metadata,
            tokenizer,
            len(dataset),
        )
        if tokenized_cache is None:
            print(
                "Tokenized dataset cache "
                + cache_status
                + "; converting raw dataset to token tensors and saving to "
                + str(cache_path)
            )
        else:
            print("Loaded tokenized dataset cache from " + str(cache_path))

    token_dataset = BPE_Token_Dataset(
        dataset,
        tokenizer,
        max_length=max_length,
        ignore_index=ignore_index,
        preprocess_workers=preprocess_workers,
        show_progress=tokenized_cache is None,
        pretokenized_cache=tokenized_cache,
    )
    if cache_path is not None and tokenized_cache is None:
        _save_tokenized_dataset_cache(cache_path, cache_metadata, token_dataset._cache)
        print("Saved converted tokenized dataset cache to " + str(cache_path))

    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2

    return DataLoader(token_dataset, **dataloader_kwargs)
