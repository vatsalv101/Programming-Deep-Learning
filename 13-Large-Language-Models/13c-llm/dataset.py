from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import Dataset

from tokenizer import BytePairTokenizer

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


_BPE_WORKER_DATASET = None
_BPE_WORKER_TOKENIZER = None
_BPE_WORKER_MAX_LENGTH = None
_BPE_WORKER_IGNORE_INDEX = None
_BPE_WORKER_MODEL_INPUT_TEMPLATE = None
_BPE_WORKER_LABELS_TEMPLATE = None


def _init_bpe_tokenize_worker(dataset, tokenizer, max_length, ignore_index):
    global _BPE_WORKER_DATASET
    global _BPE_WORKER_TOKENIZER
    global _BPE_WORKER_MAX_LENGTH
    global _BPE_WORKER_IGNORE_INDEX
    global _BPE_WORKER_MODEL_INPUT_TEMPLATE
    global _BPE_WORKER_LABELS_TEMPLATE

    _BPE_WORKER_DATASET = dataset
    _BPE_WORKER_TOKENIZER = tokenizer
    _BPE_WORKER_MAX_LENGTH = max_length
    _BPE_WORKER_IGNORE_INDEX = ignore_index
    _BPE_WORKER_MODEL_INPUT_TEMPLATE = torch.full(
        (max_length,), tokenizer.pad_id, dtype=torch.long
    )
    _BPE_WORKER_LABELS_TEMPLATE = torch.full(
        (max_length,), ignore_index, dtype=torch.long
    )


def _tokenize_bpe_example(
    example,
    tokenizer,
    max_length,
    ignore_index,
    model_input_template,
    labels_template,
):
    prompt = format_instruction_prompt(example)
    output = example["output"].strip() + "\n"

    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    output_ids = tokenizer.encode(output, add_eos=True)
    token_ids = prompt_ids + output_ids
    token_ids = token_ids[: max_length + 1]

    token_ids = torch.tensor(token_ids, dtype=torch.long)

    model_input = model_input_template.clone()
    labels = labels_template.clone()

    length = len(token_ids) - 1
    model_input[:length] = token_ids[:-1]
    labels[:length] = token_ids[1:]

    first_output_target = len(prompt_ids) - 1
    labels[:first_output_target] = ignore_index

    return {
        "model_input": model_input,
        "labels": labels,
    }


def _tokenize_bpe_worker_index(index):
    return _tokenize_bpe_example(
        _BPE_WORKER_DATASET[index],
        _BPE_WORKER_TOKENIZER,
        _BPE_WORKER_MAX_LENGTH,
        _BPE_WORKER_IGNORE_INDEX,
        _BPE_WORKER_MODEL_INPUT_TEMPLATE,
        _BPE_WORKER_LABELS_TEMPLATE,
    )


def format_instruction_prompt(example):

    """
    Formats one dataset row as an instruction prompt.

    Args:
        example (dict): Dataset row with instruction, input and output fields.

    Returns:
        prompt (str): Prompt text without the target output.
    """

    instruction = example["instruction"].strip()
    input_text = example["input"].strip()

    if input_text:
        prompt = (
            "### Instruction:\n"
            + instruction
            + "\n\n"
            + "### Input:\n"
            + input_text
            + "\n\n"
            + "### Output:\n"
        )
    else:
        prompt = "### Instruction:\n" + instruction + "\n\n" + "### Output:\n"

    return prompt


def format_instruction_text(example):

    """
    Formats one dataset row as the full text used for tokenizer training.

    Args:
        example (dict): Dataset row with instruction, input and output fields.

    Returns:
        text (str): Prompt followed by output.
    """

    text = format_instruction_prompt(example) + example["output"].strip() + "\n"

    return text


def iter_instruction_texts(dataset):

    """
    Iterates over formatted instruction texts.

    Args:
        dataset (Dataset): Instruction dataset.

    Yields:
        text (str): Formatted instruction text.
    """

    for index in range(len(dataset)):
        yield format_instruction_text(dataset[index])


def train_bpe_tokenizer(dataset, vocab_size=1024, min_frequency=2, max_examples=None):

    """
    Trains a BPE tokenizer on an instruction dataset.

    Args:
        dataset (Dataset): Instruction dataset.
        vocab_size (int): Maximum tokenizer vocabulary size.
        min_frequency (int): Minimum pair count required for a merge.
        max_examples (int): Optional number of examples used for training.

    Returns:
        tokenizer (BytePairTokenizer): Trained tokenizer.
    """

    texts = iter_instruction_texts(dataset)
    if max_examples is not None:
        texts = islice(texts, max_examples)

    tokenizer = BytePairTokenizer()
    tokenizer.train(texts, vocab_size=vocab_size, min_frequency=min_frequency)

    return tokenizer


class BPE_Token_Dataset(Dataset):

    """Fixed-length tokenized instruction dataset with ignored prompt labels."""

    def __init__(
        self,
        dataset,
        tokenizer,
        max_length=512,
        ignore_index=-100,
        show_progress=False,
        preprocess_workers=0,
        pretokenized_cache=None,
    ):

        """
        Creates a tokenized instruction dataset.

        Args:
            dataset (Dataset): Instruction dataset.
            tokenizer (BytePairTokenizer): Tokenizer used for encoding text.
            max_length (int): Fixed number of model input tokens returned per example.
            ignore_index (int): Label value ignored by cross entropy loss.
            show_progress (bool): Whether to show progress while pre-tokenizing examples.
            preprocess_workers (int): Number of worker processes used for pre-tokenizing examples.
            pretokenized_cache (list): Optional cached tokenized examples.
        """

        super().__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.show_progress = show_progress
        if preprocess_workers is None:
            preprocess_workers = 0
        self.preprocess_workers = int(preprocess_workers)

        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")
        if self.preprocess_workers < 0:
            raise ValueError("preprocess_workers must be non-negative.")

        self._model_input_template = torch.full(
            (self.max_length,), self.tokenizer.pad_id, dtype=torch.long
        )
        self._labels_template = torch.full(
            (self.max_length,), self.ignore_index, dtype=torch.long
        )
        if pretokenized_cache is None:
            self._cache = self._pretokenize_dataset()
        else:
            if len(pretokenized_cache) != len(self.dataset):
                raise ValueError("pretokenized_cache length must match dataset length.")
            self._cache = pretokenized_cache

    def _with_progress(self, iterable, total, description):
        if self.show_progress and tqdm is not None:
            return tqdm(
                iterable,
                total=total,
                desc=description,
                unit="examples",
                dynamic_ncols=True,
            )
        return iterable

    def _pretokenize_dataset(self):
        size = len(self.dataset)
        indices = range(len(self.dataset))
        worker_count = min(self.preprocess_workers, size)

        if worker_count > 1:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_bpe_tokenize_worker,
                initargs=(
                    self.dataset,
                    self.tokenizer,
                    self.max_length,
                    self.ignore_index,
                ),
            ) as executor:
                examples = executor.map(
                    _tokenize_bpe_worker_index,
                    indices,
                    chunksize=8,
                )
                examples = self._with_progress(
                    examples,
                    size,
                    "Pre-tokenizing BPE dataset (" + str(worker_count) + " workers)",
                )
                return list(examples)

        indices = self._with_progress(
            indices,
            size,
            "Pre-tokenizing BPE dataset",
        )
        return [self._tokenize_example(index) for index in indices]

    def __len__(self):
        return len(self.dataset)

    def _tokenize_example(self, index):
        return _tokenize_bpe_example(
            self.dataset[index],
            self.tokenizer,
            self.max_length,
            self.ignore_index,
            self._model_input_template,
            self._labels_template,
        )

    def __getitem__(self, index):

        """
        Returns one tokenized next-token prediction example.

        Args:
            index (int): Dataset index.

        Returns:
            example (dict): Tokenized example with model input and labels.
                model_input (torch.Tensor): Tensor of model input token ids with shape (max_length,), possibly padded with the tokenizer pad id.
                labels (torch.Tensor): Tensor of label token ids with shape (max_length,), padded with ignore_index.
        """

        return self._cache[index]


######### REST IS GIVEN #########


def bpe_tokenizer_fingerprint(tokenizer):
    """
    Creates a stable fingerprint for a trained byte-pair tokenizer.

    Args:
        tokenizer (BytePairTokenizer): Tokenizer to fingerprint.

    Returns:
        fingerprint (str): SHA-256 hex digest of tokenizer-defining state.
    """

    payload = {
        "special_tokens": list(tokenizer.special_tokens),
        "merges": [list(pair) for pair in tokenizer.merges],
        "vocab_size": int(tokenizer.vocab_size),
        "byte_offset": int(tokenizer.byte_offset),
        "first_merge_id": int(tokenizer.first_merge_id),
        "pad_id": int(tokenizer.pad_id),
        "bos_id": int(tokenizer.bos_id),
        "eos_id": int(tokenizer.eos_id),
        "unk_id": int(tokenizer.unk_id),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def save_bpe_tokenizer(tokenizer, path):
    """
    Saves a trained byte-pair tokenizer to a JSON file.

    Args:
        tokenizer (BytePairTokenizer): Trained tokenizer.
        path (str or Path): Output path.

    Returns:
        path (Path): Path of the written tokenizer file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "special_tokens": list(tokenizer.special_tokens),
        "merges": [list(pair) for pair in tokenizer.merges],
        "vocab_size": tokenizer.vocab_size,
        "fingerprint": bpe_tokenizer_fingerprint(tokenizer),
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    return path


def load_bpe_tokenizer(path):
    """
    Loads a byte-pair tokenizer saved by save_bpe_tokenizer.

    Args:
        path (str or Path): Tokenizer JSON path.

    Returns:
        tokenizer (BytePairTokenizer): Loaded tokenizer.
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    tokenizer = BytePairTokenizer(tuple(payload["special_tokens"]))
    tokenizer.merges = [tuple(pair) for pair in payload["merges"]]
    tokenizer.vocab_size = int(payload["vocab_size"])
    return tokenizer
