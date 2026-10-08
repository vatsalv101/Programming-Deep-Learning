import torch
from torch.utils.data import DataLoader

from dataset import (
    BPE_Token_Dataset,
    format_instruction_prompt,
    format_instruction_text,
    train_bpe_tokenizer,
)
from tokenizer import BytePairTokenizer


def test_byte_pair_tokenizer_interface():
    tokenizer = BytePairTokenizer()

    for name in ("pad_id", "bos_id", "eos_id", "unk_id", "vocab_size", "merges"):
        assert hasattr(tokenizer, name)

    for name in ("train", "encode", "decode", "token_to_bytes"):
        assert callable(getattr(tokenizer, name, None))


def test_special_tokens():
    tokenizer = BytePairTokenizer()

    assert tokenizer.pad_id == 0
    assert tokenizer.bos_id == 1
    assert tokenizer.eos_id == 2
    assert tokenizer.unk_id == 3
    assert tokenizer.vocab_size == 260
    assert tokenizer.decode([tokenizer.bos_id], skip_special_tokens=True) == ""
    assert tokenizer.decode([tokenizer.bos_id], skip_special_tokens=False) == "<bos>"


def test_bpe_learns_expected_merge():
    tokenizer = BytePairTokenizer()
    tokenizer.train(["abababab"], vocab_size=261, min_frequency=2)

    a_id = ord("a") + tokenizer.byte_offset
    b_id = ord("b") + tokenizer.byte_offset
    ab_id = tokenizer.first_merge_id

    assert tokenizer.merges == [(a_id, b_id)]
    assert tokenizer.vocab_size == 261
    assert tokenizer.encode("abab") == [ab_id, ab_id]
    assert tokenizer.token_to_bytes(ab_id) == list(b"ab")


def test_encode_decode_roundtrip():
    texts = [
        "hello world",
        "def add(a, b):\n    return a + b",
        "emoji: 😀, umlaut: äöü",
    ]

    tokenizer = BytePairTokenizer()
    tokenizer.train(texts, vocab_size=320, min_frequency=2)

    for text in texts:
        token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)

        assert isinstance(token_ids, list)
        assert token_ids[0] == tokenizer.bos_id
        assert token_ids[-1] == tokenizer.eos_id
        assert tokenizer.decode(token_ids) == text
        assert tokenizer.decode(token_ids, skip_special_tokens=False).startswith("<bos>")
        assert tokenizer.decode(token_ids, skip_special_tokens=False).endswith("<eos>")


def test_byte_pair_tokenizer():
    test_byte_pair_tokenizer_interface()
    test_special_tokens()
    test_bpe_learns_expected_merge()
    test_encode_decode_roundtrip()


def test_prompt_formatting():
    example = {
        "instruction": "Write a function.",
        "input": "Use Python.",
        "output": "def f():\n    pass",
    }

    prompt = format_instruction_prompt(example)
    text = format_instruction_text(example)

    assert "### Instruction:" in prompt
    assert "### Input:" in prompt
    assert prompt.endswith("### Output:\n")
    assert text.endswith("pass\n")

    no_input = {
        "instruction": "Write a function.",
        "input": "",
        "output": "pass",
    }
    no_input_prompt = format_instruction_prompt(no_input)

    assert "### Input:" not in no_input_prompt
    assert no_input_prompt.endswith("### Output:\n")


def test_bpe_dataset():
    examples = [
        {
            "instruction": "Write a Python function.",
            "input": "",
            "output": "def f():\n    return 1",
        },
        {
            "instruction": "Add two numbers.",
            "input": "a = 1, b = 2",
            "output": "a + b",
        },
    ]

    tokenizer = train_bpe_tokenizer(examples, vocab_size=300, min_frequency=1)
    dataset = BPE_Token_Dataset(examples, tokenizer, max_length=128)
    item = dataset[0]

    assert set(item.keys()) == {
        "model_input",
        "labels",
    }
    assert item["model_input"].dtype == torch.long
    assert item["model_input"].shape == (128,)
    assert item["labels"].shape == (128,)
    assert item["labels"].eq(-100).any()
    assert item["labels"].ne(-100).any()

    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["model_input"].shape[0] == 2
    assert batch["model_input"].shape == (2, 128)
    assert batch["labels"].shape == batch["model_input"].shape
    assert set(batch.keys()) == {"model_input", "labels"}
