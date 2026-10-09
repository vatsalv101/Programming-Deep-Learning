import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import llm
import sampling
from dataset import (
    BPE_Token_Dataset,
    bpe_tokenizer_fingerprint,
    load_bpe_tokenizer,
    save_bpe_tokenizer,
    train_bpe_tokenizer,
)


def tiny_examples():
    return [
        {
            "instruction": "Say hello.",
            "input": "",
            "output": "hello",
        },
        {
            "instruction": "Add two numbers.",
            "input": "1 and 2",
            "output": "3",
        },
        {
            "instruction": "Write a Python function.",
            "input": "",
            "output": "def f():\n    return 1",
        },
    ]


def tiny_tokenizer():
    return train_bpe_tokenizer(tiny_examples(), vocab_size=300, min_frequency=1)


def tiny_batch(tokenizer, max_length=32, batch_size=2):
    dataset = BPE_Token_Dataset(tiny_examples(), tokenizer, max_length=max_length)
    return next(iter(DataLoader(dataset, batch_size=batch_size)))


def tiny_model(tokenizer, context_window=32):
    return llm.TinyLLM(
        vocab_size=tokenizer.vocab_size,
        context_window=context_window,
        embedding_dim=32,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )


def test_causal_mask():
    mask = llm.make_causal_mask(4, "cpu")

    assert mask.dtype == torch.bool
    assert mask.shape == (4, 4)
    assert mask[0, 0].item() is False
    assert mask[0, 1].item() is True
    assert mask[2, 1].item() is False
    assert mask[3, 3].item() is False
    assert mask.equal(torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1))


def test_tiny_llm_forward():
    tokenizer = tiny_tokenizer()
    model = tiny_model(tokenizer, context_window=8)
    model_input = torch.randint(0, tokenizer.vocab_size, (2, 8))

    assert model.position_ids.shape == (1, 8)
    assert model.causal_mask.shape == (8, 8)
    assert model.causal_mask.dtype == torch.bool

    logits = model(model_input)

    assert logits.shape == (2, 8, tokenizer.vocab_size)
    assert torch.isfinite(logits).all()

    too_long = torch.randint(0, tokenizer.vocab_size, (1, 9))
    try:
        model(too_long)
    except ValueError:
        pass
    else:
        raise AssertionError("TinyLLM should reject sequences longer than context_window.")


def test_tiny_dataloader():
    tokenizer = tiny_tokenizer()
    dataloader = llm.make_tiny_dataloader(
        tiny_examples(),
        tokenizer,
        max_length=32,
        batch_size=2,
        max_examples=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    assert set(batch.keys()) == {"model_input", "labels"}
    assert batch["model_input"].shape == (2, 32)
    assert batch["labels"].shape == (2, 32)
    assert batch["labels"].eq(-100).any()
    assert batch["labels"].ne(-100).any()


def test_training_step():
    torch.manual_seed(0)
    tokenizer = tiny_tokenizer()
    model = tiny_model(tokenizer, context_window=32)
    batch = tiny_batch(tokenizer, max_length=32, batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = llm.training_step(model, batch, optimizer, "cpu")
    after = [parameter.detach().clone() for parameter in model.parameters()]

    assert loss is not None
    assert loss > 0
    assert any(not torch.equal(left, right) for left, right in zip(before, after))


def test_train_tiny_llm():
    torch.manual_seed(0)
    tokenizer = tiny_tokenizer()
    model = tiny_model(tokenizer, context_window=32)
    dataloader = llm.make_tiny_dataloader(
        tiny_examples(),
        tokenizer,
        max_length=32,
        batch_size=2,
        max_examples=3,
        shuffle=False,
    )

    losses = llm.train_tiny_llm(model, dataloader, steps=2, learning_rate=1e-3, device="cpu")

    assert len(losses) == 2
    assert all(loss > 0 for loss in losses)


def test_saved_tokenizer_round_trip():
    tokenizer = tiny_tokenizer()

    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "tokenizer.json"
        save_bpe_tokenizer(tokenizer, path)
        loaded = load_bpe_tokenizer(path)

    text = "hello"
    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.merges == tokenizer.merges
    assert loaded.encode(text, add_bos=True, add_eos=True) == tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
    )


def test_tiny_dataloader_cache_reuses_compatible_tokenizer():
    tokenizer = tiny_tokenizer()

    with tempfile.TemporaryDirectory() as temporary_dir:
        cache_path = Path(temporary_dir) / "tokenized_dataset.pt"
        dataloader = llm.make_tiny_dataloader(
            tiny_examples(),
            tokenizer,
            max_length=32,
            batch_size=2,
            max_examples=3,
            shuffle=False,
            cache_path=cache_path,
        )
        first_batch = next(iter(dataloader))

        assert cache_path.exists()

        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        cached_examples = cached["examples"]
        cached_examples[0]["model_input"][0] = tokenizer.eos_id
        torch.save(cached, cache_path)

        reused_dataloader = llm.make_tiny_dataloader(
            tiny_examples(),
            tokenizer,
            max_length=32,
            batch_size=2,
            max_examples=3,
            shuffle=False,
            cache_path=cache_path,
        )
        reused_batch = next(iter(reused_dataloader))

        assert first_batch["model_input"][0, 0] != tokenizer.eos_id
        assert reused_batch["model_input"][0, 0] == tokenizer.eos_id


def test_tiny_dataloader_cache_recomputes_for_new_tokenizer():
    tokenizer = tiny_tokenizer()
    changed_tokenizer = train_bpe_tokenizer(
        tiny_examples()
        + [
            {
                "instruction": "Repeat qwertyqwerty.",
                "input": "",
                "output": "qwertyqwerty qwertyqwerty",
            }
        ],
        vocab_size=300,
        min_frequency=1,
    )

    with tempfile.TemporaryDirectory() as temporary_dir:
        cache_path = Path(temporary_dir) / "tokenized_dataset.pt"
        llm.make_tiny_dataloader(
            tiny_examples(),
            tokenizer,
            max_length=32,
            batch_size=2,
            max_examples=3,
            shuffle=False,
            cache_path=cache_path,
        )
        llm.make_tiny_dataloader(
            tiny_examples(),
            changed_tokenizer,
            max_length=32,
            batch_size=2,
            max_examples=3,
            shuffle=False,
            cache_path=cache_path,
        )

        cached = torch.load(cache_path, map_location="cpu", weights_only=True)

    assert cached["metadata"]["tokenizer_fingerprint"] == bpe_tokenizer_fingerprint(
        changed_tokenizer
    )


def test_training_checkpoint_and_progress():
    torch.manual_seed(0)
    tokenizer = tiny_tokenizer()
    model = tiny_model(tokenizer, context_window=32)
    dataloader = llm.make_tiny_dataloader(
        tiny_examples(),
        tokenizer,
        max_length=32,
        batch_size=2,
        max_examples=3,
        shuffle=False,
    )
    history = []

    with tempfile.TemporaryDirectory() as temporary_dir:
        checkpoint_dir = Path(temporary_dir)
        losses = llm.train_tiny_llm(
            model,
            dataloader,
            steps=2,
            learning_rate=1e-3,
            device="cpu",
            warmup_steps=1,
            min_learning_rate_ratio=0.5,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=1,
            save_final_checkpoint=True,
            checkpoint_metadata={"tokenizer_path": "tokenizer.json"},
            progress_callback=history.append,
            progress_every=1,
        )

        assert (checkpoint_dir / "tiny_llm_step_000001.pt").exists()
        assert (checkpoint_dir / "tiny_llm_step_000002.pt").exists()
        assert (checkpoint_dir / "tiny_llm_final.pt").exists()
        loaded_model, checkpoint = llm.load_model_checkpoint(
            checkpoint_dir / "tiny_llm_final.pt",
            device="cpu",
        )

    assert len(losses) == 2
    assert len(history) == 2
    assert history[-1]["step"] == 2
    assert "learning_rate" in history[-1]
    assert "eta" in history[-1]
    assert isinstance(loaded_model, llm.TinyLLM)
    assert checkpoint["step"] == 2
    assert checkpoint["metadata"]["tokenizer_path"] == "tokenizer.json"


def test_sample_next_token():
    logits = torch.tensor([0.0, 1.0, 3.0, 2.0])

    assert sampling.sample_next_token(logits, temperature=0) == 2

    torch.manual_seed(0)
    sampled = sampling.sample_next_token(logits, temperature=1.0)
    assert 0 <= sampled < logits.shape[0]


def test_generation_helpers():
    torch.manual_seed(0)
    tokenizer = tiny_tokenizer()
    model = tiny_model(tokenizer, context_window=32)

    prompt = "### Instruction:\nSay hello.\n\n### Output:\n"
    token_ids = sampling.generate_tokens(
        model,
        tokenizer,
        prompt,
        max_new_tokens=3,
        temperature=0,
        device="cpu",
    )
    answer = sampling.answer_instruction(
        model,
        tokenizer,
        instruction="Say hello.",
        max_new_tokens=3,
        temperature=0,
        device="cpu",
    )

    assert isinstance(token_ids, list)
    assert len(token_ids) >= len(tokenizer.encode(prompt, add_bos=True))
    assert isinstance(answer, str)


def test_llm():
    test_causal_mask()
    test_tiny_llm_forward()
    test_tiny_dataloader()
    test_training_step()
    test_train_tiny_llm()
    test_saved_tokenizer_round_trip()
    test_tiny_dataloader_cache_reuses_compatible_tokenizer()
    test_tiny_dataloader_cache_recomputes_for_new_tokenizer()
    test_training_checkpoint_and_progress()
    test_sample_next_token()
    test_generation_helpers()
