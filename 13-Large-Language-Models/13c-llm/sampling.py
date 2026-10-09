import torch

from dataset import format_instruction_prompt


def sample_next_token(logits, temperature):

    """
    Samples one token id from a vector of logits.

    Args:
        logits (Tensor): Logits with shape (vocab_size,).
        temperature (float): Sampling temperature. Use 0 for greedy decoding.

    Returns:
        token_id (int): Sampled token id.
    """

    ################################################################
    # TODO
    if temperature == 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    probabilities = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probabilities, num_samples=1)



    ################################################################

    return int(next_id.item())


@torch.no_grad()
def generate_tokens(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    device=None,
):
    
    """
    Generates token ids autoregressively from one prompt.

    Args:
        model (TinyLLM): Trained language model.
        tokenizer (BytePairTokenizer): Tokenizer for prompt encoding.
        prompt (str): Prompt text.
        max_new_tokens (int): Maximum generated tokens.
        temperature (float): Sampling temperature. Use 0 for greedy decoding.
        device (str, optional): Device. If None, uses the model device.

    Returns:
        token_ids (list): Prompt and generated token ids.
    """

    ################################################################
    # TODO

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    token_ids = tokenizer.encode(prompt, add_bos=True)

    for _ in range(max_new_tokens):
        context = token_ids[-model.context_window :]
        model_input = torch.tensor([context], dtype=torch.long, device=device)

        logits = model(model_input)
        next_logits = logits[0, -1]
        next_id = sample_next_token(next_logits, temperature)

        token_ids.append(next_id)

        if next_id == tokenizer.eos_id:
            break

    ################################################################

    return token_ids




















































################### REST IS GIVEN ####################

def answer_instruction(
    model,
    tokenizer,
    instruction,
    input_text="",
    max_new_tokens=100,
    temperature=0.8,
    device=None,
):
    """
    Formats one instruction prompt and decodes the model response.

    This helper is provided so that the exercise can focus on token sampling
    and autoregressive generation.

    Args:
        model (TinyLLM): Trained language model.
        tokenizer (BytePairTokenizer): Tokenizer.
        instruction (str): User instruction.
        input_text (str): Optional input text.
        max_new_tokens (int): Maximum generated tokens.
        temperature (float): Sampling temperature.
        device (str, optional): Device.

    Returns:
        answer (str): Decoded model output.
    """

    example = {
        "instruction": instruction,
        "input": input_text,
        "output": "",
    }
    prompt = format_instruction_prompt(example)

    token_ids = generate_tokens(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
    )

    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    output_ids = token_ids[len(prompt_ids) :]
    return tokenizer.decode(output_ids)
