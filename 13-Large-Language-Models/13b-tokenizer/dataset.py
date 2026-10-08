from itertools import islice

import torch
from torch.utils.data import Dataset

from tokenizer import BytePairTokenizer


def format_instruction_prompt(example):

    """
    Formats one dataset row as an instruction prompt.

    Args:
        example (dict): Dataset row with instruction, input and output fields.

    Returns:
        prompt (str): Prompt text without the target output.
    """

    ################################################################
    # TODO
    
    if len(example['input']) > 0:
        prompt = "### Instruction: \n" + example['instruction'].strip() + "\n\n" + "### Input:\n" + example['input'].strip() + "\n\n" + "### Output:\n"
    else:
        prompt = "### Instruction: \n" + example['instruction'].strip() + "\n\n" + "### Output:\n" 

    
    ################################################################

    return prompt


def format_instruction_text(example):

    """
    Formats one dataset row as the full text used for tokenizer training.

    Args:
        example (dict): Dataset row with instruction, input and output fields.

    Returns:
        text (str): Prompt followed by output.
    """

    ################################################################
    # TODO
    
    text = format_instruction_prompt(example) + example['output'].strip() + '\n'


    ################################################################

    return text


def iter_instruction_texts(dataset):

    """
    Iterates over formatted instruction texts.

    Args:
        dataset (Dataset): Instruction dataset.

    Yields:
        text (str): Formatted instruction text.
    """

    ################################################################
    # TODO
    
    for i in range(len(dataset)):
        yield format_instruction_text(dataset[i])

    ################################################################


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

    ################################################################
    # TODO
    
    texts = iter_instruction_texts(dataset)
    if max_examples is not None:
        text = islice(texts, max_examples)

    tokenizer = BytePairTokenizer()
    tokenizer.train(texts, vocab_size = vocab_size, min_frequency=min_frequency)

    ################################################################

    return tokenizer


class BPE_Token_Dataset(Dataset):

    """Fixed-length tokenized instruction dataset with ignored prompt labels."""

    def __init__(self, dataset, tokenizer, max_length=512, ignore_index=-100):

        """
        Creates a tokenized instruction dataset.
        Args:
            dataset (Dataset): Instruction dataset.
            tokenizer (BytePairTokenizer): Tokenizer used for encoding text.
            max_length (int): Fixed number of model input tokens returned per example.
            ignore_index (int): Label value ignored by cross entropy loss.
        """

        super().__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index

        if self.max_length <= 0:
            raise ValueError("max_length must be positive.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        """
        Returns one tokenized next-token prediction example.

        Args:
            index (int): Dataset index.

        Returns:
            example (dict): Tokenized example with model input and labels.
                model_input (torch.Tensor): Tensor of model input token ids with shape (max_length,), possibly padded with the tokenizer pad id.
                labels (torch.Tensor): Tensor of label token ids with shape (max_length,), padded with ignore_index (-100).
        """

        ################################################################
        # TODO
        
        example = self.dataset[index]

        prompt = format_instruction_prompt(example)
        output = example['output'].strip() + '\n'

        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        output_ids = self.tokenizer.encode(output, add_eos=True)
        token_ids = prompt_ids + output_ids
        token_ids = token_ids[:self.max_length + 1]

        input_tokens = torch.tensor(token_ids[:-1], dtype = torch.long)
        label_tokens = torch.tensor(token_ids[1:], dtype=torch.long)

        model_input = torch.full((self.max_length,), self.tokenizer.pad_id, dtype = torch.long)
        labels = torch.full((self.max_length,), self.ignore_index, dtype = torch.long)

        length = len(input_tokens)
        model_input[:length] = input_tokens
        labels[:length] = label_tokens

        first_output_target = len(prompt_ids) - 1
        labels[:first_output_target] = self.ignore_index

        ################################################################

        example = {
            "model_input": model_input,
            "labels": labels,
        }

        return example
