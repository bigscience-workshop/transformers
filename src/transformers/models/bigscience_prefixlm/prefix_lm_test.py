import random

import torch

from transformers import GPT2TokenizerFast

from transformers.models.bigscience_prefixlm.configuration_prefixlm import PrefixLMConfig
from transformers.models.bigscience_prefixlm.modeling_prefixlm import PrefixLMLMHeadModel

def sample_new_random_id(input_id: int, vocab_size: int):
    assert vocab_size > 1, "Otherwise it's hard to find another id"

    replace_token_id = None
    while input_id == replace_token_id or replace_token_id is None:
        replace_token_id = random.randint(0, vocab_size)

    return replace_token_id

def prefix_invariants():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    config = PrefixLMConfig(n_layer=3, n_embd=34, n_head=2)
    model = PrefixLMLMHeadModel(config)

    random_inputs = ["Hello my name is Thomas, and I enjoy"]
    tokenized_inputs = tokenizer.batch_encode_plus(random_inputs, return_tensors="pt")

    print(tokenized_inputs)

    batch_size, sequence_length = tokenized_inputs["input_ids"].shape

    # Create a mock attention mask that's prefix type of attention.
    # True when we keep attention score, and False where we remove it.
    attention_mask = torch.tril(torch.ones(batch_size, sequence_length, sequence_length).bool()[:, None, ...])
    attention_mask[:, :, :sequence_length // 2, :sequence_length // 2] = True

    model.eval()
    with torch.no_grad():
        baseline = model(tokenized_inputs["input_ids"], attention_mask = attention_mask)

    # Now we change an id from the prefix
    random_prefix_index = random.randint(0, sequence_length // 2)
    updated_prefix_input_ids = tokenized_inputs["input_ids"].clone()
    for batch_id in range(batch_size):
        updated_prefix_input_ids[batch_id, random_prefix_index] = sample_new_random_id(updated_prefix_input_ids[batch_id, random_prefix_index], config.vocab_size)

    with torch.no_grad():
        updated_prefix_outputs = model(updated_prefix_input_ids, attention_mask=attention_mask)

    # Expect all outputs to change whether that's tokens before or after
    # is_different stores Tr
    is_different = torch.sum(torch.abs(baseline.logits - updated_prefix_outputs.logits), dim=-1)
    assert torch.all(is_different), f"Expected all of the to be True, got {is_different}"

    # Now we change an id from the suffix part
    random_suffix_index = random.randint(sequence_length // 2 + 1, sequence_length - 1)
    updated_suffix_input_ids = tokenized_inputs["input_ids"].clone()
    for batch_id in range(batch_size):
        updated_suffix_input_ids[batch_id, random_suffix_index] = sample_new_random_id(
            updated_suffix_input_ids[batch_id, random_suffix_index], config.vocab_size)

    with torch.no_grad():
        updated_suffix_outputs = model(updated_suffix_input_ids, attention_mask=attention_mask)

    # Expect all outputs to change
    print(random_suffix_index, attention_mask)
    is_different = torch.sum(torch.abs(baseline.logits - updated_suffix_outputs.logits), dim=-1)
    assert not torch.any(is_different[:, :random_suffix_index]), f"Expected all of them to be False, got {is_different[:, :random_suffix_index]}"
    assert torch.all(is_different[:, random_suffix_index:]), f"Expected all of them to be True, got {is_different[:, random_suffix_index:]}"


def prefix_lm_infer():
    # TODO we want to check that the generate function works
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    config = PrefixLMConfig(n_layer=3, n_embd=34, n_head=2)
    model = PrefixLMLMHeadModel(config)

    random_inputs = ["Hello my name is Thomas, and I enjoy"]
    tokenized_inputs = tokenizer.batch_encode_plus(random_inputs, return_tensors="pt")
    print(tokenized_inputs)
    print(model.generate(tokenized_inputs["input_ids"], use_cache=True))

def main():
    # prefix_invariants()
    prefix_lm_infer()

if __name__ == "__main__":
    main()