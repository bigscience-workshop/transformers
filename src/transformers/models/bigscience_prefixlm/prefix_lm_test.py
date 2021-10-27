import random

import torch

from transformers import GPT2TokenizerFast, PrefixLMConfig, PrefixLMLMHeadModel

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
    print(f"Difference in prefix inputs: {updated_prefix_input_ids != tokenized_inputs['input_ids']}")

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
    print(f"Difference in suffix inputs: {updated_suffix_input_ids != tokenized_inputs['input_ids']}")

    with torch.no_grad():
        updated_suffix_outputs = model(updated_suffix_input_ids, attention_mask=attention_mask)

    # Expect all outputs to change
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
    output = model.generate(tokenized_inputs["input_ids"],
                         use_cache=True,
                         return_dict_in_generate=True,
                         output_attentions=True,
                         output_hidden_states=True,
                         output_scores=True
                         )
    scores = output.scores
    attentions = output.attentions
    hidden_states = output.hidden_states
    print(tokenizer.batch_decode(model.generate(tokenized_inputs["input_ids"], use_cache=True)))

def prefix_lm_1B3_infer():
    # TODO we want to check that the generate function works
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # config = PrefixLMConfig.from_json_file("/Users/thomas/code/bigscience/checkpoints/megatron/iter_0048000/mp_rank_00/config.json")
    model = PrefixLMLMHeadModel.from_pretrained("/Users/thomas/code/bigscience/checkpoints/megatron/iter_0048000/mp_rank_00")
    model.eval()

    random_inputs = [" The US president is"]
    # random_inputs = [" A novel is a long, fictional narrative which describes intimate human experiences. The novel in the modern era usually makes use of a literary prose style. The development of the prose novel"]
    # random_inputs = [" In 1982, Trump was listed on the initial Forbes list of wealthy individuals as having a share of his family's estimated $200 million net worth. His financial losses in the 1980s caused him to be dropped from the list between 1990 and 1995.[42] After Trump filed mandatory financial disclosure forms with the Federal Election Commission (FEC) in July 2015, he publicly announced a net worth of about $10 billion while the records released by the FEC showed"]
    # random_inputs = [" A FEW MILES south of Soledad, the Salinas River drops in close to the hillside bank and runs deep and green. The water is warm too, for it has slipped twinkling over the yellow sands in the sunlight before reaching the narrow pool. On one side of the river the golden foothill slopes curve up to the strong and rocky Gabilan mountains, but on the valley side the water is lined with trees - willows fresh and green with every spring, carrying in their lower leaf junctures the debris of the winter’s flooding; and sycamores with mottled, white, recumbent limbs and branches that arch over the pool. On the sandy bank under the trees the leaves lie deep and so crisp that a lizard makes a great skittering if he runs among them. Rabbits"]
    tokenized_inputs = tokenizer.batch_encode_plus(random_inputs, return_tensors="pt")
    print(tokenized_inputs)

    # # DEBUG
    # def test_attention_mask(attn_mask):
    #     output = model(tokenized_inputs["input_ids"], attention_mask=attn_mask)
    #     # print(output.logits.shape)
    #     # print(output.shape)
    #     # new_token = torch.argmax(output, dim=-1)
    #     new_token = torch.topk(output.logits[:, -1, :], 10, dim=-1).indices.squeeze(0)
    #     # print(new_token)
    #     print(tokenizer.convert_ids_to_tokens(new_token))
    #     return output
    # batch_size, sequence_length = tokenized_inputs["input_ids"].shape
    # attention_mask = torch.ones(batch_size, 1, sequence_length, sequence_length, dtype=torch.bool,
    #                             device=tokenized_inputs["input_ids"].device)
    # attention_mask[:, :, :-1, -1] = False
    # output1 = test_attention_mask(attention_mask)
    # output2 = test_attention_mask(~attention_mask)
    # print(output1.logits, output2.logits)
    # print(output1.logits == output2.logits)
    # raise Exception
    #
    # # DEBUG
    # random_inputs2 = [" My first name is"]
    # tokenized_inputs2 = tokenizer.batch_encode_plus(random_inputs2, return_tensors="pt")
    # def test_random_inputs(lol):
    #     output = model(lol, attention_mask=attention_mask)
    #     new_token = torch.topk(output.logits[:, -1, :], 50, dim=-1).indices.squeeze(0)
    #     # print(new_token)
    #     print(output.logits.shape, tokenizer.convert_ids_to_tokens(new_token))
    #     return output
    # output1 = test_random_inputs(tokenized_inputs["input_ids"])
    # output2 = test_random_inputs(tokenized_inputs2["input_ids"])
    # print(output1.logits, output2.logits)
    #
    # raise Exception

    # Generate
    output_ids = model.generate(tokenized_inputs["input_ids"], use_cache=True, max_length=256)
    print(output_ids)
    print(tokenizer.batch_decode(output_ids))
    print([tokenizer.convert_ids_to_tokens(row) for row in output_ids])

def main():
    # prefix_invariants()
    # prefix_lm_infer()
    prefix_lm_1B3_infer()

if __name__ == "__main__":
    main()