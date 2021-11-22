import random

import torch
import torch.nn.functional as F
from datasets import load_dataset, tqdm
from torch.utils.data import DataLoader

from transformers import GPT2TokenizerFast, PrefixLMConfig, PrefixLMLMHeadModel, GPT2LMHeadModel, DataCollator, \
    DataCollatorForLanguageModeling, BatchEncoding


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
    # model = PrefixLMLMHeadModel.from_pretrained("/home/thomaswang/code/bigscience/checkpoints/megatron/iter_0048000/mp_rank_00")
    # model = PrefixLMLMHeadModel.from_pretrained("/home/thomaswang/code/bigscience/checkpoints/megatron/350M/megatron/iter_0017569/mp_rank_00")
    # model = PrefixLMLMHeadModel.from_pretrained("/home/thomaswang/code/bigscience/checkpoints/megatron/1B3-unbiased/megatron/iter_0034500/mp_rank_00")
    # model = PrefixLMLMHeadModel.from_pretrained("/home/thomaswang/code/bigscience/checkpoints/megatron/350M-pile/megatron/iter_0021000/mp_rank_00")
    # model = PrefixLMLMHeadModel.from_pretrained("/home/thomaswang/code/bigscience/checkpoints/megatron/1B3-pile/megatron/iter_0018000/mp_rank_00")
    # model = PrefixLMLMHeadModel.from_pretrained("/home/thomaswang/code/bigscience/checkpoints/megatron/1B3-no-loss-on-targets-only/megatron/iter_0081000/mp_rank_00")
    model = PrefixLMLMHeadModel.from_pretrained("/home/thomaswang/code/bigscience/checkpoints/megatron/1B3-rescale-loss-per-position/megatron/iter_0021000/mp_rank_00")

    model.eval()
    model.cuda()

    # random_inputs = [" The"]
    # random_inputs = [" Steve Jobs is the most"]
    # random_inputs = [" The US president"]
    # random_inputs = [" The US president is"]
    # random_inputs = [" The US president is Barack"]
    # random_inputs = [" Donald Trump will"]
    # random_inputs = [" Barack"]
    # random_inputs = [" The tsar was deposed"]
    random_inputs = [" Artificial intelligence will be the best"]
    # random_inputs = [" Today, researchers have become"]
    # random_inputs = [" In 1991, the remains of Russian Tsar Nicholas II and his family (except for Alexei and Maria) are discovered. The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the remainder of the story. 1883 Western Siberia, a young Grigori Rasputin is asked by his father and a group of men to perform magic. Rasputin has a vision and denounces one of the men as a horse thief. Although his father initially slaps him for making such an accusation, Rasputin watches as the man is chased outside and beaten. Twenty years later, Rasputin sees a vision of the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous, with people, even a bishop, begging for his blessing." + "\nThe"]
    # random_inputs = [" A novel is a long, fictional narrative which describes intimate human experiences. The novel in the modern era usually makes use of a literary prose style. The development of the prose novel"]
    # random_inputs = [" In 1982, Trump was listed on the initial Forbes list of wealthy individuals as having a share of his family's estimated $200 million net worth. His financial losses in the 1980s caused him to be dropped from the list between 1990 and 1995.[42] After Trump filed mandatory financial disclosure forms with the Federal Election Commission (FEC) in July 2015, he publicly announced a net worth of about $10 billion while the records released by the FEC showed"]
    # random_inputs = [" A FEW MILES south of Soledad, the Salinas River drops in close to the hillside bank and runs deep and green. The water is warm too, for it has slipped twinkling over the yellow sands in the sunlight before reaching the narrow pool. On one side of the river the golden foothill slopes curve up to the strong and rocky Gabilan mountains, but on the valley side the water is lined with trees - willows fresh and green with every spring, carrying in their lower leaf junctures the debris of the winter’s flooding; and sycamores with mottled, white, recumbent limbs and branches that arch over the pool. On the sandy bank under the trees the leaves lie deep and so crisp that a lizard makes a great skittering if he runs among them. Rabbits"]
    # random_inputs = [" An eleven-time World Champion, he won consecutive World Championship 100 m, 200 m and 4 × 100 metres relay gold medals from 2009 to 2015, with the exception of a 100 m false start in 2011. He is the most successful male athlete of the World Championships. Bolt is the first athlete to win four World Championship titles in the 200 m and is one of the most successful in the 100 m with three titles."]
    # random_inputs = [" Once upon a time in a far land there lived a prince"]
    # random_inputs = [" do iran and afghanistan speak the same language"]
    # random_inputs = [ " I wonder how many people I've looked at all my life and never"]
    # random_inputs = [" Huggingface is a compagny that"]
    # random_inputs = [" HuggingFace will be a unicorn in"]

    # Some examples from https://github.com/allenai/macaw
    # random_inputs = [" James went camping in the woods, but forgot to bring a hammer to bang the tent pegs in. What else might he"]
    # random_inputs = [" Q: James went camping in the woods, but forgot to bring a hammer to bang the tent pegs in. What else might he use? M: (A) a leaf (B) a log (C) a worm\n "]
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
    output_ids = model.generate(tokenized_inputs["input_ids"].cuda(), use_cache=True, max_length=256)
    print(output_ids)
    print(tokenizer.batch_decode(output_ids))
    print([tokenizer.convert_ids_to_tokens(row) for row in output_ids])

def gpt2_1B3_infer():
    # TODO we want to check that the generate function works
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # config = PrefixLMConfig.from_json_file("/Users/thomas/code/bigscience/checkpoints/megatron/iter_0048000/mp_rank_00/config.json")
    model = GPT2LMHeadModel.from_pretrained(
        "/home/thomaswang/code/bigscience/checkpoints/tr3d-1B3-oscar-checkpoints/global_step60000")
    model.eval()
    model.cuda()

    # random_inputs = [" The"]
    random_inputs = [" The US president is"]
    # random_inputs = [" A novel is a long, fictional narrative which describes intimate human experiences. The novel in the modern era usually makes use of a literary prose style. The development of the prose novel"]
    # random_inputs = [" In 1982, Trump was listed on the initial Forbes list of wealthy individuals as having a share of his family's estimated $200 million net worth. His financial losses in the 1980s caused him to be dropped from the list between 1990 and 1995.[42] After Trump filed mandatory financial disclosure forms with the Federal Election Commission (FEC) in July 2015, he publicly announced a net worth of about $10 billion while the records released by the FEC showed"]
    # random_inputs = [" A FEW MILES south of Soledad, the Salinas River drops in close to the hillside bank and runs deep and green. The water is warm too, for it has slipped twinkling over the yellow sands in the sunlight before reaching the narrow pool. On one side of the river the golden foothill slopes curve up to the strong and rocky Gabilan mountains, but on the valley side the water is lined with trees - willows fresh and green with every spring, carrying in their lower leaf junctures the debris of the winter’s flooding; and sycamores with mottled, white, recumbent limbs and branches that arch over the pool. On the sandy bank under the trees the leaves lie deep and so crisp that a lizard makes a great skittering if he runs among them. Rabbits"]
    # random_inputs = [" An eleven-time World Champion, he won consecutive World Championship 100 m, 200 m and 4 × 100 metres relay gold medals from 2009 to 2015, with the exception of a 100 m false start in 2011. He is the most successful male athlete of the World Championships. Bolt is the first athlete to win four World Championship titles in the 200 m and is one of the most successful in the 100 m with three titles."]
    # random_inputs = ["Once upon a time in a far land there lived a prince"]
    random_inputs = [" James went camping in the woods, but forgot to bring a hammer to bang the tent pegs in. What else might he"]
    tokenized_inputs = tokenizer.batch_encode_plus(random_inputs, return_tensors="pt")
    print(tokenized_inputs)

    # Generate
    output_ids = model.generate(tokenized_inputs["input_ids"].cuda(), use_cache=True, max_length=256)
    print(output_ids)
    print(tokenizer.batch_decode(output_ids))
    print([tokenizer.convert_ids_to_tokens(row) for row in output_ids])

def run_perplexity_prefix_lm_varying_context_length():
    # TODO we want to check that the generate function works
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = 2048
    # model = PrefixLMLMHeadModel.from_pretrained("/home/thomaswang/code/bigscience/checkpoints/megatron/350M-pile/megatron/iter_0021000/mp_rank_00")
    # model = PrefixLMLMHeadModel.from_pretrained("/home/thomaswang/code/bigscience/checkpoints/megatron/iter_0048000/mp_rank_00")
    # model = PrefixLMLMHeadModel.from_pretrained("/home/thomaswang/code/bigscience/checkpoints/megatron/1B3-no-loss-on-targets-only/megatron/iter_0060000/mp_rank_00")
    # model = PrefixLMLMHeadModel.from_pretrained("/home/thomaswang/code/bigscience/checkpoints/megatron/1B3-pile/megatron/iter_0018000/mp_rank_00")
    model = PrefixLMLMHeadModel.from_pretrained("/home/thomaswang/code/bigscience/checkpoints/megatron/1B3-rescale-loss-per-position/megatron/iter_0021000/mp_rank_00")
    is_prefix = True
    # model = GPT2LMHeadModel.from_pretrained("/home/thomaswang/code/bigscience/checkpoints/tr3d-1B3-oscar-checkpoints/global_step30000")
    # is_prefix = False
    model.eval()
    model.cuda()

    def encode(examples):
        tokenized_inputs = tokenizer(examples["text"])
        concatenated_input_ids = sum(tokenized_inputs["input_ids"], [])
        # Now we split in chunks of size tokenizer.model_max_length and we pad.
        return {"input_ids": [
            concatenated_input_ids[i:i + tokenizer.model_max_length]
            for i in range(0, len(concatenated_input_ids), tokenizer.model_max_length)
        ]}

    def filter_empty(examples):
        return [text != "" for text in examples['text']]

    # dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:1000]")
    dataset = load_dataset("bookcorpusopen", split="train[:1]")
    print(type(dataset))

    dataset = dataset.filter(filter_empty) \
        .map(encode, batched=True, batch_size=-1, remove_columns=["text","title"]) # feed entire dataset in callback
    # HACK: gpt2 tokenizer doesn't have padding token ... we currently bypass it by passing unk token.
    tokenizer.pad_token = tokenizer.unk_token

    def build_attention_mask(max_size: int, n_context: int, device: torch.device):
        output = torch.tril(torch.ones(max_size, max_size, dtype=torch.bool, device=device))
        # bidirectional attention in context.
        output[:n_context, :n_context] = True
        return output

    def run_perplexity_prefix_lm(n_context: int):
        assert n_context > 0

        # We want to detect at what point do we obtain very pool losses
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        def collate_fn(examples):
            input_ids = [example["input_ids"] for example in examples]
            return data_collator(input_ids, return_tensors="pt")

        dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

        agg_loss = 0
        nb_items = 0
        with torch.no_grad():
            # TODO improve batching.
            for sample in tqdm(dataloader):
                input_ids, labels = sample["input_ids"], sample["labels"]

                # TODO: maybe this isn't the best
                if n_context >= input_ids.shape[-1]:
                    continue

                input_ids = input_ids.cuda()[:, :-1]
                labels = labels.cuda()[:, 1:]
                # attention_mask invariant to batch and head
                # This is not good as we collapsed unk and pad.

                not_pad_values = (input_ids != tokenizer.pad_token_id)
                if is_prefix:
                    attention_mask = build_attention_mask(input_ids.shape[-1], n_context, input_ids.device)[None, None,:, :]
                    attention_mask = not_pad_values[:, None, None, : ] * attention_mask
                else:
                    attention_mask = not_pad_values

                # output_logits = model(input_ids, attention_mask = attention_mask)['logits']
                #
                # # select only the ones that matter
                # labels_mask = (labels != -100)[:, n_context - 1:]
                # out = output_logits[:, n_context - 1:][labels_mask, :]
                # labels_out = labels[:, n_context - 1:][labels_mask]
                # loss = F.cross_entropy(out, labels_out, reduction='none')
                #
                # agg_loss += loss.sum()
                # nb_items += labels_mask.sum()

                # select only the a portion close to the limit between prefix and suffix
                end_prediction = n_context - 1 + 4
                if is_prefix:
                    attention_mask = attention_mask[:, :, :end_prediction, :end_prediction]
                else:
                    attention_mask = attention_mask[:, :end_prediction]
                output_logits = model(input_ids[:, :end_prediction], attention_mask = attention_mask)['logits']
                labels_mask = (labels != -100)[:, n_context - 1: end_prediction]
                out = output_logits[:, n_context - 1: end_prediction][labels_mask, :]
                labels_out = labels[:, n_context - 1: end_prediction][labels_mask]
                loss = F.cross_entropy(out, labels_out, reduction='none')

                agg_loss += loss.sum()
                nb_items += labels_mask.sum()

                # # Teacher forcing
                # teacher_forced_predictions = torch.argmax(output_logits[:, n_context - 1:], dim = -1)
                # label_ids = labels[:, n_context - 1:]
                # def convert_ids_to_token(ids):
                #     return [tokenizer.convert_ids_to_tokens(row) for row in ids]
                # print("hello")
                # print(convert_ids_to_token([label_ids[0]]))
                # print(convert_ids_to_token([teacher_forced_predictions[0]]))
                #
                # vocab = tokenizer.get_vocab()
                # print("hello")
                # print(torch.gather(output_logits[:, n_context-1], dim=-1, index = labels[:, n_context-1][:, None]).squeeze(-1))
                # print(f"Check `Ġthe` logit {output_logits[:, n_context-1][:, vocab['Ġthe']]}")
                # print(f"Check `Ċ` logit {output_logits[:, n_context-1][:, vocab['Ċ']]}")
                # print(torch.amax(output_logits[0, n_context - 1:], dim=-1))

        return agg_loss / nb_items

    losses = {}
    print(tokenizer.model_max_length)
    for n_context in [1, 4, 64, 128, 512, 1024, 1536, 1920, 1984, 2044]:
        loss = run_perplexity_prefix_lm(n_context)
        print(f"Context_size: {n_context}, loss: {loss}")
        losses[n_context] = loss


def main():
    # prefix_invariants()
    # prefix_lm_infer()
    prefix_lm_1B3_infer()
    # gpt2_1B3_infer()
    # run_perplexity_prefix_lm_varying_context_length()

if __name__ == "__main__":
    main()

# arc_easy,boolq,copa,headqa,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,rte,sciq,sst,triviaqa,webqs,wic,winogrande,wnli,wsc

## Code to plot the results from varying n_context and then computing loss on the first four tokens.
# import matplotlib.pyplot as plt
#
# n_contexts  = [1, 4, 64, 128, 512, 1024, 1536, 1920, 1984, 2044]
# tr6f_60k = [7.142319679260254, 7.062319755554199, 3.636333703994751, 3.5352745056152344, 3.159292221069336, 3.3594958782196045, 3.3022403717041016, 2.9234163761138916, 3.179145097732544, 3.205340623855591]
# tr3d_60k = [5.126442909240723,4.280908584594727,3.5780608654022217,3.4606239795684814,3.0849897861480713,3.295539379119873,3.213402509689331,2.786288022994995,3.0520291328430176,3.1395297050476074]
# tr3d_30k = [5.152425765991211, 4.331642150878906, 3.665862560272217, 3.5769572257995605, 3.200692892074585, 3.400550127029419, 3.318943500518799, 2.900491952896118, 3.1829869747161865, 3.254850387573242]
#
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.plot(n_contexts, tr6f_60k, 'r', label="tr6f (60k steps)")
# ax.plot(n_contexts, tr3d_60k, 'b', label="tr3d (60k steps)")
# ax.plot(n_contexts, tr3d_30k, 'g', label="tr3d (30k steps)")
# # ax.set_yscale('log')
# # ax.set_xscale('log')
# plt.legend(loc="upper right")
# fig.show()