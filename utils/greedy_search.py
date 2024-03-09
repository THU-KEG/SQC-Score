# from app_modules.utils import greedy_search
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
import gc

# Greedy Search


def greedy_search(input_ids: torch.Tensor,
                  model: torch.nn.Module,
                  tokenizer: transformers.PreTrainedTokenizer,
                  stop_words: list,
                  max_length: int,
                  temperature: float = 1.0,
                  top_p: float = 1.0,
                  top_k: int = 25) -> str:
    generated_tokens = []
    generated_text = ""
    past_key_values = None
    current_length = 1
    for i in range(max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids)
            else:
                outputs = model(input_ids[:, -1:],
                                past_key_values=past_key_values)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # apply temperature
            logits /= temperature

            probs = torch.softmax(logits, dim=-1)
            # apply top_p
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0

            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token)

            input_ids = torch.cat((input_ids, next_token), dim=-1)

            generated_tokens.append(next_token[0].item())

            generated_text += tokenizer.decode([generated_tokens[-1]], skip_special_tokens=True)
            if any([x in generated_text for x in stop_words]):
                del past_key_values
                del logits
                del probs
                del probs_sort
                del probs_idx
                del probs_sum
                gc.collect()
                break

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def load_stop_words(model_name):
    model_name = model_name.lower()
    if 'GPT-NeoXT-Chat-Base-20B'.lower() in model_name:
        return ["<human>", "<bot>"]
    elif 'alpaca' in model_name:
        return ["</s><s>"]
    elif 'vicuna' in model_name:
        return ["</s>"]
    elif 'tulu' in model_name:
        return ["</s>"]
    else:
        return ["<eos>"]


if __name__ == "__main__":

    # device = torch.device("cuda:7")
    base_model = "/data3/MODELS/llama-65b-hf"
    model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto")
    # model.to(device)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    prompt = "The COVID-19 pandemic has caused a lot of problems for people."

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
    output = model.generate(input_ids)
    print(tokenizer.decode(output[0]))
    # print(greedy_search(input_ids, model, tokenizer, ["[|Human|]", "[|AI|]"], 100, 0.9, 0.9, 25))
