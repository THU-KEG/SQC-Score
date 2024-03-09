#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from copy import deepcopy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import numpy as np
import torch
# import transformers
from torch.utils.data import Dataset
from transformers import (
    Trainer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    HfArgumentParser,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer
)
from utils.utils import jload
import nltk
import re
import evaluate
import os
import wandb

metric = evaluate.load('rouge')
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    train_data_path: str = field(default=None, metadata={
        "help": "Path to the training data."})
    eval_data_path: str = field(default=None, metadata={
        "help": "Path to the evaluation data."})


@dataclass
class BaseTrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-
                                                num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-
                                                  num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0]
                          for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(
        strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, model_name_or_path: str):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # """
        # list_data_dict = jlload(data_path)
        # prompt_template = load_prompt(model_name_or_path, shot_num=0)
        # logging.warning("Template we use is: \n")
        # logging.warning(prompt_template)s
        # logging.warning("Formatting inputs...")
        # # prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # sources = []
        # for example in list_data_dict:
        #     prompt = deepcopy(prompt_template)
        #     prompt = prompt.replace('[[passage]]', example['passage'])
        #     prompt = prompt.replace('[[question]]', example['question'])
        #     sources.append(prompt)
        # targets = [
        #     f"{example['rationale']}{tokenizer.eos_token}" for example in list_data_dict]
        # logging.warning(
        #     "Context faithfulness dataset is loaded. we have {} examples".format(len(sources)))"""

        logging.warning("Loading data...")
        sources = []
        targets = []
        list_data_dict = jload(data_path)
        sources.extend([example['input'] for example in list_data_dict])
        targets.extend([example['output'] for example in list_data_dict])
        logging.warning(
            "Dataset is loaded. we have {} examples".format(len(sources)))

        # logging.warning("Loading alpaca data...")
        # list_data_dict = jload('data/alpaca_data.json')
        # prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        # sources.extend([
        #     prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        #     for example in list_data_dict
        # ])
        # targets.extend([f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict])
        # logging.warning("Alpaca dataset is loaded. we have {} examples".format(len(sources)))

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def make_supervised_data_module(tokenizer: PreTrainedTokenizer, data_args, model_name) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.train_data_path, model_name_or_path=model_name)
    # eval_dataset = SupervisedDataset(
    #     tokenizer=tokenizer, data_path=data_args.eval_data_path, model_name_or_path=model_name
    # )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, BaseTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # # Define compute metrics function
    # def compute_metrics(eval_preds):
    #     preds, labels = eval_preds
    #     if isinstance(preds, tuple):
    #         preds = preds[0]
    #     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    #     # Replace -100 in the labels as we can't decode them.
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #     decoded_labels = tokenizer.batch_decode(
    #         labels, skip_special_tokens=True)

    #     # Some simple post-processing
    #     decoded_preds, decoded_labels = postprocess_text(
    #         decoded_preds, decoded_labels)

    #     # result = metric.compute(predictions=decoded_preds,
    #     #                         references=decoded_labels, use_stemmer=True)
    #     result = {k: round(v * 100, 4) for k, v in result.items()}
    #     correct_count = 0
    #     for pred, label in zip(decoded_preds, decoded_labels):
    #         matches = re.findall(r'\d+', label)
    #         if matches:
    #             max_num = max(map(int, matches)) 
    #             if str(max_num) in pred:
    #                 correct_count += 1
    #     accuracy = correct_count / len(decoded_preds)
    #     result['eval_accuracy'] = accuracy
    #     return result

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, model_name=model_args.model_name_or_path)
    trainer = Trainer(model=model, tokenizer=tokenizer,
                      args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
