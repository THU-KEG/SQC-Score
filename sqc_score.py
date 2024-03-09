import json
import os
import re
import copy
import torch
import argparse
import numpy as np
from tqdm import tqdm
from train import train
from menli.MENLI import MENLI
from collections import OrderedDict
from transformers import AutoModelForCausalLM, AutoTokenizer



def post_process_grading_process(text, gold_answer, test_answer):
    if "That's the end of my answer" in text:
        text = text[text.find('<|assistant|>'):]
        text = text[:text.find('That\'s the end of my answer.')]
    else:
        return None
    grading_list = list(set(text.split('\n')))
    good_match = 0
    good_match_list = []
    for grading in grading_list:
        if re.findall(r'\d+', grading) == []:
            continue
        valid = False
        for gold_point in gold_answer:
            gold_point = gold_point.replace('.', '').strip()
            gold_point = gold_point.replace('(2 points)', '')
            if gold_point in grading:
                valid = True
                break
        if not valid:
            continue
        for test_point in test_answer:
            test_point = test_point.replace('.', '').strip()
            if test_point in grading:
                if gold_point == 'none':
                    if test_point == 'none':
                        good_match += 1
                        good_match_list.append(grading)
                        break
                else:
                    if all(int(x) <= 0 for x in re.findall(r'\d+', grading)):
                        continue
                    else:
                        good_match += 1
                        good_match_list.append(grading)
                        break
    if good_match == 0:
        P = 0
        F = 0
        R = 0
    else:
        P = min(1, good_match / len(test_answer))
        R = min(1, good_match / len(gold_answer))
        F = min(1, 2 * P * R / (P + R))
    return ((P, R, F), good_match_list)


def paraphrase(triples, device='cuda'):
    with open('./prompt_paraphrase.txt', 'r') as f:
        prompt = f.read()
    
    paraphrases = [] 
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    for triple_list in triples:
        if len(triple_list) != 3:
            continue
        triple = {'h': triple_list[0], 'r': triple_list[1], 't': triple_list[2]}
        new_prompt = prompt + '\n' + str(triple)
        messages = [
            {'role': 'user', 'content': new_prompt}
        ]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device)
        generated_ids = model.generate(model_inputs, max_new_tokens=128, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        decoded = decoded[decoded.find('[/INST]') + 7:]
        decoded = decoded[:decoded.find('(')]
        paraphrases.append(decoded.strip())
    return paraphrases

def get_threshold(dataset, ratio):
    assert ratio > 0 and ratio < 1, "ratio should be in (0, 1)"
    # refs -> list of strings
    refs = [data['text'] for data in dataset]
    # hyps -> list of list of strings
    hyps = [[s.replace("(2 points)", '')]  for data in dataset for s in data["standard_answer"]]
    scores = []
    nli_scorer = MENLI(direction="rh", formula="e", nli_weight=1.0, combine_with="None", model="D")
    for ref, hyp in zip(refs, hyps):
        score = []
        ref = [copy.deepcopy(ref) for i in range(len(hyp))]
        nli_score = nli_scorer.score_all([], ref, hyp)
        score = [float(s) for s in nli_score]
        scores.append(score)
    scores = [s for score in scores for s in score]
    scores = np.array(sorted(scores))
    threshold = np.percentile(scores, 100 * (1 - ratio))
    print(f"Now NLI threshold is {threshold}")
    return threshold

def get_nli_valid(dataset, threshold):
    refs = [data['text'] for data in dataset]
    hyps = [data['student_answer'] for data in dataset]
    scores = []
    nli_scorer = MENLI(direction="rh", formula="e", nli_weight=1.0, combine_with="None", model="D")
    for ref, hyp in zip(refs, hyps):
        score = []
        ref = [copy.deepcopy(ref) for i in range(len(hyp))]
        nli_score = nli_scorer.score_all([], ref, hyp)
        score = [float(s) for s in nli_score]
        scores.append(score)
    valid = []
    for score in scores:
        cur_valid = []
        for s in score:
            cur_valid.append(s > threshold)
        valid.append(cur_valid)
    return valid    


def report(dataset):
    precisions = [data['Precision'] for data in dataset]
    recalls = [data['Recall'] for data in dataset]
    f1s = [data['F1'] for data in dataset]
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f1 = sum(f1s) / len(f1s)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')


def sqc_score(in_file, score_model, out_dir=None, do_nli=True, gold_ratio=0.4):
    os.makedirs(out_dir, exist_ok=True)
    dataset = []
    with open(in_file, 'r') as f:
        dataset = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'scored_' + os.path.basename(in_file))
    model = AutoModelForCausalLM.from_pretrained(score_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(score_model)
    for data in tqdm(dataset):
        gold = data['standard_answer']
        gold_list = paraphrase(gold)
        part_score = 2
        total_score = part_score * len(gold_list)
        gold_list = [f'{s}(2 points)' for s in gold_list]
        gold_str = '\n'.join(gold_list)
        unique_tuples = list(OrderedDict.fromkeys(tuple(item) for item in data['student_answer']))
        test = [list(item) for item in unique_tuples]
        test_list = paraphrase(test)
        test_str = '\n'.join(test_list)
        with open("./prompt.txt", "r") as f:
            prompt = f.read()
        prompt += "\n\nStandard Answer:\n" + gold_str + "\n<end-of-standard-answer>" + "\n\nStudent Answer:\n" + test_str + "\n<end-of-student-answer>" + "\n\nTotal Score:\n" + str(total_score) + " points"
        if 'tulu' in score_model.lower():
            prompt = '<|user|>\n' + prompt + '\n<|assistant|>'

        input_ids = tokenizer(prompt, return_tensors='pt')[
            'input_ids'].to(device)
        pred = model.generate(input_ids, max_new_tokens=512, do_sample=True)
        pred = tokenizer.decode(pred[0], skip_special_tokens=True)
        assert pred != [], "No grading process generated"
        results = post_process_grading_process(
            pred, gold_list, test_list)
        while results is None:
            pred = model.generate(input_ids, max_new_tokens=256, do_sample=True)
            pred = tokenizer.decode(pred[0], skip_special_tokens=True)
            results = post_process_grading_process(
                pred, gold_list, test_list)

        P, R, F = results[0]
        good_match_list = results[1]
        data['standard_answer'] = gold_list
        data['student_answer'] = test_list
        data['Precision'] = P
        data['Recall'] = R
        data['F1'] = F
        data['good_match'] = good_match_list
    if not do_nli:
        report(dataset)
        print(f"Writing to {out_path}")
        with open(out_path, 'w') as f:
            for data in dataset:
                f.write(json.dumps(data) + '\n')
        return 
    nli_threshold = get_threshold(dataset, gold_ratio)
    for data in dataset:
        if data['F1'] == 0:
            continue
        for match in data['good_match']:
            for gold_point in data['standard_answer']:
                if gold_point in match:
                    data['standard_answer'].remove(gold_point)

            for test_point in data['student_answer']:
                if test_point in match:
                    data['student_answer'].remove(test_point)
    nlid_valid = get_nli_valid(dataset, nli_threshold)
    for data, valid in zip(dataset, nlid_valid):
        cur_data_nli_valid = len([v for v in valid if v])
        gold_num = len(data['standard_answer']) + cur_data_nli_valid
        test_num = len(data['student_answer'])
        correct_num = len(data['good_match']) + cur_data_nli_valid
        P = correct_num / test_num
        R = correct_num / gold_num
        F = 2 * P * R / (P + R)
        print(f"Precision: {P}, Recall: {R}, F1: {F}")
    with open(out_path, 'w') as f:
        for data in dataset:
            f.write(json.dumps(data) + '\n')
         
         
def main():
    parser = argparse.ArgumentParser()   
    parser.add_argument('--in_file', type=str, required=True, help='input file')
    parser.add_argument('--score_model', type=str, required=True, help='score model')
    parser.add_argument('--out_dir', type=str, required=True, help='output directory')
    parser.add_argument('--do-nli', type=str, required=True, help='whether to use nli')
    args = parser.parse_args()
    sqc_score(args.in_file, args.score_model, args.out_dir, args.do_nli)
    
if __name__ == '__main__':
    main()