from prettytable import PrettyTable
import json
from copy import deepcopy
import pandas as pd
import random
import numpy as np
qid2label = json.load(open('/data0/lyt/wikidata-5m/wikidata-5m-entity-en-label.json'))
rel2label = json.load(open('/data0/lyt/wikidata-5m/wikidata-rel-en-label.json'))
def relation_label(rel):
    if '-1' in rel:
        rel = rel[:-5]
    return rel2label[rel]['label']

def relation_description(rel):
    if '-1' in rel:
        rel = rel[:-5]
    return rel2label[rel]['description']

def entity_label(qid):
    if qid not in qid2label:
        return qid
    return qid2label[qid]['label']

def visual_triples(triples:list):
    x = PrettyTable()
    x.field_names = ["head", "rel", "tail"]
    x.add_rows([
        [
            qid2label[triple[0]]['label'],
            rel2label[triple[1]]['label'],
            qid2label[triple[2]]['label']
        ]
        for triple in triples
    ])
    return x

def get_prompt_string_for_triples(triples:list):
    prompt = ''
    for triple in triples:
        prompt += f"({qid2label[triple[0]]['label']}, {rel2label[triple[1]]['label']}, {qid2label[triple[2]]['label']})\n"
    return prompt

def get_prompt_string_for_description(triples:list):
    prompt = ''
    entity_set = set()
    rel_set = set()
    for triple in triples:
        entity_set.add(triple[0])
        entity_set.add(triple[2])
        rel_set.add(triple[1])
    # prompt += 'Entity Descriptions:\n'
    # for entity in entity_set:
    #     prompt += f"{qid2label[entity]['label']}: {qid2label[entity]['description']}\n"
    prompt += 'Predicate Descriptions:\n'
    for rel in rel_set:
        prompt += f"{rel2label[rel]['refined_description']}\n"
    
    return prompt



def get_chains_str(entity_list,rel_list):
    result = entity_label(entity_list[0])
    for idx,rel in enumerate(rel_list,start=1):
        if '-1' in rel:
            result += f' <- {relation_label(rel[:-5])} ({rel[:-5]}) <- '
        else:
            result += f' -> {relation_label(rel)} ({rel}) -> '
        result += f'{entity_label(entity_list[idx])} ({entity_list[idx]})'
    return result

def get_two_hop_prompt(head_entity,rel_list,placeholder_list):
    result = entity_label(head_entity)
    for rel,placeholder in zip(rel_list,placeholder_list):
        if '-1' in rel:
            result += f' <- {relation_label(rel[:-5])} ({rel[:-5]}) <- '
        else:
            result += f' -> {relation_label(rel)} ({rel}) -> '
        result += f'{placeholder}'
    return result

def visual_chains(entity_list,rel_list):
    result = get_chains_str(entity_list,rel_list)
    print(result)
    

def get_one_hop_prompt(question_topic_entity,quesiton_relation)->str:
    ret = entity_label(question_topic_entity) + ' -> '
    ret +=f'{relation_label(quesiton_relation)} ({relation_description(quesiton_relation)})'  + ' -> '
    ret += 'missing entity'
    return ret

def find_all_matching_index(element,temp:list) -> list:
    ret = []
    for idx,unit in enumerate(temp):
        if unit == element:
            ret.append(idx)
    return ret

# Python program to illustrate the intersection
# of two lists using set() method
def intersection(lst1, lst2)->list:
    return list(set(lst1) & set(lst2))


